import jax
import jax.numpy as jnp
import flax.linen as nn
import jraph
from typing import Callable, Optional, Dict, Any

# Reference constants for TS Model scaling (Updated for Li-Cl, Na, Si)
ATOM_REF = {
    1:  (4.5, 14.1, 6.5, 124.4, 3286.0, 40.0),    # H
    3:  (164.0, 110.0, 1390.0, 40000.0, 800000.0, 30.0), # Li
    4:  (37.7, 65.0, 214.0, 8000.0, 150000.0, 32.0), # Be
    5:  (20.5, 45.0, 99.5, 3500.0, 80000.0, 34.0),  # B
    6:  (12.0, 38.2, 46.6, 1350.0, 49000.0, 35.0), # C
    7:  (7.4, 25.4, 24.2, 760.0, 26000.0, 36.0),  # N
    8:  (5.4, 18.1, 15.6, 540.0, 18000.0, 37.0),  # O
    9:  (3.8, 12.1, 9.5, 320.0, 10000.0, 38.0),   # F
    11: (162.7, 150.0, 1550.0, 50000.0, 900000.0, 28.0), # Na
    12: (71.2, 105.0, 627.0, 25000.0, 500000.0, 30.0), # Mg
    13: (57.8, 90.0, 528.0, 20000.0, 400000.0, 31.0), # Al
    14: (37.3, 75.0, 305.0, 12000.0, 250000.0, 32.0), # Si
    15: (25.0, 65.0, 185.0, 7000.0, 150000.0, 33.0),  # P
    16: (19.6, 75.2, 134.0, 4000.0, 120000.0, 34.0),# S
    17: (15.0, 59.8, 94.6, 3200.0, 100000.0, 35.0), # Cl
}

# Pauling Electronegativity Priors
ELECTRONEGATIVITY_PRIORS = {
    1: 2.20,   # H
    3: 0.98,   # Li
    5: 2.04,   # B
    6: 2.55,   # C
    7: 3.04,   # N
    8: 3.44,   # O
    9: 3.98,   # F
    11: 0.93,  # Na
    15: 2.19,  # P
    16: 2.58,  # S
    17: 3.16,  # Cl
}

class MLP(nn.Module):
    features: list[int]
    activation: Callable = nn.silu

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = self.activation(x)
        x = nn.Dense(self.features[-1])(x)
        return x

class GTLayer(nn.Module):
    hidden_dim: int
    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        def update_edge_fn(edges, senders_node, receivers_node, globals_):
            x = jnp.concatenate([edges, senders_node, receivers_node], axis=-1)
            return MLP([self.hidden_dim])(x)
            
        def update_node_fn(nodes, sent_edges, received_edges, globals_):
            x = jnp.concatenate([nodes, received_edges], axis=-1)
            return MLP([self.hidden_dim])(x)
            
        new_graph = jraph.GraphNetwork(
            update_node_fn=update_node_fn, 
            update_edge_fn=update_edge_fn
        )(graph)
        
        nodes = nn.LayerNorm()(new_graph.nodes + graph.nodes)
        edges = nn.LayerNorm()(new_graph.edges + graph.edges)
        return new_graph._replace(nodes=nodes, edges=edges)

class QEqSolver(nn.Module):
    @nn.compact
    def __call__(self, chi, eta, total_charge=0.0):
        eta = jnp.maximum(eta, 1e-4)
        inv_eta = 1.0 / eta
        num = total_charge + jnp.sum(chi * inv_eta)
        den = jnp.sum(inv_eta)
        lambda_val = num / den
        q_final = (lambda_val - chi) * inv_eta
        return q_final

_CHI_PRIORS_ARRAY = jnp.zeros(118).at[jnp.array([1, 3, 5, 6, 7, 8, 9, 11, 15, 16, 17])].set(
    jnp.array([2.20, 0.98, 2.04, 2.55, 3.04, 3.44, 3.98, 0.93, 2.19, 2.58, 3.16])
)

class ChargeVolume(nn.Module):
    @nn.compact
    def __call__(self, latent, atomic_numbers, total_charge=0.0):
        res = MLP([64, 32, 3])(latent)
        priors = _CHI_PRIORS_ARRAY[atomic_numbers.astype(jnp.int32)]
        chi = priors + res[:, 0]
        eta = jnp.exp(res[:, 1]) 
        kappa = nn.sigmoid(res[:, 2]) * 2.0
        q = QEqSolver()(chi, eta, total_charge)
        return q, kappa, chi, eta

class Delta3DBlock(nn.Module):
    @nn.compact
    def __call__(self, node_latent, coords=None):
        if coords is None:
            return None
        delta_params = MLP([32, 2])(node_latent)
        dq = delta_params[:, 0] * 0.05
        dk = delta_params[:, 1] * 0.02
        return {"dq": dq, "dk": dk}

class FFBlock(nn.Module):
    @nn.compact
    def __call__(self, node_latent, edge_latent, atomic_numbers, total_charge=0.0, coords=None):
        q_base, kappa_base, chi, eta = ChargeVolume()(node_latent, atomic_numbers, total_charge)
        multipoles = MLP([64, 32, 9])(node_latent)
        dipole = multipoles[:, 0:3] * 0.1
        quadrupole = multipoles[:, 3:9] * 0.01

        # 3. Dispersion & Polarizability Scaling (Physics-Consistent with ATOM_REF Priors)
        scaling_params = MLP([64, 32, 2], name="scaling_head")(node_latent)
        ratio_alpha = nn.softplus(scaling_params[:, 0]) * kappa_base
        ratio_disp = nn.softplus(scaling_params[:, 1]) * kappa_base
        
        # Pre-process ATOM_REF into a JAX array for efficient lookup
        # indices are atomic numbers, shape [118, 6]
        ref_vals = jnp.zeros((118, 5))
        for z, vals in ATOM_REF.items():
            if z < 118:
                ref_vals = ref_vals.at[z, :5].set(jnp.array(vals[:5]))
        
        priors = ref_vals[atomic_numbers.astype(jnp.int32)]
        alpha = ratio_alpha * priors[:, 0]
        c6 = (ratio_disp ** 2.0) * priors[:, 1]
        c8 = (ratio_disp ** 2.66) * priors[:, 2]
        c10 = (ratio_disp ** 3.33) * priors[:, 3]

        sapt_params = MLP([64, 32, 6])(node_latent)
        A_coeffs = jnp.exp(sapt_params[:, :5]) * 100.0
        B_base = 35.0 * (kappa_base ** (-1.0/3.0))
        B_final = B_base * (1.0 + nn.tanh(sapt_params[:, 5]) * 0.1)

        delta = Delta3DBlock()(node_latent, coords)
        if delta is not None:
            q_final = q_base + delta['dq']
            kappa_final = kappa_base + delta['dk']
            q_final = q_final - jnp.mean(q_final) + (total_charge / q_final.shape[0])
        else:
            q_final, kappa_final = q_base, kappa_base

        if edge_latent.shape[0] > 0:
            edge_params = MLP([64, 32, 6])(edge_latent)
            b0, kb, vn = nn.sigmoid(edge_params[:, 0]) * 1.5 + 1.0, jnp.exp(edge_params[:, 1]) * 100.0, nn.softplus(edge_params[:, 2:6]) * 2.0
        else:
            b0, kb, vn = jnp.zeros((0,)), jnp.zeros((0,)), jnp.zeros((0, 4))

        return {
            "q": q_final, "kappa": kappa_final, "chi": chi, "eta": eta,
            "alpha": alpha, "dipole": dipole, "quadrupole": quadrupole,
            "c6": c6, "c8": c8, "c10": c10,
            "A_ex": A_coeffs[:, 0], "A_es": A_coeffs[:, 1],
            "A_pol": A_coeffs[:, 2], "A_disp": A_coeffs[:, 3], "A_dhf": A_coeffs[:, 4],
            "B": B_final, "b0": b0, "kb": kb, "vn": vn
        }

class PhyNEO_GNN_V2(nn.Module):
    hidden_dim: int = 128
    num_layers: int = 6
    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, total_charge: float = 0.0, coords: jnp.ndarray = None):
        scaled_nodes = graph.nodes.at[:, 0].set(graph.nodes[:, 0] / 10.0)
        scaled_nodes = scaled_nodes.at[:, 1].set(scaled_nodes[:, 1] / 4.0)
        node_embed = MLP([self.hidden_dim])(scaled_nodes)
        node_embed = nn.LayerNorm()(node_embed)
        if graph.edges is not None and graph.edges.shape[0] > 0:
            edge_embed = MLP([self.hidden_dim])(graph.edges)
            edge_embed = nn.LayerNorm()(edge_embed)
        else:
            edge_embed = jnp.zeros((0, self.hidden_dim))
        current_graph = graph._replace(nodes=node_embed, edges=edge_embed)
        if graph.edges is not None and graph.edges.shape[0] > 0:
            for i in range(self.num_layers):
                current_graph = GTLayer(self.hidden_dim, name=f"gt_layer_{i}")(current_graph)
        params = FFBlock()(current_graph.nodes, current_graph.edges, graph.nodes[:, 0], total_charge, coords)
        return params
