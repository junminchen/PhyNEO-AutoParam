import jax
import jax.numpy as jnp
import flax.linen as nn
import jraph
from typing import Callable, Optional

# Reference constants for TS Model scaling (must match physics/multipole_engine.py)
ATOM_REF = {
    # element: (alpha_free, vol_free, c6_free, c8_free, c10_free)
    1:  (4.5, 14.1, 6.5, 124.4, 3286.0),    # H
    6:  (12.0, 38.2, 46.6, 1350.0, 49000.0), # C
    7:  (7.4, 25.4, 24.2, 760.0, 26000.0),  # N
    8:  (5.4, 18.1, 15.6, 540.0, 18000.0),  # O
    9:  (3.8, 12.1, 9.5, 320.0, 10000.0),   # F
    16: (19.6, 75.2, 134.0, 4000.0, 120000.0),# S
    17: (15.0, 59.8, 94.6, 3200.0, 100000.0), # Cl
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

class PhyReadout(nn.Module):
    """
    Physics-Informed Readout Layer.
    Maps GNN latent features to physical FF parameters using TS-scaling laws.
    """
    @nn.compact
    def __call__(self, latent, atomic_numbers):
        # 1. Predict raw scaling factor kappa and Slater A components
        # kappa is roughly V_eff / V_free, usually around 0.5 - 1.5
        res = MLP([64, 32, 6])(latent) # [kappa, A_ex, A_es, A_pol, A_disp, A_dhf]
        
        kappa = nn.sigmoid(res[:, 0]) * 2.0  # Constrain kappa to (0, 2)
        A_raw = jnp.exp(res[:, 1:]) * 100.0 # Slater A is positive and usually large
        
        # 2. Apply Physical Scaling Laws
        params = []
        for i, z in enumerate(atomic_numbers):
            z_val = int(z)
            ref = ATOM_REF.get(z_val, ATOM_REF[6]) # Fallback to Carbon
            a_free, v_free, c6_f, c8_f, c10_f = ref
            ki = kappa[i]
            
            # TS Scaling Laws
            alpha = a_free * ki
            c6 = c6_f * (ki ** 2.0)
            c8 = c8_f * (ki ** (8.0/3.0))
            c10 = c10_f * (ki ** (10.0/3.0))
            
            # Slater B scaling: B prop to V^-1/3
            # B_ref is roughly 35.0-40.0 for typical atoms
            slater_b = 35.0 * (ki ** (-1.0/3.0))
            
            params.append({
                "alpha": alpha,
                "c6": c6, "c8": c8, "c10": c10,
                "slater_b": slater_b,
                "A_ex": A_raw[i, 0], "A_es": A_raw[i, 1],
                "A_pol": A_raw[i, 2], "A_disp": A_raw[i, 3], "A_dhf": A_raw[i, 4]
            })
            
        return params

class PhyNEOGNN(nn.Module):
    """
    PhyNEO GNN Architecture in JAX.
    Combines Message Passing with Physics-Informed Readout.
    """
    hidden_dim: int = 128
    num_layers: int = 3

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple):
        # 1. Node Embedding
        x = nn.Embed(100, self.hidden_dim)(graph.nodes['atomic_numbers'])
        
        # 2. Message Passing (Simple Graph Convolution for start)
        for _ in range(self.num_layers):
            # Update function for nodes
            def update_node_fn(nodes, senders_node, receivers_node, edges):
                return MLP([self.hidden_dim])(jnp.concatenate([nodes, receivers_node], axis=-1))
            
            graph = jraph.GraphConvolution(update_node_fn)(graph)
            x = graph.nodes
            x = nn.LayerNorm()(x)
            x = nn.silu(x)
            graph = graph._replace(nodes=x)

        # 3. Physics-Informed Readout
        params = PhyReadout()(x, graph.nodes['atomic_numbers'])
        return params
