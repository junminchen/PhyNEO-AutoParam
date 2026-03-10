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
            return MLP([self.hidden_dim])(jnp.concatenate([edges, senders_node, receivers_node], axis=-1))
        def update_node_fn(nodes, senders_node, receivers_node, edges):
            msg = MLP([self.hidden_dim])(jnp.concatenate([receivers_node, edges], axis=-1))
            return MLP([self.hidden_dim])(jnp.concatenate([nodes, msg], axis=-1))
        new_graph = jraph.GraphNetwork(update_node_fn=update_node_fn, update_edge_fn=update_edge_fn)(graph)
        nodes = nn.LayerNorm()(new_graph.nodes + graph.nodes)
        edges = nn.LayerNorm()(new_graph.edges + graph.edges)
        return new_graph._replace(nodes=nodes, edges=edges)

class FFBlock(nn.Module):
    """
    Enhanced FF Block: Supports physical derivation AND machine-learned refinement for Slater B.
    """
    @nn.compact
    def __call__(self, latent, atomic_numbers):
        # [kappa, A_ex, A_es, A_pol, A_disp, A_dhf, B_refine]
        raw_out = MLP([64, 32, 7])(latent)
        
        kappa = nn.sigmoid(raw_out[:, 0]) * 2.0
        A_coeffs = jnp.exp(raw_out[:, 1:6]) * 100.0
        B_refine = nn.tanh(raw_out[:, 6]) * 0.1 # Small delta correction (-10% to +10%)
        
        # Calculate B_base from Veff-scaling logic
        # For simplicity in this JAX skeleton, we use a placeholder for vectorized atom constants
        B_base = 35.0 * (kappa ** (-1.0/3.0)) 
        B_final = B_base * (1.0 + B_refine) # Refine the B parameter in Phase 3
        
        return {
            "kappa": kappa,
            "A_ex": A_coeffs[:, 0], "A_es": A_coeffs[:, 1],
            "A_pol": A_coeffs[:, 2], "A_disp": A_coeffs[:, 3], "A_dhf": A_coeffs[:, 4],
            "B": B_final
        }

class PhyNEO_GNN_V2(nn.Module):
    hidden_dim: int = 128
    num_layers: int = 4
    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple):
        node_embed = MLP([self.hidden_dim])(graph.nodes['features'])
        edge_embed = MLP([self.hidden_dim])(graph.edges)
        graph = graph._replace(nodes=node_embed, edges=edge_embed)
        for _ in range(self.num_layers):
            graph = GTLayer(self.hidden_dim)(graph)
        params = FFBlock()(graph.nodes, graph.nodes['features'][:, 0])
        return params
