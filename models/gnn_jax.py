import jax
import jax.numpy as jnp
import flax.linen as nn
import jraph
from typing import Callable, Optional, Dict, Any

# Reference constants for TS Model scaling
ATOM_REF = {
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

class GTLayer(nn.Module):
    """
    Graph Transformer Layer with Edge enhancement.
    Inspired by GTConv/EGT logic.
    """
    hidden_dim: int

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        # 1. Edge Attention logic
        def update_edge_fn(edges, senders_node, receivers_node, globals_):
            # Combine edge features with end-node embeddings
            return MLP([self.hidden_dim])(jnp.concatenate([edges, senders_node, receivers_node], axis=-1))
        
        # 2. Node Update logic
        def update_node_fn(nodes, senders_node, receivers_node, edges):
            # Aggregate neighbor info weighted by edge features
            msg = MLP([self.hidden_dim])(jnp.concatenate([receivers_node, edges], axis=-1))
            return MLP([self.hidden_dim])(jnp.concatenate([nodes, msg], axis=-1))

        new_graph = jraph.GraphNetwork(
            update_node_fn=update_node_fn,
            update_edge_fn=update_edge_fn
        )(graph)
        
        # Residual and Norm
        nodes = nn.LayerNorm()(new_graph.nodes + graph.nodes)
        edges = nn.LayerNorm()(new_graph.edges + graph.edges)
        return new_graph._replace(nodes=nodes, edges=edges)

class FFBlock(nn.Module):
    """
    FF Block: Maps latent node features to physical force field parameters.
    Implements mandatory physical scaling laws.
    """
    @nn.compact
    def __call__(self, latent, atomic_numbers):
        # Predict kappa (volume scaling) and 5 Slater A coefficients
        # [kappa, A_ex, A_es, A_pol, A_disp, A_dhf]
        raw_out = MLP([64, 32, 6])(latent)
        
        kappa = nn.sigmoid(raw_out[:, 0]) * 2.0  # kappa in (0, 2)
        A_coeffs = jnp.exp(raw_out[:, 1:]) * 100.0  # Force positive Slater A
        
        # Vectorized Physical Scaling (Batch-wise if possible, here per node)
        # Note: In real JAX, we prefer vectorized ops over python loops
        def compute_node_params(ki, z):
            # Basic fallback if z not in ATOM_REF
            ref = jax.lax.switch(z.astype(jnp.int32), 
                [lambda: (12.0, 38.2, 46.6, 1350.0, 49000.0)] * 100 # Simple fallback logic
            )
            # (Note: In production, use a lookup table array for ref constants)
            return ki # Simplified for the skeleton
            
        # Output structure for DMFF Hamiltonian
        return {
            "kappa": kappa,
            "A_ex": A_coeffs[:, 0],
            "A_es": A_coeffs[:, 1],
            "A_pol": A_coeffs[:, 2],
            "A_disp": A_coeffs[:, 3],
            "A_dhf": A_coeffs[:, 4],
            "B": 35.0 * (kappa ** (-1.0/3.0)) # Universal B scaling
        }

class PhyNEO_GNN_V2(nn.Module):
    hidden_dim: int = 128
    num_layers: int = 4

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple):
        # 1. Embeddings
        node_embed = MLP([self.hidden_dim])(graph.nodes['features'])
        edge_embed = MLP([self.hidden_dim])(graph.edges)
        graph = graph._replace(nodes=node_embed, edges=edge_embed)
        
        # 2. Advanced Message Passing (GT-style)
        for _ in range(self.num_layers):
            graph = GTLayer(self.hidden_dim)(graph)
            
        # 3. Physical Readout
        params = FFBlock()(graph.nodes, graph.nodes['features'][:, 0])
        return params
