import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from models.gnn_jax import PhyNEO_GNN_V2
from core.molecule import Molecule
import jraph
import json
import os
import random
from typing import Dict, Any, List
from functools import partial

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') # Silence RDKit warnings

def load_master_dataset(path: str):
    with open(path, 'r') as f:
        data = json.load(f)
    
    print(f"--- Loading and Featurizing {len(data)} molecules ---")
    processed = []
    for i, entry in enumerate(data):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(data)} molecules processed...")
            
        try:
            mol = Molecule.from_smiles(entry['smiles'])
            graph = mol.get_graph()
            
            # 1. Physical Anchors (Labels)
            q_ref = jnp.array([a['q_ref'] for a in entry['atoms']])
            k_ref = jnp.array([a['v_eff'] / 20.0 for a in entry['atoms']])
            
            # 2. Multipoles
            d_ref = jnp.array([a.get('dipole', [0,0,0]) for a in entry['atoms']])
            
            def flatten_quad(q):
                if isinstance(q, list) and len(q) == 3: # 3x3 matrix
                    return [q[0][0], q[0][1], q[1][1], q[0][2], q[1][2], q[2][2]]
                return [0.0] * 6
            
            qt_ref = jnp.array([flatten_quad(a.get('quadrupole', [])) for a in entry['atoms']])

            # 3. Slater Labels
            has_slater = "A_ex" in entry['atoms'][0]
            slater_targets = {
                "A_ex": jnp.array([a['A_ex'] for a in entry['atoms']]),
                "A_es": jnp.array([a['A_es'] for a in entry['atoms']]),
                "A_pol": jnp.array([a['A_pol'] for a in entry['atoms']]),
                "A_disp": jnp.array([a['A_disp'] for a in entry['atoms']]),
                "A_dhf": jnp.array([a['A_dhf'] for a in entry['atoms']]),
                "B": jnp.array([a['B_fitted'] for a in entry['atoms']])
            } if has_slater else None

            processed.append({
                "name": entry['name'],
                "graph": graph,
                "q_ref": q_ref,
                "k_ref": k_ref,
                "d_ref": d_ref,
                "qt_ref": qt_ref,
                "total_q": float(entry['total_charge']), # Strict float
                "slater_targets": slater_targets
            })
        except Exception as e:
            print(f"  Warning: Skipping {entry['name']} due to featurization error: {e}")
            
    print(f"--- Finished loading {len(processed)} molecules ---")
    return processed

from functools import partial

class JointTrainer:
    def __init__(self, learning_rate=1e-3):
        self.model = PhyNEO_GNN_V2(hidden_dim=128, num_layers=6)
        self.optimizer = optax.adam(learning_rate)

    def loss_fn(self, variables, item):
        out = self.model.apply(variables, item['graph'], total_charge=item['total_q'])
        
        # 1. Physical Anchor Loss (MBIS/Kappa)
        q_loss = jnp.mean((out['q'] - item['q_ref'])**2)
        k_loss = jnp.mean((out['kappa'] - item['k_ref'])**2)
        
        loss = 1.0 * q_loss + 0.1 * k_loss
        
        # 2. Slater Parameter Loss
        if item['slater_targets'] is not None:
            t = item['slater_targets']
            # MSE on A coefficients (log scale for robustness)
            a_loss = jnp.mean((jnp.log(out['A_ex']) - jnp.log(t['A_ex']))**2) + \
                     jnp.mean((jnp.log(out['A_es']) - jnp.log(t['A_es']))**2)
            b_loss = jnp.mean((out['B'] - t['B'])**2)
            loss += 0.5 * a_loss + 1.0 * b_loss
            
        return loss

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, variables, opt_state, graph, q_ref, k_ref, d_ref, qt_ref, total_q, slater_targets=None):
        def local_loss(vars):
            out = self.model.apply(vars, graph, total_charge=total_q)
            
            # 1. Physics Base Loss (Charges & Volume)
            q_loss = jnp.mean((out['q'] - q_ref)**2)
            k_loss = jnp.mean((out['kappa'] - k_ref)**2)
            
            # 2. Multipole Loss
            d_loss = jnp.mean((out['dipole'] - d_ref)**2)
            qt_loss = jnp.mean((out['quadrupole'] - qt_ref)**2)
            
            loss = 1.0 * q_loss + 0.1 * k_loss + 0.5 * d_loss + 0.5 * qt_loss
            
            if slater_targets is not None:
                # Log-scale MSE for A coefficients
                a_loss = jnp.mean((jnp.log(out['A_ex']) - jnp.log(slater_targets['A_ex']))**2) + \
                         jnp.mean((jnp.log(out['A_es']) - jnp.log(slater_targets['A_es']))**2)
                b_loss = jnp.mean((out['B'] - slater_targets['B'])**2)
                loss += 0.5 * a_loss + 1.0 * b_loss
            return loss

        val_grad_fn = jax.value_and_grad(local_loss)
        loss, grads = val_grad_fn(variables)
        updates, new_opt_state = self.optimizer.update(grads, opt_state)
        new_variables = optax.apply_updates(variables, updates)
        return new_variables, new_opt_state, loss

def run_joint_training(dataset_path: str):
    dataset = load_master_dataset(dataset_path)
    trainer = JointTrainer()
    
    # 1. Advanced LR Scheduler: Cosine Decay
    total_steps = len(dataset) * 5000
    lr_schedule = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=total_steps, alpha=1e-5)
    trainer.optimizer = optax.adam(learning_rate=lr_schedule)
    
    # Init
    rng = jax.random.PRNGKey(42)
    sample = dataset[0]
    variables = trainer.model.init(rng, sample['graph'], total_charge=sample['total_q'])
    opt_state = trainer.optimizer.init(variables)
    
    print(f"=== Starting Production Training Phase [GPU Enabled] ===")
    print(f"Dataset Size: {len(dataset)} molecules")
    print(f"Target Epochs: 5000")
    
    best_loss = 1e10
    
    for epoch in range(1, 5001):
        total_loss = 0.0
        # Shuffle dataset each epoch for better generalization
        random.shuffle(dataset)
        
        for item in dataset:
            variables, opt_state, loss = trainer.train_step(
                variables, opt_state, 
                item['graph'], item['q_ref'], item['k_ref'], 
                item['d_ref'], item['qt_ref'],
                item['total_q'], item['slater_targets']
            )
            total_loss += loss
        
        if epoch % 500 == 0 or epoch == 1:
            avg_loss = total_loss / len(dataset)
            print(f"Epoch {epoch:5d} | Avg Loss: {avg_loss:.8f}")
            
            # Simple weight saving
            if avg_loss < best_loss:
                best_loss = avg_loss
                os.makedirs("models", exist_ok=True)
                with open("models/phyneo_production_v1.flax", "wb") as f:
                    import pickle
                    pickle.dump(variables, f)
    
    print(f"Production training finished. Best Avg Loss: {best_loss:.8f}")

if __name__ == "__main__":
    print("--- PhyNEO Joint Trainer Script Started ---", flush=True)
    run_joint_training("data/master_dataset.json")
