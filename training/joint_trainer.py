import jax
import jax.numpy as jnp
import optax
from models.gnn_jax import PhyNEO_GNN_V2
from core.molecule import Molecule
import json
import os
import random
from typing import Any
from functools import partial

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') # Silence RDKit warnings

_BOHR3_TO_NM3 = 0.00014818471148644427
_HARTREE_TO_KJMOL = 2625.499638
_BOHR_TO_NM = 0.052917721092
_C6_AU_TO_DMFF = _HARTREE_TO_KJMOL * (_BOHR_TO_NM ** 6)
_C8_AU_TO_DMFF = _HARTREE_TO_KJMOL * (_BOHR_TO_NM ** 8)
_C10_AU_TO_DMFF = _HARTREE_TO_KJMOL * (_BOHR_TO_NM ** 10)

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
                "smiles": entry['smiles'],
                "graph": graph,
                "q_ref": q_ref,
                "k_ref": k_ref,
                "alpha_ref": jnp.array([a.get('alpha', 0.0) for a in entry['atoms']]) * _BOHR3_TO_NM3,
                "c6_ref": jnp.array([a.get('c6', 0.0) for a in entry['atoms']]) * _C6_AU_TO_DMFF,
                "c8_ref": jnp.array([a.get('c8', 0.0) for a in entry['atoms']]) * _C8_AU_TO_DMFF,
                "c10_ref": jnp.array([a.get('c10', 0.0) for a in entry['atoms']]) * _C10_AU_TO_DMFF,
                "d_ref": d_ref,
                "qt_ref": qt_ref,
                "total_q": float(entry['total_charge']), # Strict float
                "slater_targets": slater_targets
            })
        except Exception as e:
            print(f"  Warning: Skipping {entry['name']} due to featurization error: {e}")
            
    print(f"--- Finished loading {len(processed)} molecules ---")
    return processed

class JointTrainer:
    def __init__(self, learning_rate=1e-3, target_scales=None, loss_weights=None):
        self.model = PhyNEO_GNN_V2(hidden_dim=128, num_layers=6)
        self.optimizer = optax.adam(learning_rate)
        self.target_scales = {
            "q": 1.0,
            "alpha": 1.0,
            "dipole": 1.0,
            "quadrupole": 1.0,
            "c6": 1.0,
            "c8": 1.0,
            "c10": 1.0,
        }
        if target_scales is not None:
            self.target_scales.update(target_scales)
        self.loss_weights = {
            "q": 1.0,
            "alpha": 1.0,
            "dipole": 1.0,
            "quadrupole": 1.0,
            "c6": 1.0,
            "c8": 1.0,
            "c10": 1.0,
        }
        if loss_weights is not None:
            self.loss_weights.update(loss_weights)

    def _scaled_mse(self, pred, ref, key):
        scale = self.target_scales.get(key, 1.0)
        scale = max(float(scale), 1e-12)
        return jnp.mean(((pred - ref) / scale) ** 2)

    def loss_fn(self, variables, item):
        out = self.model.apply(variables, item['graph'], total_charge=item['total_q'])
        
        alpha_ref = item.get("alpha_ref", item["k_ref"] * 0.001)
        c6_ref = item.get("c6_ref", out["c6"])
        c8_ref = item.get("c8_ref", out["c8"])
        c10_ref = item.get("c10_ref", out["c10"])

        q_loss = self._scaled_mse(out['q'], item['q_ref'], "q")
        alpha_loss = self._scaled_mse(out['kappa'] * 0.001, alpha_ref, "alpha")
        d_loss = self._scaled_mse(out['dipole'], item['d_ref'], "dipole")
        qt_loss = self._scaled_mse(out['quadrupole'], item['qt_ref'], "quadrupole")
        c6_loss = self._scaled_mse(out['c6'], c6_ref, "c6")
        c8_loss = self._scaled_mse(out['c8'], c8_ref, "c8")
        c10_loss = self._scaled_mse(out['c10'], c10_ref, "c10")

        loss = (
            self.loss_weights["q"] * q_loss
            + self.loss_weights["alpha"] * alpha_loss
            + self.loss_weights["dipole"] * d_loss
            + self.loss_weights["quadrupole"] * qt_loss
            + self.loss_weights["c6"] * c6_loss
            + self.loss_weights["c8"] * c8_loss
            + self.loss_weights["c10"] * c10_loss
        )
        
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
    def train_step(
        self,
        variables,
        opt_state,
        graph,
        q_ref,
        alpha_ref,
        d_ref,
        qt_ref,
        c6_ref,
        c8_ref,
        c10_ref,
        total_q,
        slater_targets=None,
    ):
        def local_loss(vars):
            out = self.model.apply(vars, graph, total_charge=total_q)
            
            q_loss = self._scaled_mse(out['q'], q_ref, "q")
            alpha_loss = self._scaled_mse(out['kappa'] * 0.001, alpha_ref, "alpha")
            d_loss = self._scaled_mse(out['dipole'], d_ref, "dipole")
            qt_loss = self._scaled_mse(out['quadrupole'], qt_ref, "quadrupole")
            c6_loss = self._scaled_mse(out['c6'], c6_ref, "c6")
            c8_loss = self._scaled_mse(out['c8'], c8_ref, "c8")
            c10_loss = self._scaled_mse(out['c10'], c10_ref, "c10")

            loss = (
                self.loss_weights["q"] * q_loss
                + self.loss_weights["alpha"] * alpha_loss
                + self.loss_weights["dipole"] * d_loss
                + self.loss_weights["quadrupole"] * qt_loss
                + self.loss_weights["c6"] * c6_loss
                + self.loss_weights["c8"] * c8_loss
                + self.loss_weights["c10"] * c10_loss
            )
            
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
                item['graph'], item['q_ref'], item['alpha_ref'], 
                item['d_ref'], item['qt_ref'],
                item['c6_ref'], item['c8_ref'], item['c10_ref'],
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
