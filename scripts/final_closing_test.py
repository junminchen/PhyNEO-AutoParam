import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from models.gnn_jax import PhyNEO_GNN_V2
from core.molecule import Molecule
from training.joint_trainer import load_master_dataset, JointTrainer
from rdkit import Chem
import json

def run_closing_test():
    print("--- 1. Loading Full Dataset ---")
    dataset = load_master_dataset("data/master_dataset.json")
    
    print("--- 2. Initializing & Fast Training (50 Epochs) ---")
    trainer = JointTrainer()
    rng = jax.random.PRNGKey(42)
    sample = dataset[0]
    variables = trainer.model.init(rng, sample['graph'], total_charge=sample['total_q'])
    opt_state = trainer.optimizer.init(variables)
    
    # Simple training loop
    for epoch in range(1, 51):
        total_loss = 0.0
        for item in dataset:
            variables, opt_state, loss = trainer.train_step(
                variables, opt_state, item['graph'], item['q_ref'], item['k_ref'], 
                item['d_ref'], item['qt_ref'], item['a_ref'], item['c6_ref'], 
                item['c8_ref'], item['c10_ref'], item['total_q']
            )
            total_loss += loss
        if epoch % 10 == 0:
            print(f"  Epoch {epoch} | Loss: {total_loss/len(dataset):.6f}")

    print("\n--- 3. Final Blind Test: VEC (Vinyl Ethylene Carbonate) ---")
    smiles = "C=CC1COC(=O)O1"
    mol_obj = Molecule.from_smiles(smiles)
    graph = mol_obj.get_graph()
    total_q = float(Chem.GetFormalCharge(mol_obj.rdmol))
    pred = trainer.model.apply(variables, graph, total_charge=total_q)
    
    with open("examples/production_results/VEC/results_high.json", "r") as f:
        ref = json.load(f)
        
    metrics = {
        "Charge (e)": ("q", "charge"),
        "Polarizability": ("alpha", "alpha_iso"),
        "C6 Dispersion": ("c6", "c6_ii"),
        "C8 Dispersion": ("c8", "c8_ii"),
        "C10 Dispersion": ("c10", "c10_ii")
    }
    
    print(f"{'Metric':<15} | {'Ref (Mean)':<12} | {'Pred (Mean)':<12} | {'MAE':<10}")
    print("-" * 60)
    for label, (p_key, r_key) in metrics.items():
        y_p = pred[p_key].flatten()
        y_r = jnp.array([a[r_key] for a in ref['atoms']])
        mae = jnp.mean(jnp.abs(y_p - y_r))
        print(f"{label:<15} | {float(jnp.mean(y_r)):12.4f} | {float(jnp.mean(y_p)):12.4f} | {float(mae):10.4f}")

if __name__ == "__main__":
    run_closing_test()
