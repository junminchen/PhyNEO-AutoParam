import jax
import jax.numpy as jnp
import pickle
import json
from core.molecule import Molecule
from models.gnn_jax import PhyNEO_GNN_V2
from rdkit import Chem

def evaluate_full():
    # 1. Load weights
    with open("models/phyneo_production_v1.flax", "rb") as f:
        variables = pickle.load(f)
    
    # 2. Prepare VEC
    smiles = "C=CC1COC(=O)O1"
    mol_obj = Molecule.from_smiles(smiles)
    graph = mol_obj.get_graph()
    total_q = float(Chem.GetFormalCharge(mol_obj.rdmol))
    
    # 3. Predict
    model = PhyNEO_GNN_V2(hidden_dim=128, num_layers=6)
    pred = model.apply(variables, graph, total_charge=total_q)
    
    # 4. Load Ref
    with open("examples/production_results/VEC/results_high.json", "r") as f:
        ref = json.load(f)
    
    print(f"=== VEC FULL PHYSICS EVALUATION (PhyNEO Production v1) ===")
    
    metrics = {
        "Charge (e)": ("q", "charge"),
        "Polarizability": ("alpha", "alpha_iso"),
        "C6 Dispersion": ("c6", "c6_ii"),
        "C8 Dispersion": ("c8", "c8_ii"),
        "C10 Dispersion": ("c10", "c10_ii")
    }
    
    for label, (p_key, r_key) in metrics.items():
        y_pred = pred[p_key].flatten()
        y_ref = jnp.array([a[r_key] for a in ref['atoms']])
        mae = jnp.mean(jnp.abs(y_pred - y_ref))
        print(f"\n{label}:")
        print(f"  - MAE: {mae:.6f}")
        # Print sample for atom 5 (Carbonyl Carbon)
        print(f"  - Sample (Atom 5): Ref={float(y_ref[5]):.4f}, Pred={float(y_pred[5]):.4f}")

if __name__ == "__main__":
    evaluate_full()
