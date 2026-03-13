import jax
import jax.numpy as jnp
import pickle
import json
import os
from core.molecule import Molecule
from models.gnn_jax import PhyNEO_GNN_V2
from rdkit import Chem

def evaluate_set(test_molecules):
    # 1. Load weights
    with open("models/phyneo_production_v1.flax", "rb") as f:
        variables = pickle.load(f)
    
    model = PhyNEO_GNN_V2(hidden_dim=128, num_layers=6)
    
    print(f"=== PhyNEO Diverse Generalization Test (Full Physics) ===")
    
    for name, smiles in test_molecules.items():
        print(f"\n>>> Testing Molecule: {name} ({smiles})")
        
        # Inference
        mol_obj = Molecule.from_smiles(smiles)
        graph = mol_obj.get_graph()
        total_q = float(Chem.GetFormalCharge(mol_obj.rdmol))
        pred = model.apply(variables, graph, total_charge=total_q)
        
        # Load Reference (from production_results)
        ref_path = f"examples/production_results/{name}/results_high.json"
        if not os.path.exists(ref_path):
            print(f"  Warning: Reference for {name} not found. Skipping compare.")
            continue
            
        with open(ref_path, "r") as f:
            ref = json.load(f)
            
        # Comparison Table
        metrics = {
            "Charge": ("q", "charge"),
            "Alpha": ("alpha", "alpha_iso"),
            "C6": ("c6", "c6_ii"),
            "C8": ("c8", "c8_ii"),
            "C10": ("c10", "c10_ii")
        }
        
        print(f"{'Metric':<10} | {'Ref (Mean)':<12} | {'Pred (Mean)':<12} | {'MAE':<10}")
        print("-" * 55)
        for label, (p_key, r_key) in metrics.items():
            y_pred = pred[p_key].flatten()
            y_ref = jnp.array([a[r_key] for a in ref['atoms']])
            mae = jnp.mean(jnp.abs(y_pred - y_ref))
            print(f"{label:<10} | {float(jnp.mean(y_ref)):12.4f} | {float(jnp.mean(y_pred)):12.4f} | {float(mae):10.4f}")

if __name__ == "__main__":
    # Test on diverse functional groups
    test_set = {
        "VEC": "C=CC1COC(=O)O1",
        "DTD": "O=S1(=O)OCCO1",
        "SN": "N#CCCC#N"
    }
    evaluate_set(test_set)
