import jax
import jax.numpy as jnp
import pickle
import json
from core.molecule import Molecule
from models.gnn_jax import PhyNEO_GNN_V2
from rdkit import Chem

def evaluate_vec():
    # 1. Load Trained Weights
    weights_path = "models/phyneo_production_v1.flax"
    try:
        with open(weights_path, "rb") as f:
            variables = pickle.load(f)
        print(f"--- Successfully loaded weights from {weights_path} ---")
    except Exception as e:
        print(f"Error loading weights: {e}. Using current state.")
        return

    # 2. Prepare VEC Molecule
    smiles = "C=CC1COC(=O)O1"
    mol_obj = Molecule.from_smiles(smiles)
    graph = mol_obj.get_graph()
    total_q = float(Chem.GetFormalCharge(mol_obj.rdmol))
    
    # 3. Model Inference
    model = PhyNEO_GNN_V2(hidden_dim=128, num_layers=6)
    params = model.apply(variables, graph, total_charge=total_q)
    
    # 4. Load Reference 真值 (from results/VEC calculated earlier)
    ref_path = "results/VEC/results_high.json"
    # Note: If results/VEC moved to production_results, try that path:
    import os
    if not os.path.exists(ref_path):
        ref_path = "examples/production_results/VEC/results_high.json"
        
    with open(ref_path, "r") as f:
        ref_data = json.load(f)
    
    # 5. Contrast Analysis
    print(f"\n=== Final Production Evaluation: VEC Molecule ===")
    print(f"{'Atom':<10} | {'Q_Ref':<10} | {'Q_Pred':<10} | {'Error':<10}")
    print("-" * 50)
    
    atoms = mol_obj.rdmol.GetAtoms()
    errors = []
    for i, atom in enumerate(atoms):
        q_ref = ref_data['atoms'][i]['charge']
        q_pred = params['q'][i]
        err = abs(q_ref - q_pred)
        errors.append(err)
        print(f"{atom.GetSymbol()}{i:<8} | {q_ref:10.4f} | {float(q_pred):10.4f} | {float(err):10.4f}")
    
    print("-" * 50)
    print(f"Mean Absolute Error (MAE): {sum(errors)/len(errors):.6f} e")
    
    # Check long-range parameters (Sample Kappa/alpha)
    k_ref = ref_data['atoms'][0]['volume_eff'] / 20.0
    k_pred = params['kappa'][0]
    print(f"Kappa Prediction (Atom 0): Ref={k_ref:.4f}, Pred={float(k_pred):.4f}")

if __name__ == "__main__":
    evaluate_vec()
