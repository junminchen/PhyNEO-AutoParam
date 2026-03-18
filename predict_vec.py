import jax
import jax.numpy as jnp
import pickle
import json
from core.molecule import Molecule
from models.gnn_jax import PhyNEO_GNN_V2
from rdkit import Chem

def predict_vec():
    # 1. SMILES for Vinyl Ethylene Carbonate (VEC)
    smiles = "C=CC1COC(=O)O1"
    
    # 2. Load production weights
    weights_path = "models/phyneo_production_v1.flax"
    with open(weights_path, "rb") as f:
        variables = pickle.load(f)
    
    # 3. Model init
    model = PhyNEO_GNN_V2(hidden_dim=128, num_layers=6)
    mol_obj = Molecule.from_smiles(smiles)
    graph = mol_obj.get_graph()
    total_q = float(Chem.GetFormalCharge(mol_obj.rdmol))
    
    # 4. Inference
    pred = model.apply(variables, graph, total_charge=total_q)
    
    # 5. Output results
    print(f"=== PhyNEO Parameter Prediction for VEC ({smiles}) ===")
    print(f"{'Idx':<4} | {'Sym':<3} | {'Q (e)':>8} | {'Kappa':>8} | {'Alpha':>8} | {'C6':>10} | {'C8':>12} | {'C10':>14}")
    print("-" * 85)
    
    atoms = mol_obj.rdmol.GetAtoms()
    for i in range(len(atoms)):
        symbol = atoms[i].GetSymbol()
        q = float(pred['q'][i])
        kappa = float(pred['kappa'][i])
        alpha = float(pred['alpha'][i])
        c6 = float(pred['c6'][i])
        c8 = float(pred['c8'][i])
        c10 = float(pred['c10'][i])
        print(f"{i:<4} | {symbol:<3} | {q:8.4f} | {kappa:8.4f} | {alpha:8.4f} | {c6:10.2f} | {c8:12.2f} | {c10:14.2f}")

if __name__ == "__main__":
    predict_vec()
