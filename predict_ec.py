import jax
import jax.numpy as jnp
import pickle
import json
from core.molecule import Molecule
from models.gnn_jax import PhyNEO_GNN_V2
from rdkit import Chem
import flax.linen as nn

def predict_ec():
    smiles = "[O:1]=[C:2]1[O:3][C:4]([H:7])([H:8])[C:5]([H:9])([H:10])[O:6]1"
    weights_path = "models/phyneo_production_v1.flax"
    with open(weights_path, "rb") as f:
        variables = pickle.load(f)
    
    model = PhyNEO_GNN_V2(hidden_dim=128, num_layers=6)
    mol_obj = Molecule.from_smiles(smiles)
    graph = mol_obj.get_graph()
    total_q = float(Chem.GetFormalCharge(mol_obj.rdmol))
    
    pred = model.apply(variables, graph, total_charge=total_q)
    
    # Check scaling_head outputs directly
    # To do this, we'll need to run a partial apply or inspect the model
    # For now, let's just see what's in 'pred'
    print(f"=== PhyNEO Parameter Prediction for EC ({smiles}) ===")
    print(f"{'Idx':<4} | {'Sym':<3} | {'Q (e)':>8} | {'Kappa':>8} | {'Alpha':>8} | {'C6':>10}")
    print("-" * 65)
    
    atoms = mol_obj.rdmol.GetAtoms()
    for i in range(len(atoms)):
        symbol = atoms[i].GetSymbol()
        q = float(pred['q'][i])
        kappa = float(pred['kappa'][i])
        alpha = float(pred['alpha'][i])
        c6 = float(pred['c6'][i])
        print(f"{i:<4} | {symbol:<3} | {q:8.4f} | {kappa:8.4f} | {alpha:8.4f} | {c6:10.4f}")

if __name__ == "__main__":
    predict_ec()
