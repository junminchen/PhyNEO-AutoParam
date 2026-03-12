import jax
import jax.numpy as jnp
from core.molecule import Molecule
from models.gnn_jax import PhyNEO_GNN_V2
import argparse
from rdkit import Chem

from scripts.params_to_xml import generate_dmff_xml
import os

def predict_parameters(smiles_list, weights_path=None):
    # 1. Initialize Model
    model = PhyNEO_GNN_V2(hidden_dim=128, num_layers=4)
    rng = jax.random.PRNGKey(42)
    
    print(f"=== PhyNEO-AutoParam Inference Engine ===")
    
    for smiles in smiles_list:
        try:
            mol_obj = Molecule.from_smiles(smiles)
            graph = mol_obj.get_graph()
            total_q = float(Chem.GetFormalCharge(mol_obj.rdmol))
            
            # Init & Inference
            variables = model.init(rng, graph, total_charge=total_q)
            params = model.apply(variables, graph, total_charge=total_q)
            
            print(f"\nMolecule: {smiles}")
            
            # Generate XML & PDB
            safe_name = smiles.replace("/", "_").replace("(", "_").replace(")", "_")
            out_dir = f"results_inference/{safe_name}"
            os.makedirs(out_dir, exist_ok=True)
            
            # 1. Save FF.xml
            xml_path = os.path.join(out_dir, "FF.xml")
            generate_dmff_xml(mol_obj.rdmol, params, xml_path)
            
            # 2. Save PDB (Align atom names with XML)
            pdb_path = os.path.join(out_dir, "structure.pdb")
            # Set atom names in RDKit molecule before writing
            for i, atom in enumerate(mol_obj.rdmol.GetAtoms()):
                atom_name = f"{atom.GetSymbol()}{i}"
                # PDB atom names are max 4 chars, handle carefully if needed
                atom.GetPDBResidueInfo().SetName(f"{atom_name:<4}")
                atom.GetPDBResidueInfo().SetResidueName("MOL")
            
            Chem.MolToPDBFile(mol_obj.rdmol, pdb_path)
            print(f"PDB Structure saved to: {pdb_path}")
            
        except Exception as e:
            print(f"Error processing {smiles}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict force field parameters from SMILES.")
    parser.add_argument("smiles", nargs="+", help="One or more SMILES strings")
    # parser.add_argument("--weights", help="Path to trained flax weights")
    
    args = parser.parse_args()
    predict_parameters(args.smiles)
