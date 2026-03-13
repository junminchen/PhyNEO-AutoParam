import json
import pickle
import os
import numpy as np
from pathlib import Path
import argparse
from rdkit import Chem

from data.mol_smiles import phyneo_name_mapped_smiles as SMILES_MAP

def get_param(param_obj, index):
    """Helper to extract parameter whether it is a scalar or array."""
    if hasattr(param_obj, "__len__") and not isinstance(param_obj, (str, bytes)):
        return float(param_obj[index])
    return float(param_obj)

def distill_data(results_dir, output_file):
    """
    Fully automatic data distillation. Scans directory and extracts SMILES 
    mapping dynamically from the data source.
    """
    from data.mol_smiles import phyneo_name_mapped_smiles
    from data.fragment_smiles import fragment_bank
    from data.electrolyte_1000 import electrolyte_bank_1000
    
    # Unified reference for mapping names back to SMILES
    FULL_BANK = {**phyneo_name_mapped_smiles, **fragment_bank, **electrolyte_bank_1000}
    
    master_data = []
    results_dir = Path(results_dir)
    print(f"--- Starting Full Data Distillation from {results_dir} ---")

    for mol_dir in results_dir.iterdir():
        if not mol_dir.is_dir(): continue
        
        json_path = mol_dir / "results_high.json"
        if not json_path.exists(): continue
            
        mol_name = mol_dir.name
        # Dynamically retrieve SMILES from our banks
        smiles = FULL_BANK.get(mol_name)
        
        if not smiles:
            print(f"Skipping {mol_name}: Metadata not found in banks.")
            continue

        try:
            with open(json_path, 'r') as f:
                phys_data = json.load(f)
            
            mol_entry = {
                "name": mol_name,
                "smiles": smiles,
                "total_charge": float(Chem.GetFormalCharge(Chem.MolFromSmiles(smiles))),
                "atoms": []
            }
            
            for i, atom in enumerate(phys_data['atoms']):
                atom_entry = {
                    "symbol": atom['element'],
                    "q_ref": atom['charge'],
                    "dipole": atom.get('dipole', [0,0,0]),
                    "quadrupole": atom.get('quadrupole', [[0,0,0],[0,0,0],[0,0,0]]),
                    "v_eff": atom['volume_eff'],
                    "alpha": atom.get('alpha_iso', 0.0),
                    "c6": atom.get('c6_ii', 0.0),
                    "c8": atom.get('c8_ii', 0.0),
                    "c10": atom.get('c10_ii', 0.0)
                }
                mol_entry["atoms"].append(atom_entry)
            master_data.append(mol_entry)
        except Exception as e:
            print(f"Error processing {mol_name}: {e}")
            
    with open(output_file, 'w') as f:
        json.dump(master_data, f, indent=4)
    print(f"\nSuccess! Fully distilled {len(master_data)} molecules into {output_file}")
        
    # Save the consolidated dataset
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(master_data, f, indent=4)
        
    print(f"\nSuccess! Distilled data for {len(master_data)} molecules saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill and align physical/fitted force field parameters.")
    parser.add_argument("-i", "--input", default="examples/production_results", help="Directory containing molecular result folders")
    parser.add_argument("-o", "--output", default="data/master_dataset.json", help="Output consolidated JSON")
    
    args = parser.parse_args()
    distill_data(args.input, args.output)
