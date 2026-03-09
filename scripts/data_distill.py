import json
import pickle
import os
import numpy as np
from pathlib import Path
import argparse

def distill_data(results_dir, output_file):
    """
    Scans molecular result directories and aligns physical anchors with fitted parameters.
    """
    master_data = []
    results_dir = Path(results_dir)
    
    print(f"--- Starting Data Distillation from {results_dir} ---")
    
    if not results_dir.exists():
        print(f"Error: Directory {results_dir} does not exist.")
        return

    # Iterate through each molecule directory
    for mol_dir in results_dir.iterdir():
        if not mol_dir.is_dir():
            continue
            
        json_path = mol_dir / "results_high.json"
        pickle_path = mol_dir / "latest.pickle"
        
        # Check if results exist
        if not (json_path.exists() and pickle_path.exists()):
            # print(f"Skipping {mol_dir.name}: Missing JSON or Pickle results.")
            continue
            
        print(f"Processing {mol_dir.name}...")
        
        try:
            # 1. Load Physical Anchors (from Auto-Multipol)
            with open(json_path, 'r') as f:
                phys_data = json.load(f)
                
            # 2. Load Fitted Slater Parameters (from JAX Training)
            with open(pickle_path, 'rb') as f:
                fitted_params = pickle.load(f)
                # Handle ParamSet object or dict
                if hasattr(fitted_params, 'parameters'):
                    fitted_params = fitted_params.parameters
                
            # 3. Align and extract
            mol_entry = {
                "molecule": mol_dir.name,
                "atoms": []
            }
            
            for i, atom in enumerate(phys_data['atoms']):
                # Align by atom index
                atom_entry = {
                    "index": i,
                    "element": atom['element'],
                    "charge": atom['charge'],
                    "v_eff": atom['volume_eff'],
                    "alpha_iso": atom['alpha_iso'],
                    "c6": atom['c6_ii'],
                    "c8": atom['c8_ii'],
                    "c10": atom['c10_ii'],
                    # Extract fitted Slater A components
                    "slater_A_ex": float(fitted_params['SlaterExForce']['A'][i]),
                    "slater_A_es": float(fitted_params['SlaterSrEsForce']['A'][i]),
                    "slater_A_pol": float(fitted_params['SlaterSrPolForce']['A'][i]),
                    "slater_A_disp": float(fitted_params['SlaterSrDispForce']['A'][i]),
                    "slater_A_dhf": float(fitted_params['SlaterSrDhfForce']['A'][i]),
                    # B is typically consistent across these forces in DMFF
                    "slater_B": float(fitted_params['SlaterExForce']['B'][i]) if hasattr(fitted_params['SlaterExForce']['B'], '__len__') else float(fitted_params['SlaterExForce']['B'])
                }
                mol_entry["atoms"].append(atom_entry)
                
            master_data.append(mol_entry)
        except Exception as e:
            print(f"Error processing {mol_dir.name}: {e}")
        
    # Save the consolidated dataset
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(master_data, f, indent=4)
        
    print(f"\nSuccess! Distilled data for {len(master_data)} molecules saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill and align physical/fitted force field parameters.")
    parser.add_argument("-i", "--input", required=True, help="Directory containing molecular result folders")
    parser.add_argument("-o", "--output", default="data/master_dataset.json", help="Output consolidated JSON")
    
    args = parser.parse_args()
    distill_data(args.input, args.output)
