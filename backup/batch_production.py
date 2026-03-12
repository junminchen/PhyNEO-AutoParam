import sys
import os
from pathlib import Path

# Add project root to path to import local modules
sys.path.append(str(Path(__file__).parent.parent))

from data.mol_smiles import phyneo_name_mapped_smiles
from scripts.smiles_to_dataset_entry import process_smiles

def run_production(target_names=None):
    results_dir = "production_results"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"=== PhyNEO-AutoParam Batch Production Starting ===")
    print(f"Output Directory: {results_dir}")
    
    # If no targets provided, process all
    targets = target_names if target_names else phyneo_name_mapped_smiles.keys()
    
    for name in targets:
        smiles = phyneo_name_mapped_smiles.get(name)
        if not smiles:
            print(f"Warning: Molecule {name} not found in smiles map.")
            continue
            
        print(f"\n>>> Starting Production for: {name}")
        try:
            # Using production standards: wb97x-d / def2-TZVP
            process_smiles(smiles, name, results_dir=results_dir)
            print(f">>> Completed: {name}")
        except Exception as e:
            print(f">>> Failed: {name} | Error: {e}")

if __name__ == "__main__":
    # Run production for all molecules defined in mol_smiles.py
    run_production()
