import os
import subprocess
from data.mol_smiles import phyneo_name_mapped_smiles
from data.fragment_smiles import fragment_bank

def run_serial_production():
    output_base = "examples/production_results"
    os.makedirs(output_base, exist_ok=True)
    
    # 1. Combine all target molecules
    full_queue = {**phyneo_name_mapped_smiles, **fragment_bank}
    
    # 2. Filter for missing ones
    completed = os.listdir(output_base)
    missing = [name for name in full_queue if name not in completed]
    
    print(f"=== Starting Production Queue: {len(missing)} molecules left ===")
    
    for name in missing:
        smiles = full_queue.get(name)
        print(f"\n>>> Processing ({missing.index(name)+1}/{len(missing)}): {name}")
        
        # Calculate
        cmd = ["python", "scripts/smiles_to_dataset_entry.py", "-s", smiles, "-n", name]
        try:
            subprocess.run(cmd, check=True)
            
            # Move results
            src = f"results/{name}"
            dst = f"{output_base}/{name}"
            
            if os.path.exists(dst):
                import shutil
                shutil.rmtree(dst)
            
            os.rename(src, dst)
            print(f">>> Successfully completed: {name}")
        except Exception as e:
            print(f">>> Failed: {name} | Error: {e}")

if __name__ == "__main__":
    run_serial_production()
