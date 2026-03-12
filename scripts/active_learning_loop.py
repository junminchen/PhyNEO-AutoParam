import os
import random
import subprocess
import time
from pathlib import Path

# Add project root to sys.path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.electrolyte_1000 import electrolyte_bank_1000

def get_completed_molecules(production_dir="examples/production_results"):
    if not os.path.exists(production_dir):
        return set()
    return set(os.listdir(production_dir))

def run_cmd(cmd, desc):
    print(f"\n[AL] ---> Executing: {desc}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[AL] ERROR during {desc}: {e}")
        return False
    return True

def active_learning_loop(cycles=100, batch_size=5):
    production_dir = "examples/production_results"
    os.makedirs(production_dir, exist_ok=True)
    
    print(f"=== PhyNEO-AutoParam Active Learning Engine [PHASE 2: DEEP EVOLUTION] ===")
    print(f"Goal: Sample and calculate up to 500 new molecules from the bank.")
    
    for cycle in range(1, cycles + 1):
        completed = get_completed_molecules(production_dir)
        unlabeled = {k: v for k, v in electrolyte_bank_1000.items() if k not in completed}
        
        if not unlabeled:
            print("[AL] All molecules in the bank have been processed. AL Loop finished!")
            break
            
        print(f"\n=======================================================")
        print(f"   ACTIVE LEARNING CYCLE {cycle}/{cycles}")
        print(f"   Labeled Pool: {len(completed)} | Unlabeled Pool: {len(unlabeled)}")
        print(f"=======================================================")
        
        # 1. Active Sampling (Random strategy for prototype)
        # Advanced AL would rank by GNN uncertainty or latent distance here
        selected_names = random.sample(list(unlabeled.keys()), min(batch_size, len(unlabeled)))
        print(f"[AL] Selected {len(selected_names)} molecules for Oracle Calculation:")
        print(f"[AL] {selected_names}")
        
        # 2. Oracle: Quantum Chemistry Calculation
        success_count = 0
        for name in selected_names:
            smiles = unlabeled[name]
            print(f"\n[AL-Oracle] Calculating {name} ({smiles})...")
            
            # Note: We use def2-TZVP / wb97xd by default in the script, enabling GPU
            cmd_qc = ["python", "scripts/smiles_to_dataset_entry.py", "-s", smiles, "-n", name, "--gpu"]
            if run_cmd(cmd_qc, f"DFT Calculation for {name}"):
                # Move to production dir
                src = f"results/{name}"
                dst = f"{production_dir}/{name}"
                if os.path.exists(src):
                    if os.path.exists(dst):
                        import shutil
                        shutil.rmtree(dst)
                    os.rename(src, dst)
                    success_count += 1
                    
        if success_count == 0:
            print("[AL] Warning: No molecules succeeded in this cycle. Skipping training.")
            continue
            
        # 3. Knowledge Distillation
        cmd_distill = ["python", "scripts/data_distill.py", "-i", production_dir, "-o", "data/master_dataset.json"]
        run_cmd(cmd_distill, "Data Distillation")
        
        # 4. Model Evolution (Training)
        # We run the joint trainer to absorb new knowledge
        cmd_train = ["python", "scripts/joint_trainer.py"]
        # Since the trainer is in training/, adjust path:
        cmd_train = ["python", "training/joint_trainer.py"]
        run_cmd(cmd_train, "GNN Joint Training")
        
        print(f"\n[AL] Cycle {cycle} Completed. Model has evolved.")
        time.sleep(2)

if __name__ == "__main__":
    # Start AL loop: 20 cycles, picking 3 molecules each cycle to evolve steadily
    active_learning_loop(cycles=20, batch_size=3)
