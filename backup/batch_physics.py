import os
import subprocess
from pathlib import Path

# Charge mapping for ions
CHARGES = {
    'Li': 1, 'Na': 1, 'K': 1,
    'PF6': -1, 'BF4': -1, 'ClO4': -1, 'TFSI': -1, 'FSI': -1, 'DFOB': -1, 'BOB': -1,
}

def run_batch(monomer_dir, results_dir):
    monomer_dir = Path(monomer_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for xyz_file in [monomer_dir / f for f in ["EC.xyz", "Na.xyz", "Li.xyz", "PF6.xyz"]]:
        name = xyz_file.stem
        output_subdir = results_dir / name
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        json_out = output_subdir / "results_high.json"
        if json_out.exists():
            print(f"Skipping {name}, already exists.")
            continue
            
        charge = CHARGES.get(name, 0)
        print(f"--- Running Physics for {name} (Charge: {charge}) ---")
        
        cmd = [
            "python", "physics/multipole_engine.py",
            "-i", str(xyz_file),
            "-o", str(json_out),
            "--charge", str(charge)
        ]
        
        # We need to ensure multipole_engine.py accepts --charge
        subprocess.run(cmd)

if __name__ == "__main__":
    run_batch("data/monomers", "results")
