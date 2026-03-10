import pickle
import numpy as np
import os
from pathlib import Path

# Common species atom counts (including H)
SPECIES_ATOMS = {
    'Li': 1, 'Na': 1, 'K': 1,
    'PF6': 7, 'BF4': 5, 'ClO4': 5, 'TFSI': 15, 'FSI': 11, 'DFOB': 11, 'BOB': 17,
    'EC': 10, 'DMC': 12, 'DEC': 18, 'PC': 13, 'EMC': 15, 'VC': 8, 'FEC': 10,
    'DME': 16, 'DOL': 11, 'G1': 16, 'G2': 22, 'G3': 28, 'G4': 34,
    'SL': 15, 'PS': 12, 'PP': 12, 'ADN': 12,
    'H2O': 3, 'EA': 9, # Ethanol
}

def extract_monomers(pkl_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    found_monomers = {}
    
    for d in data:
        pair = d['pair']
        m1_name, m2_name = pair.split('_')
        symbols = d['symbols']
        coords = d['coords'] # Assumed to be in Angstrom or Bohr? Let's check
        
        # Determine split index
        n1 = SPECIES_ATOMS.get(m1_name)
        if n1 is None: 
            # Try to infer from pair name or symbols if needed
            continue
            
        # Extract Monomer 1
        if m1_name not in found_monomers:
            m1_symbols = symbols[:n1]
            m1_coords = coords[:n1]
            found_monomers[m1_name] = (m1_symbols, m1_coords)
            
        # Extract Monomer 2
        if m2_name not in found_monomers:
            m2_symbols = symbols[n1:]
            m2_coords = coords[n1:]
            found_monomers[m2_name] = (m2_symbols, m2_coords)
            
    # Save as XYZ
    for name, (syms, crds) in found_monomers.items():
        xyz_path = output_dir / f"{name}.xyz"
        with open(xyz_path, 'w') as f:
            f.write(f"{len(syms)}\n{name} extracted from eda_dataset\n")
            for s, c in zip(syms, crds):
                f.write(f"{s:2s} {c[0]:12.6f} {c[1]:12.6f} {c[2]:12.6f}\n")
        print(f"Saved {name}.xyz")

if __name__ == "__main__":
    pkl_path = "/home/jmchen/project/volc_qcclient/analyze_data/eda_dataset.pkl"
    output_dir = "data/monomers"
    extract_monomers(pkl_path, output_dir)
