import argparse
import os
import sys
from pathlib import Path

# Add project root to path to import local modules
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

from core.molecule import Molecule
from physics.multipole_engine import MoleculeAnalyzer

def process_smiles(smiles_list, output_dir, basis, xc):
    """
    Automates the pipeline from SMILES to high-precision physical anchors.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, smiles in enumerate(smiles_list):
        mol_name = f"mol_{i:03d}"
        print(f"\n[SMILES Pipeline] Processing: {smiles} as {mol_name}")
        
        try:
            # 1. SMILES -> 3D via RDKit
            mol_obj = Molecule.from_smiles(smiles, name=mol_name)
            xyz_path = output_dir / f"{mol_name}.xyz"
            
            # Save temporary XYZ for the physics engine
            coords = mol_obj.get_coords()
            symbols = [atom.GetSymbol() for atom in mol_obj.rdmol.GetAtoms()]
            
            with open(xyz_path, 'w') as f:
                f.write(f"{len(symbols)}\nGenerated from SMILES: {smiles}\n")
                for sym, c in zip(symbols, coords):
                    f.write(f"{sym:2s} {c[0]:12.6f} {c[1]:12.6f} {c[2]:12.6f}\n")
            
            # 2. 3D XYZ -> Physical Anchors via Multipole Engine
            json_out = output_dir / f"{mol_name}_results.json"
            analyzer = MoleculeAnalyzer(str(xyz_path), basis=basis, xc=xc)
            results = analyzer.run_pipeline()
            
            import json
            with open(json_out, 'w') as f:
                json.dump(results, f, indent=4)
                
            print(f"[Success] Physical anchors saved to {json_out}")
            
        except Exception as e:
            print(f"[Error] Failed to process {smiles}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SMILES directly to high-precision physical anchors.")
    parser.add_argument("-s", "--smiles", nargs="+", required=True, help="One or more SMILES strings")
    parser.add_argument("-o", "--output", default="data/processed_smiles", help="Output directory")
    parser.add_argument("-b", "--basis", default="aug-cc-pVTZ", help="Basis set")
    parser.add_argument("-f", "--functional", default="wb97x-d", help="DFT Functional")
    
    args = parser.parse_args()
    process_smiles(args.smiles, args.output, args.basis, args.functional)
