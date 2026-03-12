import os
import json
import argparse
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from physics.multipole_engine import MoleculeAnalyzer

def smiles_to_xyz(smiles, output_path, name="MOL"):
    """
    Generate an optimized 3D XYZ from SMILES using RDKit (ETKDG + MMFF).
    """
    print(f"--- Converting SMILES: {smiles} to XYZ ---")
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    # Quick pre-optimization with MMFF to get a sane structure
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except:
        pass
    
    conf = mol.GetConformer()
    with open(output_path, "w") as f:
        f.write(f"{mol.GetNumAtoms()}\n")
        f.write(f"{name} generated from SMILES\n")
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            f.write(f"{atom.GetSymbol()} {pos.x:10.6f} {pos.y:10.6f} {pos.z:10.6f}\n")
    return output_path

def process_smiles(smiles, name, results_dir="results", basis="def2-TZVP", xc="wb97xd", use_gpu=False):
    results_dir = Path(results_dir)
    mol_dir = results_dir / name
    mol_dir.mkdir(parents=True, exist_ok=True)
    
    xyz_path = mol_dir / f"{name}.xyz"
    json_path = mol_dir / "results_high.json"
    
    # 1. SMILES -> XYZ
    smiles_to_xyz(smiles, xyz_path, name)
    
    # 2. Run Physics Engine (PySCF + MBIS)
    mol_rd = Chem.MolFromSmiles(smiles)
    total_charge = int(Chem.GetFormalCharge(mol_rd))
    
    print(f"--- Processing {name} (Charge: {total_charge} / Basis: {basis}) ---")
    analyzer = MoleculeAnalyzer(
        input_path=xyz_path, 
        basis=basis, 
        xc=xc, 
        charge=total_charge,
        use_gpu=use_gpu
    )
    results = analyzer.run_pipeline()
    
    # 3. Save Results
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nSuccess! Physics anchors for {name} saved to {json_path}")
    return json_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Produce physical anchors directly from SMILES.")
    parser.add_argument("-s", "--smiles", required=True, help="SMILES string")
    parser.add_argument("-n", "--name", required=True, help="Molecule name (e.g., DMC)")
    parser.add_argument("-b", "--basis", default="def2-TZVP", help="Basis set")
    parser.add_argument("-f", "--functional", default="wb97xd", help="DFT functional")
    parser.add_argument("--gpu", action="store_true", help="Use gpu4pyscf for calculation")
    
    args = parser.parse_args()
    process_smiles(args.smiles, args.name, basis=args.basis, xc=args.functional, use_gpu=args.gpu)
