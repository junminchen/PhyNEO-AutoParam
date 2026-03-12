from rdkit import Chem
import itertools
import os

def generate_electrolyte_bank():
    print("=== Generating Comprehensive Electrolyte Molecule Bank ===")
    
    # Electrolyte-specific Substituents (R-groups)
    R_groups = [
        # 1. Aliphatic (Standard)
        "C", "CC", "CCC", "C(C)C", "CCCC", "CC(C)C", "C(C)(C)C", 
        # 2. Fluorinated (High voltage / SEI modifiers)
        "C(F)(F)F", "CC(F)(F)F", "C(F)(F)C(F)(F)H", "C(F)(F)C(F)(F)F", "C(F)C",
        # 3. Cyano (Nitriles - High voltage stability)
        "C#N", "CC#N", "CCC#N", 
        # 4. Unsaturated (Polymerization/SEI forming)
        "C=C", "CC=C", "C#C", 
        # 5. Ether-linked (Solvation power)
        "COC", "CCOC"
    ]
    
    unique_smiles = set()
    mol_dict = {}

    def add_mol(smiles, prefix):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            canon_smi = Chem.MolToSmiles(mol)
            if canon_smi not in unique_smiles:
                unique_smiles.add(canon_smi)
                mol_dict[f"{prefix}_{len(mol_dict)}"] = canon_smi

    # 1. Linear Carbonates: R1-O-C(=O)-O-R2
    for r1, r2 in itertools.combinations_with_replacement(R_groups, 2):
        add_mol(f"{r1}OC(=O)O{r2}", "LCarb")

    # 2. Linear Ethers: R1-O-R2
    for r1, r2 in itertools.combinations_with_replacement(R_groups, 2):
        add_mol(f"{r1}O{r2}", "LEth")

    # 3. Esters: R1-C(=O)-O-R2
    for r1 in R_groups:
        for r2 in R_groups:
            add_mol(f"{r1}C(=O)O{r2}", "Est")

    # 4. Sulfones: R1-S(=O)(=O)-R2
    for r1, r2 in itertools.combinations_with_replacement(R_groups, 2):
        add_mol(f"{r1}S(=O)(=O){r2}", "Sulf")
        
    # 5. Phosphates: P(=O)(OR1)(OR2)(OR3) (Symmetric & AAB to limit explosion)
    for r1 in R_groups:
        add_mol(f"O=P(O{r1})(O{r1})O{r1}", "Phos")
        for r2 in R_groups:
            if r1 != r2:
                add_mol(f"O=P(O{r1})(O{r1})O{r2}", "Phos")

    # 6. Cyclic Carbonates (Ethylene carbonate derivatives)
    for r1 in R_groups:
        add_mol(f"O=C1OCC({r1})O1", "CCarb")
        for r2 in R_groups[:10]: # Limit combinations to prevent explosion
            add_mol(f"O=C1OC({r1})C({r2})O1", "CCarb")

    # 7. Lactones (gamma-butyrolactone derivatives)
    for r1 in R_groups:
        add_mol(f"O=C1CCC({r1})O1", "Lact")
        
    # 8. Known complex additives / Specific structures
    bases = [
        "C1=COC(=O)O1", # VC
        "C=CC1COC(=O)O1", # VEC
        "O=S1(=O)OCCO1", # DTD
        "O=S1(=O)OCCCO1", # PS
        "O=C1COCCO1", # 1,4-Dioxane-2-one
    ]
    for b in bases:
        add_mol(b, "Spec")

    print(f"Targeting completion... Generated {len(mol_dict)} strictly unique, valid electrolyte molecules.")
    
    os.makedirs("data", exist_ok=True)
    output_path = "data/electrolyte_1000.py"
    with open(output_path, "w") as f:
        f.write("electrolyte_bank_1000 = {\n")
        for k, v in mol_dict.items():
            f.write(f"    '{k}': '{v}',\n")
        f.write("}\n")
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    generate_electrolyte_bank()
