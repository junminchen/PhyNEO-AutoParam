from rdkit import Chem
from rdkit.Chem import AllChem
import os

def generate_organic_fragments():
    """
    Generates a diversity set of small organic molecules (1-6 heavy atoms)
    covering common force field chemical environments.
    """
    smiles_list = [
        # Alkanes
        "C", "CC", "CCC", "CC(C)C", "CCCC",
        # Ethers
        "COC", "COCC", "CCOCC", "C1CO1", "C1CCOC1",
        # Alcohols
        "CO", "CCO", "CCCO", "CC(O)C",
        # Carbonyls (Aldehydes/Ketones)
        "C=O", "CC=O", "CC(=O)C", "CCC=O",
        # Esters/Acids
        "CC(=O)O", "CC(=O)OC", "COC=O", "CCOC(=O)C",
        # Nitriles
        "C#N", "CC#N", "CCC#N",
        # Fluorinated (Crucial for Electrolytes)
        "CF", "C(F)F", "C(F)(F)F", "CCF", "CC(F)F", "CC(F)(F)F", "FC1COC(=O)O1",
        # Sulfur/Phosphorus (From our salt bank)
        "CS(=O)(=O)C", "CP(=O)(O)O", "COP(=O)(OC)OC"
    ]
    
    # Add some combinations
    extended = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            extended.append((s, f"FRAG_{s.replace('=', 'eq').replace('#', 'tri').replace('(', '_').replace(')', '_')}"))
            
    return extended

def update_production_queue(fragments):
    output_path = "data/fragment_smiles.py"
    with open(output_path, "w") as f:
        f.write("fragment_bank = {\n")
        for smiles, name in fragments:
            f.write(f"    '{name}': '{smiles}',\n")
        f.write("}\n")
    print(f"Generated {len(fragments)} fragments in {output_path}")

if __name__ == "__main__":
    frags = generate_organic_fragments()
    update_production_queue(frags)
