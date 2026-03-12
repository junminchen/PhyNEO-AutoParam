import xml.etree.ElementTree as ET
from xml.dom import minidom
import jax.numpy as jnp
from rdkit import Chem

def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def generate_dmff_xml(mol, params, output_path):
    root = ET.Element("ForceField")
    
    # 1. AtomTypes
    atom_types = ET.SubElement(root, "AtomTypes")
    for i, atom in enumerate(mol.GetAtoms()):
        ET.SubElement(atom_types, "Type", {
            "name": str(i),
            "class": atom.GetSymbol(),
            "element": atom.GetSymbol(),
            "mass": str(atom.GetMass())
        })
        
    # 2. Residues
    residues = ET.SubElement(root, "Residues")
    res = ET.SubElement(residues, "Residue", {"name": "MOL"})
    for i, atom in enumerate(mol.GetAtoms()):
        ET.SubElement(res, "Atom", {"name": f"{atom.GetSymbol()}{i}", "type": str(i)})
    for bond in mol.GetBonds():
        ET.SubElement(res, "Bond", {"from": str(bond.GetBeginAtomIdx()), "to": str(bond.GetEndAtomIdx())})

    # 3. ADMPPmeForce (Electrostatics with Multipoles & Polarization)
    admp = ET.SubElement(root, "ADMPPmeForce", {"lmax": "2"}) 
    for i in range(mol.GetNumAtoms()):
        d = params['dipole'][i]
        q_tensor = params['quadrupole'][i]
        ET.SubElement(admp, "Atom", {
            "type": str(i), 
            "c0": f"{params['q'][i]:.8f}",
            "dX": f"{d[0]:.8f}", "dY": f"{d[1]:.8f}", "dZ": f"{d[2]:.8f}",
            "qXX": f"{q_tensor[0]:.8f}", "qXY": f"{q_tensor[1]:.8f}", "qYY": f"{q_tensor[2]:.8f}",
            "qXZ": f"{q_tensor[3]:.8f}", "qYZ": f"{q_tensor[4]:.8f}", "qZZ": f"{q_tensor[5]:.8f}"
        })
        # Polarize
        alpha = params['kappa'][i] * 0.001
        ET.SubElement(admp, "Polarize", {
            "type": str(i),
            "polarizabilityXX": f"{alpha:.6e}",
            "polarizabilityYY": f"{alpha:.6e}",
            "polarizabilityZZ": f"{alpha:.6e}",
            "thole": "0.33"
        })

    # 4. ADMPDispPmeForce (Dispersion)
    disp = ET.SubElement(root, "ADMPDispPmeForce", {"mScale12": "0.00", "mScale13": "0.00"})
    for i in range(mol.GetNumAtoms()):
        ET.SubElement(disp, "Atom", {
            "type": str(i),
            "C6": f"{params['c6'][i]:.6e}",
            "C8": f"{params['c8'][i]:.6e}",
            "C10": f"{params['c10'][i]:.6e}"
        })

    # 5. Slater & Damping Forces
    slater_tags = ["SlaterExForce", "SlaterSrEsForce", "SlaterSrPolForce", "SlaterSrDispForce", "SlaterDhfForce"]
    force_keys = {"SlaterExForce": "A_ex", "SlaterSrEsForce": "A_es", "SlaterSrPolForce": "A_pol", "SlaterSrDispForce": "A_disp", "SlaterDhfForce": "A_dhf"}

    for tag in slater_tags:
        pk = force_keys[tag]
        f_elem = ET.SubElement(root, tag, {"mScale12": "0.00", "mScale13": "0.00"})
        for i in range(mol.GetNumAtoms()):
            attr = {"type": str(i), "A": f"{params[pk][i]:.4f}", "B": f"{params['B'][i]:.4f}"}
            if tag == "SlaterSrEsForce": attr["Q"] = f"{params['q'][i]:.8f}"
            ET.SubElement(f_elem, "Atom", attr)

    # 6. Damping Forces (QqTt and SlaterDamping)
    qqtt = ET.SubElement(root, "QqTtDampingForce", {"mScale12": "0.00"})
    for i in range(mol.GetNumAtoms()):
        ET.SubElement(qqtt, "Atom", {"type": str(i), "B": f"{params['B'][i]:.4f}", "Q": f"{params['q'][i]:.8f}"})

    sldamp = ET.SubElement(root, "SlaterDampingForce", {"mScale12": "0.00"})
    for i in range(mol.GetNumAtoms()):
        ET.SubElement(sldamp, "Atom", {
            "type": str(i), "B": f"{params['B'][i]:.4f}",
            "C6": f"{params['c6'][i]:.6e}", "C8": f"{params['c8'][i]:.6e}", "C10": f"{params['c10'][i]:.6e}"
        })

    # Save to file
    with open(output_path, "w") as f:
        f.write(prettify(root))
    print(f"DMFF XML ForceField saved to: {output_path}")

if __name__ == "__main__":
    # This would be called from smiles_to_physics.py
    pass
