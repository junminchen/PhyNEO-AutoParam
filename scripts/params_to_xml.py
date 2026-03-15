import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np

from physics.local_frames import AmoebaFrameConverter, cartesian_to_tinker_quadrupole


_DIPOLE_TO_DMFF = 0.1
_QUADRUPOLE_TO_DMFF = 1.0 / 300.0
_POLARIZABILITY_TO_DMFF = 0.001
_BOHR_TO_NM = 0.052917721092
_HARTREE_TO_KJMOL = 2625.499638
_PHYS_DIPOLE_TO_DMFF = _BOHR_TO_NM
_PHYS_QUADRUPOLE_TO_DMFF = (_BOHR_TO_NM ** 2) / 3.0
_PHYS_POLARIZABILITY_TO_DMFF = _BOHR_TO_NM ** 3
_PHYS_C6_TO_DMFF = _HARTREE_TO_KJMOL * (_BOHR_TO_NM ** 6)
_PHYS_C8_TO_DMFF = _HARTREE_TO_KJMOL * (_BOHR_TO_NM ** 8)
_PHYS_C10_TO_DMFF = _HARTREE_TO_KJMOL * (_BOHR_TO_NM ** 10)

def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def _as_numpy(value):
    return np.asarray(value, dtype=float)


def _neighbor_lists(mol):
    return {
        atom.GetIdx(): [nbr.GetIdx() for nbr in atom.GetNeighbors()]
        for atom in mol.GetAtoms()
    }


def _pick_local_frame_refs(mol, atom_idx, neighbors):
    refs = {}
    if not neighbors:
        return refs
    refs["kz"] = str(neighbors[0])
    if len(neighbors) > 1:
        refs["kx"] = str(neighbors[1])
    else:
        refs["kx"] = ""
    return refs


def _get_atom_position(mol, atom_idx):
    conf = mol.GetConformer()
    pos = conf.GetAtomPosition(atom_idx)
    return np.array([pos.x, pos.y, pos.z], dtype=float)


def _build_local_multipoles(mol, params, include_local_frames=False):
    coords = np.vstack([_get_atom_position(mol, i) for i in range(mol.GetNumAtoms())])
    frame_converter = AmoebaFrameConverter(coords)
    neighbors = _neighbor_lists(mol)

    dipoles = _as_numpy(params["dipole"])
    quadrupoles = _as_numpy(params["quadrupole"])

    local_data = []
    for i in range(mol.GetNumAtoms()):
        atom_neighbors = neighbors[i]
        refs = _pick_local_frame_refs(mol, i, atom_neighbors) if include_local_frames else {}

        dipole_vec = dipoles[i]
        quad_raw = quadrupoles[i]
        if quad_raw.shape == (6,):
            quad_matrix = np.array(
                [
                    [quad_raw[0], quad_raw[1], quad_raw[3]],
                    [quad_raw[1], quad_raw[2], quad_raw[4]],
                    [quad_raw[3], quad_raw[4], quad_raw[5]],
                ],
                dtype=float,
            )
        else:
            quad_matrix = np.asarray(quad_raw, dtype=float).reshape(3, 3)

        if include_local_frames and len(atom_neighbors) >= 2:
            d_local, q_local = frame_converter.rotate_multipoles(
                i,
                atom_neighbors[0],
                atom_neighbors[1],
                mode="Z-then-X",
                dipole=dipole_vec,
                quadrupole=quad_matrix,
            )
        else:
            d_local = dipole_vec
            q_local = quad_matrix

        quad_tinker = cartesian_to_tinker_quadrupole(q_local)
        local_data.append(
            {
                "refs": refs,
                "dipole": d_local * _DIPOLE_TO_DMFF,
                "quadrupole": {
                    key: value * _QUADRUPOLE_TO_DMFF
                    for key, value in quad_tinker.items()
                },
            }
        )
    return local_data


def _get_optional_vector(params, primary_key, fallback_key=None):
    if primary_key in params:
        return _as_numpy(params[primary_key])
    if fallback_key and fallback_key in params:
        return _as_numpy(params[fallback_key])
    return None


def _get_forcefield_terms(params, unit_style):
    charges = _as_numpy(params["q"])
    dipoles = _as_numpy(params["dipole"])
    quadrupoles = _as_numpy(params["quadrupole"])

    if unit_style == "model_head":
        dipole_scale = _DIPOLE_TO_DMFF
        quadrupole_scale = _QUADRUPOLE_TO_DMFF
        alpha = _as_numpy(params["kappa"]) * _POLARIZABILITY_TO_DMFF
        c6 = _as_numpy(params["c6"])
        c8 = _as_numpy(params["c8"])
        c10 = _as_numpy(params["c10"])
        b = _as_numpy(params["B"])
    elif unit_style == "physical_au":
        dipole_scale = _PHYS_DIPOLE_TO_DMFF
        quadrupole_scale = _PHYS_QUADRUPOLE_TO_DMFF
        alpha = _get_optional_vector(params, "alpha", "alpha_iso")
        c6 = _get_optional_vector(params, "c6", "c6_ii")
        c8 = _get_optional_vector(params, "c8", "c8_ii")
        c10 = _get_optional_vector(params, "c10", "c10_ii")
        b = _get_optional_vector(params, "B", "slater_b_init")
        if alpha is None or c6 is None or c8 is None or c10 is None or b is None:
            raise KeyError("physical_au export requires alpha/c6/c8/c10/B-like arrays in params")
        alpha = alpha * _PHYS_POLARIZABILITY_TO_DMFF
        c6 = c6 * _PHYS_C6_TO_DMFF
        c8 = c8 * _PHYS_C8_TO_DMFF
        c10 = c10 * _PHYS_C10_TO_DMFF
    else:
        raise ValueError(f"Unknown unit_style: {unit_style}")

    n_atoms = charges.shape[0]
    defaults = {
        "A_ex": np.full(n_atoms, 1.0),
        "A_es": np.full(n_atoms, 2.0),
        "A_pol": np.full(n_atoms, 3.0),
        "A_disp": np.full(n_atoms, 4.0),
        "A_dhf": np.full(n_atoms, 5.0),
    }
    a_terms = {
        key: _as_numpy(params[key]) if key in params else value
        for key, value in defaults.items()
    }

    converted = {
        "q": charges,
        "alpha": alpha,
        "c6": c6,
        "c8": c8,
        "c10": c10,
        "B": b,
        **a_terms,
    }
    local_input = {
        "dipole": dipoles * dipole_scale,
        "quadrupole": quadrupoles * quadrupole_scale,
    }
    return converted, local_input


def generate_dmff_xml(
    mol,
    params,
    output_path,
    residue_name="MOL",
    include_local_frames=False,
    unit_style="model_head",
):
    root = ET.Element("ForceField")
    converted, multipole_input = _get_forcefield_terms(params, unit_style)
    
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
    res = ET.SubElement(residues, "Residue", {"name": residue_name})
    for i, atom in enumerate(mol.GetAtoms()):
        ET.SubElement(res, "Atom", {"name": f"{atom.GetSymbol()}{i}", "type": str(i)})
    for bond in mol.GetBonds():
        ET.SubElement(res, "Bond", {"from": str(bond.GetBeginAtomIdx()), "to": str(bond.GetEndAtomIdx())})

    # 3. ADMPPmeForce (Electrostatics with Multipoles & Polarization)
    admp = ET.SubElement(
        root,
        "ADMPPmeForce",
        {
            "lmax": "2",
            "mScale12": "0.00",
            "mScale13": "0.00",
            "mScale14": "0.00",
            "mScale15": "0.00",
            "mScale16": "0.00",
            "pScale12": "0.00",
            "pScale13": "0.00",
            "pScale14": "0.00",
            "pScale15": "0.00",
            "pScale16": "0.00",
            "dScale12": "1.00",
            "dScale13": "1.00",
            "dScale14": "1.00",
            "dScale15": "1.00",
            "dScale16": "1.00",
        },
    )
    local_multipoles = _build_local_multipoles(
        mol,
        multipole_input,
        include_local_frames=include_local_frames,
    )
    for i in range(mol.GetNumAtoms()):
        multipole = local_multipoles[i]
        d = multipole["dipole"]
        q_tensor = multipole["quadrupole"]
        atom_attrs = {
            "type": str(i),
            "c0": f"{float(converted['q'][i]):.8f}",
            "dX": f"{d[0]:.8f}",
            "dY": f"{d[1]:.8f}",
            "dZ": f"{d[2]:.8f}",
            "qXX": f"{q_tensor['qXX']:.8f}",
            "qXY": f"{q_tensor['qXY']:.8f}",
            "qYY": f"{q_tensor['qYY']:.8f}",
            "qXZ": f"{q_tensor['qXZ']:.8f}",
            "qYZ": f"{q_tensor['qYZ']:.8f}",
            "qZZ": f"{q_tensor['qZZ']:.8f}",
        }
        atom_attrs.update(multipole["refs"])
        ET.SubElement(admp, "Atom", atom_attrs)
        # Polarize
        alpha = float(converted["alpha"][i])
        ET.SubElement(admp, "Polarize", {
            "type": str(i),
            "polarizabilityXX": f"{alpha:.6e}",
            "polarizabilityYY": f"{alpha:.6e}",
            "polarizabilityZZ": f"{alpha:.6e}",
            "thole": "0.33"
        })

    # 4. ADMPDispPmeForce (Dispersion)
    disp = ET.SubElement(
        root,
        "ADMPDispPmeForce",
        {
            "mScale12": "0.00",
            "mScale13": "0.00",
            "mScale14": "0.00",
            "mScale15": "0.00",
            "mScale16": "1.00",
        },
    )
    for i in range(mol.GetNumAtoms()):
        ET.SubElement(disp, "Atom", {
            "type": str(i),
            "C6": f"{converted['c6'][i]:.6e}",
            "C8": f"{converted['c8'][i]:.6e}",
            "C10": f"{converted['c10'][i]:.6e}"
        })

    # 5. Slater & Damping Forces
    slater_tags = ["SlaterExForce", "SlaterSrEsForce", "SlaterSrPolForce", "SlaterSrDispForce", "SlaterDhfForce"]
    force_keys = {"SlaterExForce": "A_ex", "SlaterSrEsForce": "A_es", "SlaterSrPolForce": "A_pol", "SlaterSrDispForce": "A_disp", "SlaterDhfForce": "A_dhf"}

    for tag in slater_tags:
        pk = force_keys[tag]
        f_elem = ET.SubElement(root, tag, {"mScale12": "0.00", "mScale13": "0.00"})
        for i in range(mol.GetNumAtoms()):
            attr = {"type": str(i), "A": f"{converted[pk][i]:.4f}", "B": f"{converted['B'][i]:.4f}"}
            if tag == "SlaterSrEsForce": attr["Q"] = f"{converted['q'][i]:.8f}"
            ET.SubElement(f_elem, "Atom", attr)

    # 6. Damping Forces (QqTt and SlaterDamping)
    qqtt = ET.SubElement(root, "QqTtDampingForce", {"mScale12": "0.00"})
    for i in range(mol.GetNumAtoms()):
        ET.SubElement(qqtt, "Atom", {"type": str(i), "B": f"{converted['B'][i]:.4f}", "Q": f"{converted['q'][i]:.8f}"})

    sldamp = ET.SubElement(root, "SlaterDampingForce", {"mScale12": "0.00"})
    for i in range(mol.GetNumAtoms()):
        ET.SubElement(sldamp, "Atom", {
            "type": str(i), "B": f"{converted['B'][i]:.4f}",
            "C6": f"{converted['c6'][i]:.6e}", "C8": f"{converted['c8'][i]:.6e}", "C10": f"{converted['c10'][i]:.6e}"
        })

    # Save to file
    with open(output_path, "w") as f:
        f.write(prettify(root))
    print(f"DMFF XML ForceField saved to: {output_path}")

if __name__ == "__main__":
    # This would be called from smiles_to_physics.py
    pass
