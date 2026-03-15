import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from openmm import Vec3, app, unit


def ensure_topology_box(topology, box_nm: float):
    if topology.getPeriodicBoxVectors() is not None:
        return
    length = float(box_nm)
    vectors = (
        Vec3(length, 0.0, 0.0),
        Vec3(0.0, length, 0.0),
        Vec3(0.0, 0.0, length),
    )
    topology.setPeriodicBoxVectors(vectors)


def main():
    parser = argparse.ArgumentParser(description="Quick single-molecule DMFF energy check.")
    parser.add_argument("--xml", required=True, help="Path to FF.xml")
    parser.add_argument("--pdb", required=True, help="Path to structure.pdb")
    parser.add_argument(
        "--dmff-root",
        default="/Users/jeremychen/Desktop/Project/project_h2o_etoh_github/3_MD/vendor/DMFF",
        help="Path whose child folder is dmff/",
    )
    parser.add_argument(
        "--box-nm",
        type=float,
        default=8.0,
        help="Cubic box size in nm used for no-cutoff evaluation.",
    )
    args = parser.parse_args()

    dmff_root = Path(args.dmff_root)
    if not dmff_root.exists():
        raise FileNotFoundError(f"DMFF root not found: {dmff_root}")
    if str(dmff_root) not in sys.path:
        sys.path.insert(0, str(dmff_root))

    from dmff.api import Hamiltonian
    from dmff.common.nblist import NoCutoffNeighborList

    xml_path = Path(args.xml)
    pdb_path = Path(args.pdb)
    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB not found: {pdb_path}")

    pdb = app.PDBFile(str(pdb_path))
    ensure_topology_box(pdb.topology, args.box_nm)
    h = Hamiltonian(str(xml_path))
    potential = h.createPotential(pdb.topology, nonbondedMethod=app.NoCutoff)
    params = h.getParameters()

    positions = jnp.array(pdb.positions.value_in_unit(unit.nanometer))
    box = jnp.eye(3) * float(args.box_nm)

    nbl = NoCutoffNeighborList(potential.meta["cov_map"], padding=False)
    pairs = nbl.allocate(np.asarray(positions))

    efunc = potential.getPotentialFunc()
    energy = efunc(positions, box, pairs, params)
    energy_val = float(energy)

    print("=== DMFF Quick Energy ===")
    print(f"XML: {xml_path}")
    print(f"PDB: {pdb_path}")
    print(f"Atoms: {positions.shape[0]}")
    print(f"Energy (raw DMFF units, typically kJ/mol): {energy_val:.10f}")


if __name__ == "__main__":
    main()
