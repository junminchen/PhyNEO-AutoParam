#!/usr/bin/env python3
import argparse
from pathlib import Path

import MDAnalysis as mda
import numpy as np


def parse_key(key: str):
    parts = key.split("_")
    if len(parts) < 4:
        raise ValueError(f"Invalid key format: {key}")
    _, conf_id, monomer_a, monomer_b = parts[:4]
    return conf_id, monomer_a, monomer_b


def build_dimer_pdb(workdir: Path, key: str, box_nm: float = 30.0, shift_angstrom: float = 10.0):
    conf_id, monomer_a, monomer_b = parse_key(key)
    pdb_a = workdir / monomer_a / f"{monomer_a}.pdb"
    pdb_b = workdir / monomer_b / f"{monomer_b}.pdb"
    if not pdb_a.exists():
        raise FileNotFoundError(f"Missing monomer A pdb: {pdb_a}")
    if not pdb_b.exists():
        raise FileNotFoundError(f"Missing monomer B pdb: {pdb_b}")

    u_a = mda.Universe(str(pdb_a))
    u_b = mda.Universe(str(pdb_b))
    merged = mda.Merge(u_a.atoms, u_b.atoms)

    pos_a = u_a.atoms.positions.copy()
    pos_b = u_b.atoms.positions.copy() + np.array([shift_angstrom, 0.0, 0.0], dtype=float)
    merged.atoms.positions = np.vstack([pos_a, pos_b])
    merged.dimensions = np.array([box_nm * 10.0, box_nm * 10.0, box_nm * 10.0, 90.0, 90.0, 90.0], dtype=float)

    out_dir = workdir / "dimer_bank"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dimer_{conf_id}_{monomer_a}_{monomer_b}.pdb"
    merged.atoms.write(str(out_path))

    # Rewrite chain/residue IDs to avoid duplicated atom identity when read by OpenMM.
    n_a = len(u_a.atoms)
    fixed_lines = []
    with open(out_path, "r") as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 27:
                serial = int(line[6:11].strip())
                if serial <= n_a:
                    chain = "A"
                    resid = 1
                else:
                    chain = "B"
                    resid = 2
                line = f"{line[:21]}{chain}{resid:4d}{line[26:]}"
            fixed_lines.append(line)
    with open(out_path, "w") as f:
        f.writelines(fixed_lines)

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build dimer PDB(s) with MDAnalysis in a large periodic box.")
    parser.add_argument("--workdir", default=".", help="Folder containing monomer subfolders")
    parser.add_argument("--key", default="conf_003_EC_EC", help="Dimer key, e.g. conf_003_EC_EC")
    parser.add_argument("--box-nm", type=float, default=30.0, help="Box length in nm")
    parser.add_argument("--shift-angstrom", type=float, default=10.0, help="Initial shift of monomer B in Angstrom")
    args = parser.parse_args()

    out_path = build_dimer_pdb(Path(args.workdir).resolve(), args.key, box_nm=args.box_nm, shift_angstrom=args.shift_angstrom)
    print(f"Saved dimer PDB: {out_path}")


if __name__ == "__main__":
    main()
