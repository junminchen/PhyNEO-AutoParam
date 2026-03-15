#!/usr/bin/env python3
import argparse
import pickle
import sys
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import vmap
from openmm.app import CutoffPeriodic, Modeller, PDBFile
from openmm.unit import angstrom


def build_pairs(coords_nm: np.ndarray, cov_map: np.ndarray, rc_nm: float = 2.5) -> jnp.ndarray:
    out = []
    nat = coords_nm.shape[0]
    for i in range(nat):
        for j in range(i + 1, nat):
            if np.linalg.norm(coords_nm[i] - coords_nm[j]) < rc_nm:
                out.append((i, j, int(cov_map[i, j])))
    if not out:
        return jnp.zeros((0, 3), dtype=jnp.int32)
    return jnp.array(out, dtype=jnp.int32)


class BasePairsManual:
    def __init__(
        self,
        ff_xml: str,
        dimer_pdb: str,
        monomer_a_pdb: str,
        monomer_b_pdb: str,
        box_nm: float = 30.0,
    ):
        from dmff.api import Hamiltonian

        pdb = PDBFile(dimer_pdb)
        pdb_a = PDBFile(monomer_a_pdb)
        pdb_b = PDBFile(monomer_b_pdb)

        self.h = Hamiltonian(ff_xml)
        self.h_a = Hamiltonian(ff_xml)
        self.h_b = Hamiltonian(ff_xml)

        self.pot = self.h.createPotential(
            pdb.topology, nonbondedCutoff=25 * angstrom, nonbondedMethod=CutoffPeriodic, ethresh=1e-4, step_pol=20
        )
        self.pot_a = self.h_a.createPotential(
            pdb_a.topology, nonbondedCutoff=25 * angstrom, nonbondedMethod=CutoffPeriodic, ethresh=1e-4, step_pol=20
        )
        self.pot_b = self.h_b.createPotential(
            pdb_b.topology, nonbondedCutoff=25 * angstrom, nonbondedMethod=CutoffPeriodic, ethresh=1e-4, step_pol=20
        )

        self.g = self.h.getGenerators()
        self.g_a = self.h_a.getGenerators()
        self.g_b = self.h_b.getGenerators()

        self.box = jnp.eye(3) * float(box_nm)
        self.rc = 2.5

        pos = np.array(pdb.positions._value)
        pos_a = np.array(pdb_a.positions._value)
        pos_b = np.array(pdb_b.positions._value)

        self.pairs = build_pairs(pos, np.array(self.pot.meta["cov_map"]), rc_nm=self.rc)
        self.pairs_a = build_pairs(pos_a, np.array(self.pot_a.meta["cov_map"]), rc_nm=self.rc)
        self.pairs_b = build_pairs(pos_b, np.array(self.pot_b.meta["cov_map"]), rc_nm=self.rc)

        self.pot_es = self.pot.dmff_potentials["ADMPPmeForce"]
        self.pot_es_a = self.pot_a.dmff_potentials["ADMPPmeForce"]
        self.pot_es_b = self.pot_b.dmff_potentials["ADMPPmeForce"]
        self.pot_disp = self.pot.dmff_potentials["ADMPDispPmeForce"]
        self.pot_disp_a = self.pot_a.dmff_potentials["ADMPDispPmeForce"]
        self.pot_disp_b = self.pot_b.dmff_potentials["ADMPDispPmeForce"]

    def cal_e(self, params, pos_a, pos_b):
        # Keep notebook convention: positions in A -> nm for potential calls.
        pos_a = pos_a * 0.1
        pos_b = pos_b * 0.1
        pos_ab = jnp.concatenate([pos_a, pos_b], axis=0)
        box = self.box

        e_espol_a = self.pot_es_a(pos_a, box, self.pairs_a, params)
        e_espol_b = self.pot_es_b(pos_b, box, self.pairs_b, params)
        e_espol_ab = self.pot_es(pos_ab, box, self.pairs, params) - e_espol_a - e_espol_b

        pme_ab = self.g[0]
        pme_a = self.g_a[0]
        pme_b = self.g_b[0]
        u_ind_ab = jnp.vstack((pme_a.pme_force.U_ind, pme_b.pme_force.U_ind))
        params_pme = params["ADMPPmeForce"]
        map_atypes = self.pot.meta["ADMPPmeForce_map_atomtype"]
        map_poltypes = self.pot.meta["ADMPPmeForce_map_poltype"]
        q_local = params_pme["Q_local"][map_atypes]
        pol = params_pme["pol"][map_poltypes]
        tholes = params_pme["thole"][map_poltypes]
        e_nonpol_ab = pme_ab.pme_force.energy_fn(
            pos_ab * 10,
            box * 10,
            self.pairs,
            q_local,
            u_ind_ab,
            pol,
            tholes,
            pme_ab.mScales,
            pme_ab.pScales,
            pme_ab.dScales,
        )
        e_es = e_nonpol_ab - e_espol_a - e_espol_b
        e_pol = e_espol_ab - e_es

        e_disp = (
            self.pot_disp(pos_ab, box, self.pairs, params)
            - self.pot_disp_a(pos_a, box, self.pairs_a, params)
            - self.pot_disp_b(pos_b, box, self.pairs_b, params)
        )
        return e_es, e_pol, e_disp


def make_dimer_pdb_from_monomer(monomer_pdb: str) -> str:
    pdb_a = PDBFile(monomer_pdb)
    modeller = Modeller(pdb_a.topology, pdb_a.positions)
    modeller.add(pdb_a.topology, pdb_a.positions)
    fd, tmp = tempfile.mkstemp(prefix="dimer_", suffix=".pdb")
    Path(tmp).unlink(missing_ok=True)
    with open(tmp, "w") as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)
    return tmp


def main():
    parser = argparse.ArgumentParser(description="Compute long-range ES/POL/DISP from pickle scan data.")
    parser.add_argument("--workdir", default=".", help="Folder with check_lr.ipynb-style assets")
    parser.add_argument("--pickle", default="data_dimer.pickle", help="Input pickle file")
    parser.add_argument("--key", default="conf_003_EC_EC", help="Dimer key in pickle")
    parser.add_argument("--batch", default="000", help="Batch id under key")
    parser.add_argument("--ff", default="EC/EC.xml", help="FF XML path (relative to workdir)")
    parser.add_argument("--pdb-a", default="EC/EC.pdb", help="Monomer A PDB path (relative to workdir)")
    parser.add_argument("--pdb-b", default="EC/EC.pdb", help="Monomer B PDB path (relative to workdir)")
    parser.add_argument("--dimer-pdb", default=None, help="Optional dimer PDB path (relative to workdir)")
    parser.add_argument("--box-nm", type=float, default=30.0, help="Periodic box length in nm")
    parser.add_argument(
        "--dmff-root",
        default="/Users/jeremychen/Desktop/Project/project_h2o_etoh_github/3_MD/vendor/DMFF",
        help="Path that contains dmff/",
    )
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    ff_xml = str((workdir / args.ff).resolve())
    pdb_a = str((workdir / args.pdb_a).resolve())
    pdb_b = str((workdir / args.pdb_b).resolve())
    pickle_path = workdir / args.pickle

    sys.path.insert(0, str(Path(args.dmff_root).resolve()))
    from dmff.api import Hamiltonian

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    scan = data[args.key][args.batch]

    if args.dimer_pdb:
        dimer_pdb = str((workdir / args.dimer_pdb).resolve())
        cleanup = False
    else:
        dimer_pdb = make_dimer_pdb_from_monomer(pdb_a)
        cleanup = True
    try:
        base = BasePairsManual(ff_xml, dimer_pdb, pdb_a, pdb_b, box_nm=args.box_nm)
        params = Hamiltonian(ff_xml).getParameters()
        calc = vmap(base.cal_e, in_axes=(None, 0, 0), out_axes=(0, 0, 0))
        e_es, e_pol, e_disp = calc(params, jnp.array(scan["posA"]), jnp.array(scan["posB"]))
        e_tot = e_es + e_pol + e_disp
    finally:
        if cleanup:
            Path(dimer_pdb).unlink(missing_ok=True)

    print(f"Key={args.key} Batch={args.batch} N={len(e_tot)}")
    print(
        "First point (kJ/mol-ish): "
        f"ES={float(e_es[0]):.6f} POL={float(e_pol[0]):.6f} DISP={float(e_disp[0]):.6f} TOT={float(e_tot[0]):.6f}"
    )
    print(
        "Mean over scan: "
        f"ES={float(jnp.mean(e_es)):.6f} POL={float(jnp.mean(e_pol)):.6f} DISP={float(jnp.mean(e_disp)):.6f} TOT={float(jnp.mean(e_tot)):.6f}"
    )

    if "lr_tot" in scan:
        ref = np.array(scan["lr_tot"])
        diff = np.array(e_tot) - ref
        print(f"Reference lr_tot(first/mean): {float(ref[0]):.6f} / {float(np.mean(ref)):.6f}")
        print(f"Delta vs reference(first/mean): {float(diff[0]):.6f} / {float(np.mean(diff)):.6f}")


if __name__ == "__main__":
    main()
