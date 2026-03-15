#!/usr/bin/env python3
import argparse
import csv
import pickle
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import vmap


def parse_key(key: str):
    parts = key.split("_")
    if len(parts) < 4:
        raise ValueError(f"Invalid key format: {key}")
    _, conf_id, monomer_a, monomer_b = parts[:4]
    return conf_id, monomer_a, monomer_b


def main():
    parser = argparse.ArgumentParser(description="Compute LR terms for a dimer key and export comparison CSV.")
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--pickle", default="data_dimer.pickle")
    parser.add_argument("--key", default="conf_003_EC_EC")
    parser.add_argument("--ff", default="EC/EC.xml")
    parser.add_argument("--box-nm", type=float, default=30.0)
    parser.add_argument("--max-batches", type=int, default=10, help="How many sorted batches to export (0=all)")
    parser.add_argument("--out", default="ec_ec_lr_compare.csv")
    parser.add_argument(
        "--dmff-root",
        default="/Users/jeremychen/Desktop/Project/project_h2o_etoh_github/3_MD/vendor/DMFF",
    )
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    sys.path.insert(0, str(workdir))
    sys.path.insert(0, str(Path(args.dmff_root).resolve()))

    from calc_lr_from_pickle import BasePairsManual
    from dmff.api import Hamiltonian

    conf_id, monomer_a, monomer_b = parse_key(args.key)
    dimer_pdb = workdir / "dimer_bank" / f"dimer_{conf_id}_{monomer_a}_{monomer_b}.pdb"
    if not dimer_pdb.exists():
        raise FileNotFoundError(f"Missing dimer pdb: {dimer_pdb}")

    pdb_a = workdir / monomer_a / f"{monomer_a}.pdb"
    pdb_b = workdir / monomer_b / f"{monomer_b}.pdb"
    ff_xml = workdir / args.ff

    with open(workdir / args.pickle, "rb") as f:
        data = pickle.load(f)
    if args.key not in data:
        raise KeyError(f"Key not in pickle: {args.key}")

    base = BasePairsManual(str(ff_xml), str(dimer_pdb), str(pdb_a), str(pdb_b), box_nm=args.box_nm)
    params = Hamiltonian(str(ff_xml)).getParameters()
    calc = vmap(base.cal_e, in_axes=(None, 0, 0), out_axes=(0, 0, 0))

    batches = sorted(data[args.key].keys())
    if args.max_batches > 0:
        batches = batches[: args.max_batches]

    out_csv = workdir / args.out
    fieldnames = [
        "batch",
        "point",
        "distance",
        "calc_lr_es",
        "calc_lr_pol",
        "calc_lr_disp",
        "calc_lr_tot",
        "ref_lr_es",
        "ref_lr_pol",
        "ref_lr_disp",
        "ref_lr_tot",
        "delta_lr_es",
        "delta_lr_pol",
        "delta_lr_disp",
        "delta_lr_tot",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for batch in batches:
            scan = data[args.key][batch]
            pos_a = jnp.array(scan["posA"])
            pos_b = jnp.array(scan["posB"])

            e_es, e_pol, e_disp = calc(params, pos_a, pos_b)
            e_tot = e_es + e_pol + e_disp

            dist = np.linalg.norm(np.mean(np.array(scan["posA"]), axis=1) - np.mean(np.array(scan["posB"]), axis=1), axis=1)
            ref_es = np.array(scan.get("lr_es", np.full_like(np.array(e_es), np.nan)))
            ref_pol = np.array(scan.get("lr_pol", np.full_like(np.array(e_pol), np.nan)))
            ref_disp = np.array(scan.get("lr_disp", np.full_like(np.array(e_disp), np.nan)))
            ref_tot = np.array(scan.get("lr_tot", np.full_like(np.array(e_tot), np.nan)))

            for i in range(len(e_tot)):
                row = {
                    "batch": batch,
                    "point": i,
                    "distance": float(dist[i]),
                    "calc_lr_es": float(e_es[i]),
                    "calc_lr_pol": float(e_pol[i]),
                    "calc_lr_disp": float(e_disp[i]),
                    "calc_lr_tot": float(e_tot[i]),
                    "ref_lr_es": float(ref_es[i]),
                    "ref_lr_pol": float(ref_pol[i]),
                    "ref_lr_disp": float(ref_disp[i]),
                    "ref_lr_tot": float(ref_tot[i]),
                    "delta_lr_es": float(e_es[i] - ref_es[i]),
                    "delta_lr_pol": float(e_pol[i] - ref_pol[i]),
                    "delta_lr_disp": float(e_disp[i] - ref_disp[i]),
                    "delta_lr_tot": float(e_tot[i] - ref_tot[i]),
                }
                writer.writerow(row)

    print(f"Saved CSV: {out_csv}")
    print(f"Batches exported: {len(batches)}")


if __name__ == "__main__":
    main()
