#!/usr/bin/env python3
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def plot_sapt_ff_multi_y(key, scan_res, batch, out_png):
    sapt_es = scan_res["es"] + scan_res["lr_es"]
    sapt_pol = scan_res["pol"] + scan_res["lr_pol"]
    sapt_disp = scan_res["disp"] + scan_res["lr_disp"]
    sapt_tot = scan_res["tot_full"]
    dist = scan_res["shift"]

    ff_es = scan_res["lr_es"]
    ff_pol = scan_res["lr_pol"]
    ff_disp = scan_res["lr_disp"]
    ff_lr = scan_res["lr_tot"]

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    labels = ["Total", "Electrostatics", "Polarization", "Dispersion"]
    sapt_data = [sapt_tot, sapt_es, sapt_pol, sapt_disp]
    ff_data = [ff_lr, ff_es, ff_pol, ff_disp]

    fig = go.Figure()
    y_axes = ["y", "y2", "y3", "y4"]

    for i, (label, color) in enumerate(zip(labels, colors)):
        fig.add_trace(
            go.Scatter(
                x=dist,
                y=sapt_data[i],
                mode="lines+markers",
                name=f"{label} (SAPT)",
                line=dict(color=color, dash="dash", width=2),
                marker=dict(symbol="x", size=7),
                yaxis=y_axes[i],
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dist,
                y=ff_data[i],
                mode="lines+markers",
                name=f"{label} (FFs)",
                line=dict(color=color, width=2),
                marker=dict(symbol="circle", size=6),
                yaxis=y_axes[i],
            )
        )

    fig.update_layout(
        title=f"{key} {batch} Average Params",
        width=1300,
        height=560,
        xaxis=dict(title="Distance (A)", showgrid=True, gridcolor="rgba(128,128,128,0.3)"),
        yaxis=dict(title="Total Energy (kJ/mol)", titlefont=dict(color=colors[0]), tickfont=dict(color=colors[0])),
        yaxis2=dict(
            title="Electrostatics Energy (kJ/mol)",
            titlefont=dict(color=colors[1]),
            tickfont=dict(color=colors[1]),
            overlaying="y",
            side="right",
            position=0.87,
        ),
        yaxis3=dict(
            title="Polarization Energy (kJ/mol)",
            titlefont=dict(color=colors[2]),
            tickfont=dict(color=colors[2]),
            overlaying="y",
            side="right",
            position=0.93,
        ),
        yaxis4=dict(
            title="Dispersion Energy (kJ/mol)",
            titlefont=dict(color=colors[3]),
            tickfont=dict(color=colors[3]),
            overlaying="y",
            side="right",
            position=0.99,
        ),
        legend=dict(x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.75)"),
        margin=dict(l=80, r=200, t=60, b=70),
    )
    fig.write_html(out_png)


def main():
    parser = argparse.ArgumentParser(description="Plot EC-EC long-range curves in check_lr.ipynb style.")
    parser.add_argument("--workdir", default=".", help="Folder with data_dimer.pickle and EC assets")
    parser.add_argument("--key", default="conf_003_EC_EC")
    parser.add_argument("--batch", default="000")
    parser.add_argument("--ff", default="EC/EC.xml")
    parser.add_argument("--pdb-a", default="EC/EC.pdb")
    parser.add_argument("--pdb-b", default="EC/EC.pdb")
    parser.add_argument("--out", default="plot_long-range_conf_003_EC_EC_000.html")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute lr_es/lr_pol/lr_disp with DMFF. Default uses stored lr_* in pickle.",
    )
    parser.add_argument(
        "--dmff-root",
        default="/Users/jeremychen/Desktop/Project/project_h2o_etoh_github/3_MD/vendor/DMFF",
        help="Path that contains dmff/",
    )
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    with open(workdir / "data_dimer.pickle", "rb") as f:
        data = pickle.load(f)

    scan_res = data[args.key][args.batch]
    if args.recompute:
        import jax.numpy as jnp
        from jax import vmap

        sys.path.insert(0, str(workdir))
        sys.path.insert(0, str(Path(args.dmff_root).resolve()))

        # Reuse the manual-pairs LR calculator written for this folder.
        from calc_lr_from_pickle import BasePairsManual, make_dimer_pdb_from_monomer
        from dmff.api import Hamiltonian

        ff_xml = str((workdir / args.ff).resolve())
        pdb_a = str((workdir / args.pdb_a).resolve())
        pdb_b = str((workdir / args.pdb_b).resolve())

        dimer_pdb = make_dimer_pdb_from_monomer(pdb_a)
        try:
            bp = BasePairsManual(ff_xml, dimer_pdb, pdb_a, pdb_b)
            params = Hamiltonian(ff_xml).getParameters()
            calc = vmap(bp.cal_e, in_axes=(None, 0, 0), out_axes=(0, 0, 0))
            e_es, e_pol, e_disp = calc(params, jnp.array(scan_res["posA"]), jnp.array(scan_res["posB"]))
        finally:
            Path(dimer_pdb).unlink(missing_ok=True)

        # Keep notebook semantics: SAPT terms are short-range+long-range.
        scan_res["lr_es"] = np.array(e_es)
        scan_res["lr_pol"] = np.array(e_pol)
        scan_res["lr_disp"] = np.array(e_disp)
        scan_res["lr_tot"] = np.array(e_es + e_pol + e_disp)
        if "tot_full" not in scan_res:
            scan_res["tot_full"] = np.array(scan_res["tot"]) + np.array(scan_res["lr_tot"])

    if "shift" not in scan_res or len(scan_res["shift"]) != len(scan_res["lr_tot"]):
        dist = np.linalg.norm(np.mean(scan_res["posA"], axis=1) - np.mean(scan_res["posB"], axis=1), axis=1)
        scan_res["shift"] = dist

    out_png = (workdir / args.out).resolve()
    plot_sapt_ff_multi_y(args.key, scan_res, args.batch, str(out_png))
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
