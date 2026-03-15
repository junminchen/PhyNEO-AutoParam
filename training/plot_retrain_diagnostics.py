import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.joint_trainer import load_master_dataset, JointTrainer
from core.molecule import Molecule
from scripts.params_to_xml import _build_local_multipoles


def parity_stats(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rmse = float(np.sqrt(np.mean((x - y) ** 2)))
    mae = float(np.mean(np.abs(x - y)))
    return {"rmse": rmse, "mae": mae}


def collect_validation_predictions(dataset_path: Path, checkpoint_path: Path, seed: int, val_frac: float):
    dataset = load_master_dataset(str(dataset_path))
    rng = random.Random(seed)
    rng.shuffle(dataset)
    val_size = max(1, int(len(dataset) * val_frac))
    val_data = dataset[:val_size]

    with open(checkpoint_path, "rb") as f:
        variables = pickle.load(f)
    trainer = JointTrainer()

    q_ref, q_pred = [], []
    alpha_ref, alpha_pred = [], []
    d_ref, d_pred = [], []
    dmag_ref, dmag_pred = [], []
    dlocal_ref, dlocal_pred = [], []
    qt_ref, qt_pred = [], []

    for item in val_data:
        out = trainer.model.apply(variables, item["graph"], total_charge=item["total_q"])
        mol = Molecule.from_smiles(item["smiles"], name=item["name"]) if "smiles" in item else None
        q_ref.extend(np.asarray(item["q_ref"]).ravel())
        q_pred.extend(np.asarray(out["q"]).ravel())
        alpha_ref.extend(np.asarray(item.get("alpha_ref", np.zeros_like(item["k_ref"]))).ravel())
        alpha_pred.extend((np.asarray(out["kappa"]).ravel()) * 0.001)
        d_ref_arr = np.asarray(item["d_ref"])
        d_pred_arr = np.asarray(out["dipole"])
        d_ref.extend(d_ref_arr.ravel())
        d_pred.extend(d_pred_arr.ravel())
        dmag_ref.extend(np.linalg.norm(d_ref_arr, axis=1))
        dmag_pred.extend(np.linalg.norm(d_pred_arr, axis=1))

        # Compare the actual exported local-frame components rather than only global xyz.
        if mol is not None:
            ref_local = _build_local_multipoles(
                mol.rdmol,
                {"dipole": d_ref_arr, "quadrupole": np.asarray(item["qt_ref"]).reshape(-1, 6)},
                include_local_frames=True,
            )
            pred_local = _build_local_multipoles(
                mol.rdmol,
                {"dipole": d_pred_arr, "quadrupole": np.asarray(out["quadrupole"]).reshape(-1, 6)},
                include_local_frames=True,
            )
            dlocal_ref.extend(np.asarray([entry["dipole"] for entry in ref_local]).ravel())
            dlocal_pred.extend(np.asarray([entry["dipole"] for entry in pred_local]).ravel())
        qt_ref.extend(np.asarray(item["qt_ref"]).ravel())
        qt_pred.extend(np.asarray(out["quadrupole"]).ravel())

    return {
        "q": (np.asarray(q_ref), np.asarray(q_pred)),
        "alpha": (np.asarray(alpha_ref), np.asarray(alpha_pred)),
        "dipole_global": (np.asarray(d_ref), np.asarray(d_pred)),
        "dipole_magnitude": (np.asarray(dmag_ref), np.asarray(dmag_pred)),
        "dipole_local": (np.asarray(dlocal_ref), np.asarray(dlocal_pred)),
        "quadrupole": (np.asarray(qt_ref), np.asarray(qt_pred)),
        "val_molecules": len(val_data),
    }


def plot_loss(metrics_path: Path, output_path: Path):
    metrics = json.loads(metrics_path.read_text())
    history = metrics["history"]
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(epochs, train_loss, label="Train", lw=2)
    ax.plot(epochs, val_loss, label="Validation", lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Retrain Loss Curve")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_parity(predictions, output_path: Path):
    keys = ["q", "alpha", "dipole_magnitude", "quadrupole"]
    titles = {
        "q": "Charge",
        "alpha": "Polarizability (nm^3)",
        "dipole_magnitude": "Dipole Magnitude",
        "quadrupole": "Quadrupole Components",
    }

    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    axes = axes.ravel()
    summary = {}

    for ax, key in zip(axes, keys):
        ref, pred = predictions[key]
        stats = parity_stats(ref, pred)
        summary[key] = stats
        low = float(min(ref.min(), pred.min()))
        high = float(max(ref.max(), pred.max()))
        pad = (high - low) * 0.05 if high > low else 1.0
        ax.scatter(ref, pred, s=8, alpha=0.45)
        ax.plot([low - pad, high + pad], [low - pad, high + pad], "--", lw=1.5, color="black")
        ax.set_title(f"{titles[key]}\nRMSE={stats['rmse']:.4g}, MAE={stats['mae']:.4g}")
        ax.set_xlabel("Reference")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.2)

    fig.suptitle("Validation Parity Plots", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return summary


def plot_local_dipole_parity(predictions, output_path: Path):
    ref, pred = predictions["dipole_local"]
    stats = parity_stats(ref, pred)
    low = float(min(ref.min(), pred.min()))
    high = float(max(ref.max(), pred.max()))
    pad = (high - low) * 0.05 if high > low else 1.0

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.scatter(ref, pred, s=8, alpha=0.45)
    ax.plot([low - pad, high + pad], [low - pad, high + pad], "--", lw=1.5, color="black")
    ax.set_title(f"Local-Frame Dipole Components\nRMSE={stats['rmse']:.4g}, MAE={stats['mae']:.4g}")
    ax.set_xlabel("Reference")
    ax.set_ylabel("Predicted")
    ax.grid(alpha=0.2)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Plot retrain diagnostics: loss curve and validation parity.")
    parser.add_argument(
        "--dataset",
        default="/Users/jeremychen/Desktop/Project/repo/PhyNEO-AutoParam/data/master_dataset_production_results.json",
    )
    parser.add_argument(
        "--checkpoint",
        default="/Users/jeremychen/Desktop/Project/repo/PhyNEO-AutoParam/models/phyneo_production_results_v2_e150.flax",
    )
    parser.add_argument(
        "--metrics",
        default="/Users/jeremychen/Desktop/Project/repo/PhyNEO-AutoParam/models/phyneo_production_results_v2_e150.metrics.json",
    )
    parser.add_argument(
        "--out-dir",
        default="/Users/jeremychen/Desktop/Project/repo/PhyNEO-AutoParam/examples/results_pdb_bank_inference/retrain_diagnostics",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = collect_validation_predictions(
        Path(args.dataset),
        Path(args.checkpoint),
        args.seed,
        args.val_frac,
    )
    parity_path = out_dir / "validation_parity.png"
    summary = plot_parity(predictions, parity_path)
    local_dipole_path = out_dir / "dipole_local_parity.png"
    local_dipole_summary = plot_local_dipole_parity(predictions, local_dipole_path)

    metrics_path = Path(args.metrics)
    loss_path = None
    if metrics_path.exists():
        loss_path = out_dir / "loss_curve.png"
        plot_loss(metrics_path, loss_path)

    summary_path = out_dir / "parity_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "val_molecules": predictions["val_molecules"],
                "parity": summary,
                "dipole_local": local_dipole_summary,
                "loss_curve": str(loss_path) if loss_path else None,
                "parity_plot": str(parity_path),
                "dipole_local_plot": str(local_dipole_path),
            },
            indent=2,
        )
    )
    print(f"parity_plot -> {parity_path}")
    print(f"dipole_local_plot -> {local_dipole_path}")
    if loss_path is not None:
        print(f"loss_plot -> {loss_path}")
    else:
        print("loss_plot -> metrics file not available yet")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
