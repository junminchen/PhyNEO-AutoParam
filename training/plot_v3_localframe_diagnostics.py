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

from scripts.params_to_xml import _PHYS_C10_TO_DMFF, _PHYS_C6_TO_DMFF, _PHYS_C8_TO_DMFF, _POLARIZABILITY_TO_DMFF
from training.joint_trainer import JointTrainer
from training.retrain_from_production import load_dataset


def stats(ref, pred):
    ref = np.asarray(ref, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean((ref - pred) ** 2))),
        "mae": float(np.mean(np.abs(ref - pred))),
    }


def collect_predictions(production_root: Path, checkpoint_path: Path, seed: int, val_frac: float, multipole_frame: str):
    dataset = load_dataset(production_root, multipole_frame=multipole_frame)
    rng = random.Random(seed)
    rng.shuffle(dataset)
    val_size = max(1, int(len(dataset) * val_frac))
    val_data = dataset[:val_size]

    with open(checkpoint_path, "rb") as f:
        variables = pickle.load(f)
    trainer = JointTrainer()

    q_ref, q_pred = [], []
    d_ref, d_pred = [], []
    qt_ref, qt_pred = [], []
    a_ref, a_pred = [], []
    c6_ref, c6_pred = [], []
    c8_ref, c8_pred = [], []
    c10_ref, c10_pred = [], []

    for item in val_data:
        out = trainer.model.apply(variables, item["graph"], total_charge=item["total_q"])
        q_ref.extend(np.asarray(item["q_ref"]).ravel())
        q_pred.extend(np.asarray(out["q"]).ravel())
        d_ref.extend(np.asarray(item["d_ref"]).ravel())
        d_pred.extend(np.asarray(out["dipole"]).ravel())
        qt_ref.extend(np.asarray(item["qt_ref"]).ravel())
        qt_pred.extend(np.asarray(out["quadrupole"]).ravel())
        alpha_ref = np.asarray(item.get("alpha_ref", np.asarray(item["k_ref"]) * _POLARIZABILITY_TO_DMFF))
        a_ref.extend(alpha_ref.ravel())
        a_pred.extend((np.asarray(out["kappa"]) * _POLARIZABILITY_TO_DMFF).ravel())

        raw = json.loads((production_root / item["name"] / "results_high.json").read_text())["atoms"]
        c6_ref.extend([atom["c6_ii"] * _PHYS_C6_TO_DMFF for atom in raw])
        c8_ref.extend([atom["c8_ii"] * _PHYS_C8_TO_DMFF for atom in raw])
        c10_ref.extend([atom["c10_ii"] * _PHYS_C10_TO_DMFF for atom in raw])
        c6_pred.extend(np.asarray(out["c6"]).ravel())
        c8_pred.extend(np.asarray(out["c8"]).ravel())
        c10_pred.extend(np.asarray(out["c10"]).ravel())

    return {
        "charge": (np.asarray(q_ref), np.asarray(q_pred)),
        f"dipole_{multipole_frame}": (np.asarray(d_ref), np.asarray(d_pred)),
        f"quadrupole_{multipole_frame}": (np.asarray(qt_ref), np.asarray(qt_pred)),
        "polarizability": (np.asarray(a_ref), np.asarray(a_pred)),
        "C6": (np.asarray(c6_ref), np.asarray(c6_pred)),
        "C8": (np.asarray(c8_ref), np.asarray(c8_pred)),
        "C10": (np.asarray(c10_ref), np.asarray(c10_pred)),
        "val_molecules": len(val_data),
        "val_names": [item["name"] for item in val_data],
    }


def plot_loss(metrics_path: Path, output_path: Path, title: str):
    metrics = json.loads(metrics_path.read_text())
    hist = metrics["history"]
    epochs = [x["epoch"] for x in hist]
    train = [x["train_loss"] for x in hist]
    val = [x["val_loss"] for x in hist]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(epochs, train, lw=2, label="Train")
    ax.plot(epochs, val, lw=2, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_parity_grid(predictions, output_path: Path, title: str):
    keys = [key for key in ["charge", "dipole_local", "dipole_global", "quadrupole_local", "quadrupole_global", "polarizability", "C6", "C8", "C10"] if key in predictions]
    fig, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
    axes = axes.ravel()
    summary = {}

    for ax, key in zip(axes, keys):
        ref, pred = predictions[key]
        metric = stats(ref, pred)
        summary[key] = metric
        lo = float(min(ref.min(), pred.min()))
        hi = float(max(ref.max(), pred.max()))
        pad = (hi - lo) * 0.05 if hi > lo else 1.0
        ax.scatter(ref, pred, s=8, alpha=0.4)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="black", lw=1.25)
        ax.set_title(f"{key}\nRMSE={metric['rmse']:.3g}, MAE={metric['mae']:.3g}")
        ax.set_xlabel("Reference")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.2)

    for ax in axes[len(keys):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Plot diagnostics for a local-frame PhyNEO model.")
    parser.add_argument(
        "--production-root",
        default="/Users/jeremychen/Desktop/Project/repo/PhyNEO-AutoParam/examples/production_results",
    )
    parser.add_argument(
        "--checkpoint",
        default="/Users/jeremychen/Desktop/Project/repo/PhyNEO-AutoParam/models/phyneo_production_results_v3_localframe_alpha_e80.flax",
    )
    parser.add_argument(
        "--metrics",
        default="/Users/jeremychen/Desktop/Project/repo/PhyNEO-AutoParam/models/phyneo_production_results_v3_localframe_alpha_e80.metrics.json",
    )
    parser.add_argument(
        "--out-dir",
        default="/Users/jeremychen/Desktop/Project/repo/PhyNEO-AutoParam/examples/results_pdb_bank_inference/retrain_diagnostics_v3",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--multipole-frame", choices=["local", "global"], default="local")
    parser.add_argument("--loss-title", default="Training Loss")
    parser.add_argument("--parity-title", default="Validation Parity")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = collect_predictions(
        Path(args.production_root),
        Path(args.checkpoint),
        args.seed,
        args.val_frac,
        args.multipole_frame,
    )
    loss_path = out_dir / "loss_curve.png"
    parity_path = out_dir / "parity_grid.png"
    plot_loss(Path(args.metrics), loss_path, args.loss_title)
    summary = plot_parity_grid(predictions, parity_path, args.parity_title)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "val_molecules": predictions["val_molecules"],
                "val_names": predictions["val_names"],
                "metrics": summary,
                "loss_plot": str(loss_path),
                "parity_plot": str(parity_path),
            },
            indent=2,
        )
    )
    print(f"loss_plot -> {loss_path}")
    print(f"parity_plot -> {parity_path}")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
