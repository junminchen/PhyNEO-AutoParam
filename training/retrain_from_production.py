import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import jax
import numpy as np
import optax
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.molecule import Molecule
from scripts.params_to_xml import (
    _DIPOLE_TO_DMFF,
    _PHYS_C10_TO_DMFF,
    _PHYS_C6_TO_DMFF,
    _PHYS_C8_TO_DMFF,
    _PHYS_DIPOLE_TO_DMFF,
    _PHYS_POLARIZABILITY_TO_DMFF,
    _PHYS_QUADRUPOLE_TO_DMFF,
    _POLARIZABILITY_TO_DMFF,
    _QUADRUPOLE_TO_DMFF,
    _build_local_multipoles,
)
from training.joint_trainer import JointTrainer


def build_mol_from_xyz(xyz_path: Path, total_charge: int):
    mol = Chem.MolFromXYZBlock(xyz_path.read_text())
    if mol is None:
        raise ValueError(f"Cannot parse XYZ: {xyz_path}")
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=int(total_charge), embedChiral=False)
    except Exception:
        pass
    mol.UpdatePropertyCache(strict=False)
    return mol


def _flatten_cartesian_quadrupole(quadrupole):
    q = np.asarray(quadrupole, dtype=float)
    if q.shape == (6,):
        return q
    q = q.reshape(3, 3)
    return np.array([q[0, 0], q[0, 1], q[1, 1], q[0, 2], q[1, 2], q[2, 2]], dtype=float)


def load_dataset(production_root: Path, multipole_frame: str = "local"):
    dataset = []
    dipole_scale = _PHYS_DIPOLE_TO_DMFF / _DIPOLE_TO_DMFF
    quadrupole_scale = _PHYS_QUADRUPOLE_TO_DMFF / _QUADRUPOLE_TO_DMFF

    for json_path in sorted(production_root.glob("*/results_high.json")):
        name = json_path.parent.name
        xyz_path = json_path.with_name(f"{name}.xyz")
        if not xyz_path.exists():
            continue

        data = json.loads(json_path.read_text())
        atoms = data["atoms"]
        total_q = float(int(round(sum(atom["charge"] for atom in atoms))))
        mol = build_mol_from_xyz(xyz_path, int(total_q))
        mol_obj = Molecule(mol, name=name)

        q_ref = np.array([atom["charge"] for atom in atoms], dtype=float)
        alpha_nm3 = np.array([atom["alpha_iso"] for atom in atoms], dtype=float) * _PHYS_POLARIZABILITY_TO_DMFF
        c6_ref = np.array([atom["c6_ii"] for atom in atoms], dtype=float) * _PHYS_C6_TO_DMFF
        c8_ref = np.array([atom["c8_ii"] for atom in atoms], dtype=float) * _PHYS_C8_TO_DMFF
        c10_ref = np.array([atom["c10_ii"] for atom in atoms], dtype=float) * _PHYS_C10_TO_DMFF

        dipole_raw = np.array([atom["dipole"] for atom in atoms], dtype=float) * dipole_scale
        quadrupole_raw = np.array([_flatten_cartesian_quadrupole(atom["quadrupole"]) for atom in atoms], dtype=float)
        quadrupole_raw = quadrupole_raw * quadrupole_scale

        if multipole_frame == "local":
            local = _build_local_multipoles(
                mol,
                {"dipole": dipole_raw, "quadrupole": quadrupole_raw},
                include_local_frames=True,
            )
            d_ref = np.array([entry["dipole"] for entry in local], dtype=float)
            qt_ref = np.array(
                [
                    [
                        entry["quadrupole"]["qXX"],
                        entry["quadrupole"]["qXY"],
                        entry["quadrupole"]["qYY"],
                        entry["quadrupole"]["qXZ"],
                        entry["quadrupole"]["qYZ"],
                        entry["quadrupole"]["qZZ"],
                    ]
                    for entry in local
                ],
                dtype=float,
            )
        elif multipole_frame == "global":
            d_ref = dipole_raw
            qt_ref = quadrupole_raw
        else:
            raise ValueError(f"Unsupported multipole_frame: {multipole_frame}")

        dataset.append(
            {
                "name": name,
                "graph": mol_obj.get_graph(),
                "q_ref": q_ref,
                "alpha_ref": alpha_nm3,
                "k_ref": alpha_nm3 / _POLARIZABILITY_TO_DMFF,
                "c6_ref": c6_ref,
                "c8_ref": c8_ref,
                "c10_ref": c10_ref,
                "d_ref": d_ref,
                "qt_ref": qt_ref,
                "total_q": total_q,
                "slater_targets": None,
            }
        )

    print(f"Loaded {len(dataset)} production molecules with {multipole_frame}-frame multipole labels")
    return dataset


def _flatten_targets(dataset, key):
    if not dataset:
        return np.array([1.0], dtype=float)
    arrays = [np.asarray(item[key], dtype=float).reshape(-1) for item in dataset]
    merged = np.concatenate(arrays)
    finite = merged[np.isfinite(merged)]
    if finite.size == 0:
        return np.array([1.0], dtype=float)
    return finite


def compute_target_scales(dataset):
    scales = {}
    key_map = {
        "q": "q_ref",
        "alpha": "alpha_ref",
        "dipole": "d_ref",
        "quadrupole": "qt_ref",
        "c6": "c6_ref",
        "c8": "c8_ref",
        "c10": "c10_ref",
    }
    for scale_key, data_key in key_map.items():
        values = _flatten_targets(dataset, data_key)
        std = float(np.std(values))
        scales[scale_key] = std if std > 1e-8 else 1.0
    return scales


def evaluate_dataset(trainer: JointTrainer, variables, dataset):
    if not dataset:
        return float("nan")
    losses = [float(trainer.loss_fn(variables, item)) for item in dataset]
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser(description="Retrain PhyNEO on distilled production_results dataset.")
    parser.add_argument(
        "--production-root",
        default="/Users/jeremychen/Desktop/Project/repo/PhyNEO-AutoParam/examples/production_results",
    )
    parser.add_argument(
        "--output",
        default="/Users/jeremychen/Desktop/Project/repo/PhyNEO-AutoParam/models/phyneo_production_results_v4_direct_alpha_disp_localframe.flax",
    )
    parser.add_argument(
        "--metrics-out",
        default="/Users/jeremychen/Desktop/Project/repo/PhyNEO-AutoParam/models/phyneo_production_results_v4_direct_alpha_disp_localframe.metrics.json",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--multipole-frame", choices=["local", "global"], default="local")
    parser.add_argument("--dipole-weight", type=float, default=1.0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    dataset = load_dataset(Path(args.production_root), multipole_frame=args.multipole_frame)
    if len(dataset) < 2:
        raise ValueError("Dataset too small for retraining.")

    rng.shuffle(dataset)
    val_size = max(1, int(len(dataset) * args.val_frac))
    val_data = dataset[:val_size]
    train_data = dataset[val_size:]

    target_scales = compute_target_scales(train_data)
    trainer = JointTrainer(
        learning_rate=args.learning_rate,
        target_scales=target_scales,
        loss_weights={
            "q": 1.0,
            "alpha": 1.0,
            "dipole": args.dipole_weight,
            "quadrupole": 1.0,
            "c6": 1.0,
            "c8": 1.0,
            "c10": 1.0,
        },
    )
    total_steps = max(1, len(train_data) * args.epochs)
    lr_schedule = optax.cosine_decay_schedule(
        init_value=args.learning_rate,
        decay_steps=total_steps,
        alpha=1e-2,
    )
    trainer.optimizer = optax.adam(learning_rate=lr_schedule)

    sample = train_data[0]
    variables = trainer.model.init(jax.random.PRNGKey(args.seed), sample["graph"], total_charge=sample["total_q"])
    opt_state = trainer.optimizer.init(variables)

    best_val = float("inf")
    best_variables = variables
    metrics = []

    print(f"Train molecules: {len(train_data)} | Val molecules: {len(val_data)}")
    print(f"Epochs: {args.epochs}")
    print("Target scales:")
    for key in ["q", "alpha", "dipole", "quadrupole", "c6", "c8", "c10"]:
        print(f"  {key}: {target_scales[key]:.6e}")

    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_data)
        epoch_losses = []
        for item in train_data:
            variables, opt_state, loss = trainer.train_step(
                variables,
                opt_state,
                item["graph"],
                item["q_ref"],
                item["alpha_ref"],
                item["d_ref"],
                item["qt_ref"],
                item["c6_ref"],
                item["c8_ref"],
                item["c10_ref"],
                item["total_q"],
                item["slater_targets"],
            )
            epoch_losses.append(float(loss))

        train_loss = float(np.mean(epoch_losses))
        val_loss = evaluate_dataset(trainer, variables, val_data)
        metrics.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "target_scales": target_scales,
            }
        )

        if val_loss < best_val:
            best_val = val_loss
            best_variables = variables
            with open(args.output, "wb") as f:
                pickle.dump(best_variables, f)

        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:4d} | train {train_loss:.6f} | val {val_loss:.6f}")

    Path(args.metrics_out).write_text(json.dumps({"best_val_loss": best_val, "history": metrics}, indent=2))
    print(f"Best checkpoint: {args.output}")
    print(f"Metrics: {args.metrics_out}")


if __name__ == "__main__":
    main()
