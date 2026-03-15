import argparse
import json
import os
import pickle
import shutil
from pathlib import Path

import jax
from rdkit import Chem

from core.molecule import Molecule
from models.gnn_jax import PhyNEO_GNN_V2
from scripts.params_to_xml import generate_dmff_xml


def load_rdkit_pdb(pdb_path: Path):
    mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=True)
    if mol is not None:
        return mol

    # Fallback for species that often trigger strict valence errors (e.g. BF4-/BOB-/DFOB-).
    mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=False)
    if mol is None:
        raise ValueError(f"Cannot parse PDB: {pdb_path}")
    mol.UpdatePropertyCache(strict=False)

    try:
        Chem.GetSymmSSSR(mol)
    except Exception:
        pass
    return mol


def run_batch(
    pdb_dir: Path,
    output_dir: Path,
    weights_path: Path,
    hidden_dim: int = 128,
    num_layers: int = 6,
):
    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    if not pdb_files:
        raise ValueError(f"No PDB files found in {pdb_dir}")

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    with open(weights_path, "rb") as f:
        variables = pickle.load(f)

    model = PhyNEO_GNN_V2(hidden_dim=hidden_dim, num_layers=num_layers)
    rng = jax.random.PRNGKey(0)

    output_dir.mkdir(parents=True, exist_ok=True)

    success = []
    failed = []

    for i, pdb_path in enumerate(pdb_files, start=1):
        name = pdb_path.stem
        try:
            rdmol = load_rdkit_pdb(pdb_path)
            mol_obj = Molecule(rdmol, name=name)
            graph = mol_obj.get_graph()
            total_q = float(Chem.GetFormalCharge(rdmol))

            # Ensure model shape compatibility on first call.
            if i == 1:
                _ = model.init(rng, graph, total_charge=total_q)

            params = model.apply(variables, graph, total_charge=total_q)

            mol_out_dir = output_dir / name
            mol_out_dir.mkdir(parents=True, exist_ok=True)
            xml_path = mol_out_dir / "FF.xml"
            pdb_out_path = mol_out_dir / "structure.pdb"

            generate_dmff_xml(rdmol, params, str(xml_path), include_local_frames=True)
            shutil.copy2(pdb_path, pdb_out_path)

            success.append(
                {
                    "name": name,
                    "input_pdb": str(pdb_path),
                    "xml": str(xml_path),
                    "pdb": str(pdb_out_path),
                    "total_charge": total_q,
                    "num_atoms": int(rdmol.GetNumAtoms()),
                }
            )
            print(f"[{i:03d}/{len(pdb_files):03d}] OK   {name}")
        except Exception as e:
            failed.append({"name": name, "input_pdb": str(pdb_path), "error": str(e)})
            print(f"[{i:03d}/{len(pdb_files):03d}] FAIL {name} -> {e}")

    summary = {
        "pdb_dir": str(pdb_dir),
        "output_dir": str(output_dir),
        "weights_path": str(weights_path),
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "total": len(pdb_files),
        "success_count": len(success),
        "failed_count": len(failed),
        "success": success,
        "failed": failed,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Batch inference done ===")
    print(f"Total: {len(pdb_files)} | Success: {len(success)} | Failed: {len(failed)}")
    print(f"Summary: {output_dir / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch infer DMFF XML from a folder of PDB files using trained PhyNEO weights."
    )
    parser.add_argument(
        "--pdb-dir",
        default="/Users/jeremychen/Desktop/Project/project_electrolyte/0_paper_revision/repo/all_data/pdb_bank",
        help="Folder containing input *.pdb files",
    )
    parser.add_argument(
        "--output-dir",
        default="results_pdb_bank_inference",
        help="Output directory for per-molecule FF.xml and copied structure.pdb",
    )
    parser.add_argument(
        "--weights",
        default="models/phyneo_production_v1.flax",
        help="Pickled Flax variables file",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=6)
    args = parser.parse_args()

    run_batch(
        pdb_dir=Path(args.pdb_dir),
        output_dir=Path(args.output_dir),
        weights_path=Path(args.weights),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )


if __name__ == "__main__":
    main()
