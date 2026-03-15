import argparse
import json
import pickle
import sys
from pathlib import Path

import jax
from rdkit import Chem

# Allow running this script from outside repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.molecule import Molecule
from models.gnn_jax import PhyNEO_GNN_V2
from scripts.params_to_xml import generate_dmff_xml

DEFAULT_NAME_CHARGES = {
    "Li": 1.0,
    "Na": 1.0,
    "PF6": -1.0,
    "BF4": -1.0,
    "FSI": -1.0,
    "TFSI": -1.0,
    "NO3": -1.0,
    "BOB": -1.0,
    "DFOB": -1.0,
}


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


def load_charge_map(charge_map_path: Path | None):
    if charge_map_path is None:
        return {}
    with open(charge_map_path, "r") as f:
        data = json.load(f)
    return {str(k): float(v) for k, v in data.items()}


def resolve_total_charge(name: str, rdmol, user_charge_map: dict):
    if name in user_charge_map:
        return float(user_charge_map[name]), "user_map"
    if name in DEFAULT_NAME_CHARGES:
        return float(DEFAULT_NAME_CHARGES[name]), "builtin_map"
    return float(Chem.GetFormalCharge(rdmol)), "rdkit_formal_charge"


def run_batch(
    pdb_dir: Path,
    output_dir: Path,
    weights_path: Path,
    charge_map_path: Path | None = None,
    only_names: list[str] | None = None,
    residue_name: str | None = None,
    hidden_dim: int = 128,
    num_layers: int = 6,
):
    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    if not pdb_files:
        raise ValueError(f"No PDB files found in {pdb_dir}")
    if only_names:
        only_set = set(only_names)
        pdb_files = [p for p in pdb_files if p.stem in only_set]
        if not pdb_files:
            raise ValueError(f"No matched PDB files for --only {only_names} in {pdb_dir}")

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    user_charge_map = load_charge_map(charge_map_path)

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
            total_q, charge_source = resolve_total_charge(name, rdmol, user_charge_map)

            # Ensure model shape compatibility on first call.
            if i == 1:
                _ = model.init(rng, graph, total_charge=total_q)

            params = model.apply(variables, graph, total_charge=total_q)

            mol_out_dir = output_dir / name
            mol_out_dir.mkdir(parents=True, exist_ok=True)
            xml_path = mol_out_dir / f"{name}.xml"
            pdb_out_path = mol_out_dir / f"{name}.pdb"

            # Default residue naming: use molecule name.
            xml_resname = residue_name if residue_name else name
            generate_dmff_xml(
                rdmol,
                params,
                str(xml_path),
                residue_name=xml_resname,
                include_local_frames=True,
            )

            # Write a normalized PDB with unified residue naming.
            pdb_resname = f"{xml_resname[:3]:>3}"  # PDB residue name field is 3-char wide.
            for atom in rdmol.GetAtoms():
                info = atom.GetPDBResidueInfo()
                if info is None:
                    info = Chem.AtomPDBResidueInfo()
                    atom.SetPDBResidueInfo(info)
                info.SetResidueName(pdb_resname)
                info.SetResidueNumber(1)
            Chem.MolToPDBFile(rdmol, str(pdb_out_path))

            success.append(
                {
                    "name": name,
                    "input_pdb": str(pdb_path),
                    "xml": str(xml_path),
                    "pdb": str(pdb_out_path),
                    "total_charge": total_q,
                    "charge_source": charge_source,
                    "xml_residue_name": xml_resname,
                    "pdb_residue_name": pdb_resname.strip(),
                    "num_atoms": int(rdmol.GetNumAtoms()),
                }
            )
            print(f"[{i:03d}/{len(pdb_files):03d}] OK   {name}  (Q={total_q:+.1f}, {charge_source})")
        except Exception as e:
            failed.append({"name": name, "input_pdb": str(pdb_path), "error": str(e)})
            print(f"[{i:03d}/{len(pdb_files):03d}] FAIL {name} -> {e}")

    summary = {
        "pdb_dir": str(pdb_dir),
        "output_dir": str(output_dir),
        "weights_path": str(weights_path),
        "charge_map_path": str(charge_map_path) if charge_map_path else None,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "residue_name_mode": "per_molecule_name" if residue_name is None else f"fixed:{residue_name}",
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
        help="Output directory for per-molecule <name>.xml and <name>.pdb",
    )
    parser.add_argument(
        "--weights",
        default="models/phyneo_production_v1.flax",
        help="Pickled Flax variables file",
    )
    parser.add_argument(
        "--charge-map",
        default=None,
        help="Optional JSON path: {\"MolName\": total_charge}. Overrides builtin charge map.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset names, e.g. --only Li PF6 Na",
    )
    parser.add_argument(
        "--resname",
        default=None,
        help="Optional fixed residue name override. Default: use each molecule name.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=6)
    args = parser.parse_args()

    run_batch(
        pdb_dir=Path(args.pdb_dir),
        output_dir=Path(args.output_dir),
        weights_path=Path(args.weights),
        charge_map_path=Path(args.charge_map) if args.charge_map else None,
        only_names=args.only,
        residue_name=args.resname,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )


if __name__ == "__main__":
    main()
