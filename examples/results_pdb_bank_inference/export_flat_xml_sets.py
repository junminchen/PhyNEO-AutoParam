import argparse
import json
import pickle
import sys
from pathlib import Path

import jax
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


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
    if charge_map_path is None or not charge_map_path.exists():
        return {}
    return {str(k): float(v) for k, v in json.loads(charge_map_path.read_text()).items()}


def resolve_total_charge(name: str, rdmol, charge_map: dict):
    if name in charge_map:
        return float(charge_map[name])
    if name in DEFAULT_NAME_CHARGES:
        return float(DEFAULT_NAME_CHARGES[name])
    return float(Chem.GetFormalCharge(rdmol))


def build_mol_from_xyz(xyz_path: Path, total_charge: int):
    mol = Chem.MolFromXYZBlock(xyz_path.read_text())
    if mol is None:
        raise ValueError(f"Cannot parse XYZ: {xyz_path}")
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=int(total_charge), embedChiral=False)
    except Exception:
        # Alkali ions and some unusual fragments do not support RDKit bond ordering.
        pass
    mol.UpdatePropertyCache(strict=False)
    return mol


def normalize_pdb_names(mol, residue_name: str):
    pdb_resname = f"{residue_name[:3]:>3}"
    for i, atom in enumerate(mol.GetAtoms()):
        info = Chem.AtomPDBResidueInfo()
        info.SetName(f"{atom.GetSymbol()}{i:<3}"[:4])
        info.SetResidueName(pdb_resname)
        info.SetResidueNumber(1)
        atom.SetMonomerInfo(info)


def export_production_results(prod_root: Path, xml_dir: Path, pdb_dir: Path):
    xml_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for json_path in sorted(prod_root.glob("*/results_high.json")):
        name = json_path.parent.name
        xyz_path = json_path.with_name(f"{name}.xyz")
        if not xyz_path.exists():
            print(f"skip {name}: missing XYZ")
            continue

        data = json.loads(json_path.read_text())
        atoms = data["atoms"]
        total_charge = int(round(sum(atom["charge"] for atom in atoms)))
        mol = build_mol_from_xyz(xyz_path, total_charge)

        params = {
            "q": np.array([atom["charge"] for atom in atoms], dtype=float),
            "dipole": np.array([atom["dipole"] for atom in atoms], dtype=float),
            "quadrupole": np.array([atom["quadrupole"] for atom in atoms], dtype=float),
            "alpha_iso": np.array([atom["alpha_iso"] for atom in atoms], dtype=float),
            "c6_ii": np.array([atom["c6_ii"] for atom in atoms], dtype=float),
            "c8_ii": np.array([atom["c8_ii"] for atom in atoms], dtype=float),
            "c10_ii": np.array([atom["c10_ii"] for atom in atoms], dtype=float),
            "slater_b_init": np.array([atom["slater_b_init"] for atom in atoms], dtype=float),
        }

        xml_path = xml_dir / f"{name}.xml"
        pdb_path = pdb_dir / f"{name}.pdb"
        generate_dmff_xml(
            mol,
            params,
            str(xml_path),
            residue_name=name,
            unit_style="physical_au",
            include_local_frames=True,
        )
        normalize_pdb_names(mol, name)
        Chem.MolToPDBFile(mol, str(pdb_path))
        summary.append({"name": name, "xml": str(xml_path), "pdb": str(pdb_path), "charge": total_charge})
        print(f"production -> {name}")
    return summary


def export_model_results(
    pdb_root: Path,
    weights_path: Path,
    xml_dir: Path,
    pdb_dir: Path,
    charge_map_path: Path | None = None,
    hidden_dim: int = 128,
    num_layers: int = 6,
):
    xml_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir.mkdir(parents=True, exist_ok=True)

    charge_map = load_charge_map(charge_map_path)
    with open(weights_path, "rb") as f:
        variables = pickle.load(f)
    model = PhyNEO_GNN_V2(hidden_dim=hidden_dim, num_layers=num_layers)
    rng = jax.random.PRNGKey(0)

    summary = []
    for i, pdb_path in enumerate(sorted(pdb_root.glob("*.pdb")), start=1):
        name = pdb_path.stem
        rdmol = load_rdkit_pdb(pdb_path)
        mol_obj = Molecule(rdmol, name=name)
        graph = mol_obj.get_graph()
        total_charge = resolve_total_charge(name, rdmol, charge_map)
        if i == 1:
            _ = model.init(rng, graph, total_charge=total_charge)
        params = model.apply(variables, graph, total_charge=total_charge)

        dst_xml = xml_dir / f"{name}.xml"
        dst_pdb = pdb_dir / f"{name}.pdb"
        generate_dmff_xml(
            rdmol,
            params,
            str(dst_xml),
            residue_name=name,
            include_local_frames=True,
        )
        normalize_pdb_names(rdmol, name)
        Chem.MolToPDBFile(rdmol, str(dst_pdb))
        summary.append({"name": name, "xml": str(dst_xml), "pdb": str(dst_pdb), "charge": total_charge})
        print(f"model -> {name}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Collect flat XML/PDB exports for model and production-results versions.")
    parser.add_argument(
        "--model-pdb-root",
        default="/Users/jeremychen/Desktop/Project/project_electrolyte/0_paper_revision/repo/all_data/pdb_bank",
    )
    parser.add_argument(
        "--weights",
        default="/Users/jeremychen/Desktop/Project/repo/PhyNEO-AutoParam/models/phyneo_production_v1.flax",
    )
    parser.add_argument(
        "--charge-map",
        default="/Users/jeremychen/Desktop/Project/repo/PhyNEO-AutoParam/examples/results_pdb_bank_inference/charge_map.json",
    )
    parser.add_argument(
        "--production-root",
        default="/Users/jeremychen/Desktop/Project/repo/PhyNEO-AutoParam/examples/production_results",
    )
    parser.add_argument(
        "--output-root",
        default="/Users/jeremychen/Desktop/Project/repo/PhyNEO-AutoParam/examples/results_pdb_bank_inference/flat_exports",
    )
    args = parser.parse_args()

    out_root = Path(args.output_root)
    model_xml_dir = out_root / "model_xml"
    model_pdb_dir = out_root / "model_pdb"
    prod_xml_dir = out_root / "production_xml"
    prod_pdb_dir = out_root / "production_pdb"

    model_summary = export_model_results(
        Path(args.model_pdb_root),
        Path(args.weights),
        model_xml_dir,
        model_pdb_dir,
        Path(args.charge_map) if args.charge_map else None,
    )
    production_summary = export_production_results(Path(args.production_root), prod_xml_dir, prod_pdb_dir)

    summary = {
        "model_count": len(model_summary),
        "production_count": len(production_summary),
        "model": model_summary,
        "production": production_summary,
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
