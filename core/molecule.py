import jax.numpy as jnp
import jraph
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Optional, List, Dict, Any

class Molecule:
    """
    Core Molecule class handling cheminformatics and Graph construction.
    Inspired by bytemol/core.
    """
    def __init__(self, rdmol: Chem.Mol, name: str = "Unknown"):
        self.rdmol = rdmol
        self.name = name
        self.num_atoms = rdmol.GetNumAtoms()
        
    @classmethod
    def from_smiles(cls, smiles: str, name: Optional[str] = None):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        
        # Initialize PDBResidueInfo for all atoms to allow naming
        for atom in mol.GetAtoms():
            info = Chem.AtomPDBResidueInfo()
            info.SetResidueName("MOL")
            info.SetResidueNumber(1)
            atom.SetPDBResidueInfo(info)
            
        return cls(mol, name or smiles)

    @classmethod
    def from_xyz(cls, xyz_path: str, name: Optional[str] = None):
        # Basic XYZ parser + RDKit connectivity inference
        mol = Chem.MolFromXYZFile(xyz_path)
        if mol is None:
            # Fallback: parse manually and use ConnectTheDots or similar
            with open(xyz_path, 'r') as f:
                lines = f.readlines()
            # (Simplified for now: assumes RDKit can handle standard XYZ)
            pass
        return cls(mol, name or Path(xyz_path).stem)

    def get_graph(self) -> jraph.GraphsTuple:
        """
        Converts the molecule into a jraph.GraphsTuple with enriched features.
        """
        node_features = []
        for atom in self.rdmol.GetAtoms():
            features = [
                float(atom.GetAtomicNum()),
                float(atom.GetHybridization()), # RDKit HybridizationType enum
                float(atom.GetFormalCharge()),
                float(atom.GetTotalValence()),
                float(atom.GetImplicitValence()),
                float(atom.IsInRing()),
                float(atom.GetIsAromatic())
            ]
            node_features.append(features)
        
        # 2. Edge Features: [Bond Type, Is_Conjugated, Is_In_Ring, Is_Stereo]
        senders = []
        receivers = []
        edge_features = []
        
        for bond in self.rdmol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            
            # Bi-directional edges
            senders.extend([u, v])
            receivers.extend([v, u])
            
            b_feat = [
                float(bond.GetBondTypeAsDouble()),
                float(bond.GetIsConjugated()),
                float(bond.IsInRing()),
                float(bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE)
            ]
            edge_features.extend([b_feat, b_feat])

        return jraph.GraphsTuple(
            nodes=jnp.array(node_features),
            edges=jnp.array(edge_features),
            senders=jnp.array(senders),
            receivers=jnp.array(receivers),
            n_node=jnp.array([self.num_atoms]),
            n_edge=jnp.array([len(senders)]),
            globals=None
        )

    def get_coords(self) -> jnp.ndarray:
        """Extracts 3D coordinates in Angstrom."""
        conf = self.rdmol.GetConformer()
        pos = conf.GetPositions()
        return jnp.array(pos)
