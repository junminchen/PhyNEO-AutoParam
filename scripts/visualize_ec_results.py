import matplotlib.pyplot as plt
import jax.numpy as jnp
import pickle
import json
import os
from core.molecule import Molecule
from models.gnn_jax import PhyNEO_GNN_V2
from rdkit import Chem

def generate_ec_physics_plots():
    # 1. Load Data
    weights_path = "models/phyneo_production_v1.flax"
    ref_path = "examples/production_results/EC/results_high.json"
    # Canonical SMILES for EC with hydrogens
    smiles = "C1COC(=O)O1" 
    os.makedirs("examples/plots", exist_ok=True)

    with open(weights_path, "rb") as f:
        variables = pickle.load(f)
    
    mol_obj = Molecule.from_smiles(smiles)
    model = PhyNEO_GNN_V2(hidden_dim=128, num_layers=6)
    # Ensure total charge is 0 for EC
    pred = model.apply(variables, mol_obj.get_graph(), total_charge=0.0)

    with open(ref_path, "r") as f:
        ref_data = json.load(f)
    
    # 2. Setup Figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    
    configs = [
        ("Charge (e)", pred['q'], [a['charge'] for a in ref_data['atoms']], "plasma"),
        ("Polarizability (Ang^3)", pred['alpha'], [a['alpha_iso'] for a in ref_data['atoms']], "viridis"),
        ("C6 Dispersion", pred['c6'], [a['c6_ii'] for a in ref_data['atoms']], "coolwarm"),
        ("C8 Dispersion", pred['c8'], [a['c8_ii'] for a in ref_data['atoms']], "YlOrRd"),
        ("C10 Dispersion", pred['c10'], [a['c10_ii'] for a in ref_data['atoms']], "PuBu")
    ]

    for i, (title, y_p, y_r, cmap) in enumerate(configs):
        y_p = jnp.array(y_p).flatten()
        y_r = jnp.array(y_r).flatten()
        
        axes[i].scatter(y_r, y_p, c=y_r, cmap=cmap, s=100, edgecolors='k', alpha=0.8)
        
        # Perfect match line
        mn, mx = float(jnp.min(y_r)), float(jnp.max(y_r))
        # Add buffer
        pad = (mx - mn) * 0.1 if mx != mn else 0.1
        axes[i].plot([mn-pad, mx+pad], [mn-pad, mx+pad], 'k--', alpha=0.3)
        
        axes[i].set_title(title, fontsize=15, fontweight='bold')
        axes[i].set_xlabel('Reference (DFT)', fontsize=12)
        axes[i].set_ylabel('Predicted (PhyNEO)', fontsize=12)
        
        mae = jnp.mean(jnp.abs(y_p - y_r))
        print(f"  - {title}: MAE = {mae:.6f}")
        axes[i].text(0.05, 0.95, f"MAE: {mae:.4f}", transform=axes[i].transAxes, 
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        axes[i].grid(alpha=0.3)

    # Clean up empty slot
    fig.delaxes(axes[5])
    plt.tight_layout()
    
    output_path = "examples/plots/ec_all_physics_parity.png"
    plt.savefig(output_path, dpi=300)
    print(f"EC combined physics parity plot saved to {output_path}")

if __name__ == "__main__":
    generate_ec_physics_plots()
