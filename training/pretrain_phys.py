import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from models.gnn_jax import PhyNEO_GNN_V2
from core.molecule import Molecule
import jraph
import json
import os
from typing import List, Dict, Any

def load_physical_dataset(results_dir: str):
    """
    Loads molecules and their physical anchors (MBIS charges, volume_eff)
    from the results/ directory.
    """
    dataset = []
    # Hardcoded mapping for demo/pretrain - in production use data_distill.py
    # Each subdir in results/ contains results_high.json
    for mol_name in os.listdir(results_dir):
        json_path = os.path.join(results_dir, mol_name, "results_high.json")
        if not os.path.exists(json_path):
            continue
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # 1. Construct Molecule/Graph
        # Note: In a real case, we'd need the SMILES or XYZ. 
        # For EC we know the SMILES, for ions we can infer from name.
        name_to_smiles = {
            "EC": "C1COC(=O)O1",
            "Li": "[Li+]",
            "Na": "[Na+]",
            "PF6": "F[P-](F)(F)(F)(F)F"
        }
        smiles = name_to_smiles.get(mol_name)
        if not smiles: continue
        
        mol = Molecule.from_smiles(smiles)
        graph = mol.get_graph()
        
        # 2. Extract Labels
        atoms_data = data['atoms']
        q_ref = jnp.array([a['charge'] for a in atoms_data])
        # Scale volume_eff to a reasonable kappa range (e.g., / 20.0)
        k_ref = jnp.array([a['volume_eff'] / 20.0 for a in atoms_data])
        total_q = float(sum(q_ref))
        
        dataset.append({
            "name": mol_name,
            "graph": graph,
            "q_ref": q_ref,
            "k_ref": k_ref,
            "total_q": total_q
        })
        print(f"Loaded {mol_name} with {len(q_ref)} atoms (Total Q: {total_q:.1f})")
    
    return dataset

def train_phys():
    # 1. Setup
    results_dir = "examples/production_results"
    dataset = load_physical_dataset(results_dir)
    if not dataset:
        print("No data found in results/ directory!")
        return

    model = PhyNEO_GNN_V2(hidden_dim=64, num_layers=3)
    optimizer = optax.adam(learning_rate=1e-3)
    
    # Initialize variables
    rng = jax.random.PRNGKey(0)
    sample_item = dataset[0]
    variables = model.init(rng, sample_item['graph'], total_charge=sample_item['total_q'])
    opt_state = optimizer.init(variables)

    # 2. Loss Function
    @jax.jit
    def loss_fn(params, graph, q_ref, k_ref, total_q):
        out = model.apply(params, graph, total_charge=total_q)
        
        # QEq Charge Loss
        q_loss = jnp.mean((out['q'] - q_ref)**2)
        # Volume/Kappa Loss
        k_loss = jnp.mean((out['kappa'] - k_ref)**2)
        
        return q_loss + 0.1 * k_loss

    @jax.jit
    def train_step(variables, opt_state, graph, q_ref, k_ref, total_q):
        val_grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = val_grad_fn(variables, graph, q_ref, k_ref, total_q)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_variables = optax.apply_updates(variables, updates)
        return new_variables, new_opt_state, loss

    # 3. Training Loop
    print("\n--- Starting Phase A: Physical Pre-training ---")
    num_epochs = 500
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for item in dataset:
            variables, opt_state, loss = train_step(
                variables, opt_state, 
                item['graph'], item['q_ref'], item['k_ref'], item['total_q']
            )
            epoch_loss += loss
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | Total Loss: {epoch_loss:.6f}")

    # 4. Final Verification (EC)
    print("\n--- Final Verification (EC) ---")
    ec_item = [d for d in dataset if d['name'] == 'EC'][0]
    out = model.apply(variables, ec_item['graph'], total_charge=ec_item['total_q'])
    
    print(f"{'Atom':<6} | {'Q_Ref':<10} | {'Q_Pred':<10} | {'Diff':<10}")
    print("-" * 45)
    for i in range(len(ec_item['q_ref'])):
        q_r = ec_item['q_ref'][i]
        q_p = out['q'][i]
        print(f"Atom {i:<2} | {q_r:10.4f} | {q_p:10.4f} | {abs(q_r-q_p):10.4f}")

if __name__ == "__main__":
    train_phys()
