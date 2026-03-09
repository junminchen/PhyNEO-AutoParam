
import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path

try:
    import pyscf
    from pyscf import gto, dft, lib
except ImportError:
    print("Error: PySCF is not installed. Please install it via 'pip install pyscf'.")
    sys.exit(1)

try:
    from grid import AtomGrid, MolGrid, BeckeWeights
    from grid.molgrid import _generate_default_rgrid
    from horton_part import MBISWPart
except ImportError:
    print("Error: HORTON 3 / qc-grid components not found.")
    print("Please install them via: pip install horton-part qc-grid qc-iodata")
    sys.exit(1)

class MoleculeAnalyzer:
    """
    Auto-Multipol Analyzer using PySCF for Electronic Structure 
    and HORTON 3/MBIS for density partitioning.
    """
    def __init__(self, input_path, basis='aug-cc-pVTZ', xc='wb97x-d', charge=0, spin=0, use_gpu=False):
        self.input_path = Path(input_path)
        self.basis = basis
        self.xc = xc
        self.charge = charge
        self.spin = spin
        self.use_gpu = use_gpu
        self.mol = None
        self.mf = None
        
        # Free atom reference data (TS Model & Extended for C8/C10)
        # alpha (Bohr^3), volume (Bohr^3), C6 (a.u.), C8 (a.u.), C10 (a.u.)
        self.atom_ref = {
            'H':  {'alpha_free': 4.5,  'vol_free': 14.1, 'c6_free': 6.5,   'c8_free': 124.4,  'c10_free': 3286.0},
            'C':  {'alpha_free': 12.0, 'vol_free': 38.2, 'c6_free': 46.6,  'c8_free': 1350.0, 'c10_free': 49000.0},
            'N':  {'alpha_free': 7.4,  'vol_free': 25.4, 'c6_free': 24.2,  'c8_free': 760.0,  'c10_free': 26000.0},
            'O':  {'alpha_free': 5.4,  'vol_free': 18.1, 'c6_free': 15.6,  'c8_free': 540.0,  'c10_free': 18000.0},
            'F':  {'alpha_free': 3.8,  'vol_free': 12.1, 'c6_free': 9.5,   'c8_free': 320.0,  'c10_free': 10000.0},
            'S':  {'alpha_free': 19.6, 'vol_free': 75.2, 'c6_free': 134.0, 'c8_free': 4000.0, 'c10_free': 120000.0},
            'Cl': {'alpha_free': 15.0, 'vol_free': 59.8, 'c6_free': 94.6,  'c8_free': 3200.0, 'c10_free': 100000.0},
        }

    def build_mol(self):
        """Build PySCF molecule object from XYZ."""
        print(f"--- Building Molecule from {self.input_path} ---")
        self.mol = gto.M(
            atom=str(self.input_path),
            basis=self.basis,
            charge=self.charge,
            spin=self.spin,
            verbose=4
        )
        return self.mol

    def run_scf(self):
        """Execute DFT calculation with intelligent grid selection."""
        if self.mol is None:
            self.build_mol()
        
        print(f"--- Running DFT ({self.xc}/{self.basis}) {'on GPU (gpu4pyscf)' if self.use_gpu else 'on CPU'} ---")
        
        if self.use_gpu:
            try:
                from gpu4pyscf import dft as gdft
                self.mf = gdft.RKS(self.mol)
            except ImportError:
                print("Warning: gpu4pyscf not found. Falling back to CPU PySCF.")
                self.mf = dft.RKS(self.mol)
        else:
            self.mf = dft.RKS(self.mol)
            
        self.mf.xc = self.xc
        
        # --- Advanced Grid Intelligence ---
        # High-quality functionals (Hybrids/Meta-GGAs) need denser grids for alpha accuracy
        xc_lower = self.xc.lower()
        is_high_quality = any(m in xc_lower for m in ['wb97', 'm06', 'scan', 'cam-b3lyp', 'b97m'])
        if is_high_quality:
            print("--- High-quality functional detected: Setting grid level to 4 for better density tails ---")
            self.mf.grids.level = 4
            
        self.mf.kernel()
        
        if not self.mf.converged:
            raise RuntimeError("SCF calculation did not converge.")
        return self.mf

    def perform_mbis_partitioning(self):
        """
        Partition electronic density using MBIS logic (HORTON 3).
        """
        print("--- Performing MBIS Partitioning (HORTON 3) ---")
        
        numbers = self.mol.atom_charges()
        coordinates = self.mol.atom_coords() # Bohr
        
        # 1. Setup HORTON MolGrid
        atgrids = []
        for i in range(self.mol.natm):
            rgrid = _generate_default_rgrid(numbers[i])
            # Use degree 15 for better accuracy (comparable to Multiwfn/PySCF high levels)
            atgrid = AtomGrid(rgrid, [15], center=coordinates[i])
            atgrids.append(atgrid)
        
        molgrid = MolGrid(numbers, atgrids, BeckeWeights(), store=True)
        
        # 2. Evaluate PySCF density on this grid
        dm = self.mf.make_rdm1()
        ao_value = dft.numint.eval_ao(self.mol, molgrid.points, deriv=0)
        rho = dft.numint.eval_rho(self.mol, ao_value, dm, xctype='LDA')
        
        # 3. MBIS Iteration
        wpart = MBISWPart(coordinates, numbers, numbers, molgrid, rho)
        wpart.do_partitioning()
        
        # 4. Extract Charges and Moments
        wpart.do_charges()
        wpart.do_moments()
        
        # wpart.cache contains 'charges', 'populations', 'cartesian_multipoles', 'radial_moments'
        charges = wpart.cache['charges']
        cart_mult = wpart.cache['cartesian_multipoles']
        radial_moments = wpart.cache['radial_moments']
        
        results = []
        for i in range(self.mol.natm):
            symbol = self.mol.atom_symbol(i)
            # cart_mult[i] is order-sorted Cartesian moments:
            # Order 0: [q]
            # Order 1: [mux, muy, muz]
            # Order 2: [Qxx, Qyy, Qzz, Qxy, Qxz, Qyz]
            m = cart_mult[i]
            
            results.append({
                'index': i,
                'element': symbol,
                'charge': float(charges[i]),
                'dipole': [float(x) for x in m[1:4]],
                'quadrupole': [
                    [float(m[4]), float(m[7]), float(m[8])], # XX, XY, XZ
                    [float(m[7]), float(m[5]), float(m[9])], # YX, YY, YZ
                    [float(m[8]), float(m[9]), float(m[6])]  # ZX, ZY, ZZ
                ],
                'volume_eff': float(radial_moments[i, 3]) # TS volume = 3rd radial moment
            })
        return results

    def derive_polarizability(self, mbis_data):
        """Derive atomic polarizability using TS scaling model."""
        print("--- Deriving Atomic Polarizabilities (TS Model) ---")
        for atom in mbis_data:
            elem = atom['element']
            ref = self.atom_ref.get(elem, self.atom_ref['C']) # fallback to C
            
            # alpha_mol = alpha_free * (V_eff / V_free)
            ratio = atom['volume_eff'] / ref['vol_free']
            atom['alpha_iso'] = ref['alpha_free'] * ratio
        return mbis_data

    def derive_dispersion(self, mbis_data):
        """Derive C6, C8 and C10 coefficients using TS-style scaling rules."""
        print("--- Deriving Dispersion Coefficients (C6, C8, C10) ---")
        n = len(mbis_data)
        c6_matrix = np.zeros((n, n))
        c8_matrix = np.zeros((n, n))
        c10_matrix = np.zeros((n, n))
        
        # 1. Compute diagonal ii terms for each atom
        for i in range(n):
            elem = mbis_data[i]['element']
            ref = self.atom_ref.get(elem, self.atom_ref['C'])
            ratio = mbis_data[i]['volume_eff'] / ref['vol_free']
            
            # C6 scales with V^2
            mbis_data[i]['c6_ii'] = ref['c6_free'] * (ratio ** 2.0)
            # C8 scales with V^(8/3) 
            mbis_data[i]['c8_ii'] = ref['c8_free'] * (ratio ** (8.0/3.0))
            # C10 scales with V^(10/3)
            mbis_data[i]['c10_ii'] = ref['c10_free'] * (ratio ** (10.0/3.0))
            
        # 2. Compute C_ij using combination rules
        for i in range(n):
            for j in range(i, n):
                # C6 Combination (TS-standard)
                alpha_i = mbis_data[i]['alpha_iso']
                alpha_j = mbis_data[j]['alpha_iso']
                c6_ii = mbis_data[i]['c6_ii']
                c6_jj = mbis_data[j]['c6_ii']
                
                denom_c6 = (alpha_j/alpha_i) * c6_ii + (alpha_i/alpha_j) * c6_jj
                c6_val = (2.0 * c6_ii * c6_jj) / (denom_c6 + 1e-12)
                c6_matrix[i, j] = c6_matrix[j, i] = c6_val
                
                # C8 & C10 Combination (Geometric mean approximation)
                c8_val = np.sqrt(mbis_data[i]['c8_ii'] * mbis_data[j]['c8_ii'])
                c8_matrix[i, j] = c8_matrix[j, i] = c8_val
                
                c10_val = np.sqrt(mbis_data[i]['c10_ii'] * mbis_data[j]['c10_ii'])
                c10_matrix[i, j] = c10_matrix[j, i] = c10_val
                
        return c6_matrix.tolist(), c8_matrix.tolist(), c10_matrix.tolist()

    def run_pipeline(self):
        """Full execution pipeline."""
        self.run_scf()
        mbis_data = self.perform_mbis_partitioning()
        mbis_data = self.derive_polarizability(mbis_data)
        c6_matrix, c8_matrix, c10_matrix = self.derive_dispersion(mbis_data)
        
        return {
            'molecule': self.input_path.stem,
            'atoms': mbis_data,
            'c6_matrix': c6_matrix,
            'c8_matrix': c8_matrix,
            'c10_matrix': c10_matrix
        }

def main():
    parser = argparse.ArgumentParser(description="Auto-Multipol (PySCF+Horton MBIS Edition)")
    parser.add_argument("-i", "--input", required=True, help="Input XYZ file")
    parser.add_argument("-b", "--basis", default="aug-cc-pVTZ", help="Basis set")
    parser.add_argument("-f", "--functional", default="pbe0", help="DFT Functional")
    parser.add_argument("-o", "--output", default="params.json", help="Output JSON file")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration via gpu4pyscf")
    
    args = parser.parse_args()
    
    try:
        analyzer = MoleculeAnalyzer(
            args.input, 
            basis=args.basis, 
            xc=args.functional,
            use_gpu=args.gpu
        )
        results = analyzer.run_pipeline()
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nSuccess! Results saved to {args.output}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
