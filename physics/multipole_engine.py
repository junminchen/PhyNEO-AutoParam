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
    Supports physical derivation of C6, C8, C10 and Slater B.
    """
    def __init__(self, input_path, basis='def2-TZVP', xc='wb97xd', charge=0, spin=0, use_gpu=False):
        self.input_path = Path(input_path)
        self.basis = basis
        self.xc = xc
        self.charge = charge
        self.spin = spin
        self.use_gpu = use_gpu
        self.mol = None
        self.mf = None
        
        # Free atom reference data (TS Model & Extended for C8/C10/SlaterB)
        # alpha (Bohr^3), volume (Bohr^3), C6 (a.u.), C8 (a.u.), C10 (a.u.), B_ref (nm^-1)
        self.atom_ref = {
            'H':  {'alpha_free': 4.5,  'vol_free': 14.1, 'c6_free': 6.5,   'c8_free': 124.4,  'c10_free': 3286.0,   'b_ref': 40.0},
            'Li': {'alpha_free': 164.0,'vol_free': 110.0,'c6_free': 1390.0,'c8_free': 40000.0,'c10_free': 800000.0, 'b_ref': 30.0},
            'Be': {'alpha_free': 37.7, 'vol_free': 65.0, 'c6_free': 214.0, 'c8_free': 8000.0, 'c10_free': 150000.0, 'b_ref': 32.0},
            'B':  {'alpha_free': 20.5, 'vol_free': 45.0, 'c6_free': 99.5,  'c8_free': 3500.0, 'c10_free': 80000.0,  'b_ref': 34.0},
            'C':  {'alpha_free': 12.0, 'vol_free': 38.2, 'c6_free': 46.6,  'c8_free': 1350.0, 'c10_free': 49000.0,  'b_ref': 35.0},
            'N':  {'alpha_free': 7.4,  'vol_free': 25.4, 'c6_free': 24.2,  'c8_free': 760.0,  'c10_free': 26000.0,  'b_ref': 36.0},
            'O':  {'alpha_free': 5.4,  'vol_free': 18.1, 'c6_free': 15.6,  'c8_free': 540.0,  'c10_free': 18000.0,  'b_ref': 37.0},
            'F':  {'alpha_free': 3.8,  'vol_free': 12.1, 'c6_free': 9.5,   'c8_free': 320.0,  'c10_free': 10000.0,  'b_ref': 38.0},
            'Na': {'alpha_free': 162.7,'vol_free': 150.0,'c6_free': 1550.0,'c8_free': 50000.0,'c10_free': 900000.0, 'b_ref': 28.0},
            'Mg': {'alpha_free': 71.2, 'vol_free': 105.0,'c6_free': 627.0, 'c8_free': 25000.0,'c10_free': 500000.0, 'b_ref': 30.0},
            'Al': {'alpha_free': 57.8, 'vol_free': 90.0, 'c6_free': 528.0, 'c8_free': 20000.0,'c10_free': 400000.0, 'b_ref': 31.0},
            'Si': {'alpha_free': 37.3, 'vol_free': 75.0, 'c6_free': 305.0, 'c8_free': 12000.0,'c10_free': 250000.0, 'b_ref': 32.0},
            'P':  {'alpha_free': 25.0, 'vol_free': 65.0, 'c6_free': 185.0, 'c8_free': 7000.0, 'c10_free': 150000.0, 'b_ref': 33.0},
            'S':  {'alpha_free': 19.6, 'vol_free': 75.2, 'c6_free': 134.0, 'c8_free': 4000.0, 'c10_free': 120000.0, 'b_ref': 34.0},
            'Cl': {'alpha_free': 15.0, 'vol_free': 59.8, 'c6_free': 94.6,  'c8_free': 3200.0, 'c10_free': 100000.0, 'b_ref': 35.0},
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
        """Partition electronic density using MBIS logic (HORTON 3)."""
        print("--- Performing MBIS Partitioning (HORTON 3) ---")
        
        numbers = self.mol.atom_charges()
        coordinates = self.mol.atom_coords() # Bohr
        atgrids = []
        for i in range(self.mol.natm):
            rgrid = _generate_default_rgrid(numbers[i])
            atgrid = AtomGrid(rgrid, [15], center=coordinates[i])
            atgrids.append(atgrid)
        
        molgrid = MolGrid(numbers, atgrids, BeckeWeights(), store=True)
        dm = self.mf.make_rdm1()
        ao_value = dft.numint.eval_ao(self.mol, molgrid.points, deriv=0)
        rho = dft.numint.eval_rho(self.mol, ao_value, dm, xctype='LDA')
        
        wpart = MBISWPart(coordinates, numbers, numbers, molgrid, rho)
        wpart.do_partitioning()
        wpart.do_charges()
        wpart.do_moments()
        
        charges = wpart.cache['charges']
        cart_mult = wpart.cache['cartesian_multipoles']
        radial_moments = wpart.cache['radial_moments']
        
        results = []
        for i in range(self.mol.natm):
            symbol = self.mol.atom_symbol(i)
            m = cart_mult[i]
            results.append({
                'index': i,
                'element': symbol,
                'charge': float(charges[i]),
                'dipole': [float(x) for x in m[1:4]],
                'quadrupole': [
                    [float(m[4]), float(m[7]), float(m[8])],
                    [float(m[7]), float(m[5]), float(m[9])],
                    [float(m[8]), float(m[9]), float(m[6])]
                ],
                'volume_eff': float(radial_moments[i, 3])
            })
        return results

    def derive_physical_params(self, mbis_data):
        """Derive alpha, C6, C8, C10 and Slater B using physical scaling laws."""
        print("--- Deriving Physical Parameters (alpha, C6, C8, C10, Slater B) ---")
        n = len(mbis_data)
        
        for i in range(n):
            elem = mbis_data[i]['element']
            ref = self.atom_ref.get(elem, self.atom_ref['C'])
            ratio = mbis_data[i]['volume_eff'] / ref['vol_free']
            
            # alpha prop to V
            mbis_data[i]['alpha_iso'] = ref['alpha_free'] * ratio
            # Cn scaling
            mbis_data[i]['c6_ii'] = ref['c6_free'] * (ratio ** 2.0)
            mbis_data[i]['c8_ii'] = ref['c8_free'] * (ratio ** (8.0/3.0))
            mbis_data[i]['c10_ii'] = ref['c10_free'] * (ratio ** (10.0/3.0))
            
            # Slater B scaling: B prop to V^-1/3
            # Ensures atoms have unique, density-dependent decay constants
            mbis_data[i]['slater_b_init'] = ref['b_ref'] * (ratio ** (-1.0/3.0))
            
        return mbis_data

    def run_pipeline(self):
        """Full execution pipeline."""
        self.run_scf()
        mbis_data = self.perform_mbis_partitioning()
        mbis_data = self.derive_physical_params(mbis_data)
        return {
            'molecule': self.input_path.stem,
            'atoms': mbis_data
        }

def main():
    parser = argparse.ArgumentParser(description="Auto-Multipol Physics Engine (PhyNEO Edition)")
    parser.add_argument("-i", "--input", required=True, help="Input XYZ file")
    parser.add_argument("-b", "--basis", default="aug-cc-pVTZ", help="Basis set")
    parser.add_argument("-f", "--functional", default="cam-b3lyp", help="DFT Functional")
    parser.add_argument("-o", "--output", default="params.json", help="Output JSON file")
    parser.add_argument("--charge", type=int, default=0, help="Total charge of the molecule")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration")
    
    args = parser.parse_args()
    try:
        analyzer = MoleculeAnalyzer(args.input, basis=args.basis, xc=args.functional, use_gpu=args.gpu, charge=args.charge)
        results = analyzer.run_pipeline()
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nSuccess! Results saved to {args.output}")
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
