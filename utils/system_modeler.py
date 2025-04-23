# --*-- conding:utf-8 --*--
# @time:4/23/25 15:40
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:system_modeler.py

# -*- coding: utf-8 -*-
import os
import numpy as np
import warnings
from typing import Dict, Any, Optional, Tuple

# Qiskit Nature Imports
from qiskit_nature.second_q.drivers import PySCFDriver, UnitsType, BasisType
from qiskit_nature.second_q import  QubitConverter, FermionicOpMapper
from qiskit_nature.second_q.mappers import QubitConverter, FermionicOpMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer, BaseTransformer
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.units import DistanceUnit
from qiskit_nature.settings import settings as qn_settings # For atomic numbers

# Qiskit Core Imports
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit

# Assume custom utils are available
try:
    from utils.pdb_system_builder import PDBSystemBuilder
    from utils.active_space_selector import ActiveSpaceSelector
except ImportError:
    print("Warning: Could not import custom utils (PDBSystemBuilder, ActiveSpaceSelector).")
    print("Please ensure these files exist or integrate their logic.")
    # Define dummy classes if needed for the script to load
    class PDBSystemBuilder:
         def __init__(self, pdb_path, charge, spin, basis): pass
         def build_mole(self): print("Dummy build_mole called"); return None # Or a dummy Mole object if needed
    class ActiveSpaceSelector:
         def __init__(self, threshold): pass
         def run_scf(self, mol): print("Dummy run_scf called"); return None # Or a dummy SCF result object
         def select_active_space_with_energy(self, mf, n_before_homo, n_after_lumo): print("Dummy select_active_space called"); return (None, None, None, None)


class QuantumSystemModeler:
    """
    Handles reading a PDB file, performing quantum chemistry modeling
    (including spin correction), and generating VQE inputs (Hamiltonian, Ansatz).
    """

    def __init__(self,
                 basis: BasisType = "sto3g",
                 active_space_config: Optional[Dict[str, Any]] = None,
                 mapper: Optional[FermionicOpMapper] = None):
        """
        Initializes the Quantum System Modeler.

        Args:
            basis (BasisType): The basis set to use (e.g., "sto3g").
            active_space_config (dict, optional): Configuration for active space selection.
                Example: {'method': 'energy', 'threshold': 0.6, 'n_before_homo': 1, 'n_after_lumo': 1}
                         or {'method': 'manual', 'num_electrons': X, 'num_spatial_orbitals': Y}
                Defaults to a simple full space if None.
            mapper (FermionicOpMapper, optional): The fermion-to-qubit mapper.
                Defaults to ParityMapper().
        """
        self.basis = basis

        # Default active space config (effectively full space, or handle explicitly)
        default_active_space_config = {'method': 'full'} # Indicate no active space selection by default
        self.active_space_config = default_active_space_config
        if active_space_config:
            self.active_space_config.update(active_space_config)

        # Default mapper
        if mapper is None:
            from qiskit_nature.second_q.mappers import ParityMapper # Import locally if default
            self.mapper = ParityMapper()
            print("Info: Using default ParityMapper.")
        else:
            self.mapper = mapper

        # Instantiate helpers here if they are classes without state,
        # otherwise instantiate within build_model if they need specific configs per run.
        # Assuming functional usage or local instantiation is fine for now.

    def _get_atoms_from_pdb(self, pdb_path: str) -> Optional[List[Tuple[str, Tuple[float, float, float]]]]:
        """
        Parses a PDB file to extract atom symbols and coordinates.
        Uses a simple parser for ATOM/HETATM lines.
        Replace with PDBSystemBuilder or Biopython if more robust parsing is needed.
        """
        atoms = []
        try:
            with open(pdb_path, 'r') as f:
                for line in f:
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        try:
                            # PDB format is fixed-width
                            symbol = line[76:78].strip().upper() # Element symbol
                            if not symbol: # Fallback to atom name if symbol missing (common for H)
                                 symbol = line[12:16].strip()[0].upper() # First char of atom name

                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            atoms.append((symbol, (x, y, z)))
                        except (ValueError, IndexError) as parse_err:
                            print(f"Warning: Could not parse line in {pdb_path}: {line.strip()} - Error: {parse_err}")
            if not atoms:
                 print(f"Error: No ATOM/HETATM records found or parsed in {pdb_path}.")
                 return None
            return atoms
        except FileNotFoundError:
            print(f"Error: PDB file not found at {pdb_path}")
            return None
        except Exception as e:
            print(f"Error reading PDB file {pdb_path}: {e}")
            return None


    def build_model(self, pdb_path: str, input_charge: int, input_spin_2S: int) -> Tuple[Optional[SparsePauliOp], Optional[QuantumCircuit]]:
        """
        Builds the quantum model (qubit Hamiltonian and ansatz) for the system in the PDB file.

        Args:
            pdb_path (str): Path to the PDB file.
            input_charge (int): The desired total charge of the system.
            input_spin_2S (int): The desired total spin (2*S) of the system (0=singlet, 1=doublet, ...).

        Returns:
            tuple: (qubit_op, ansatz) where:
                   qubit_op (SparsePauliOp | None): The qubit Hamiltonian operator.
                   ansatz (QuantumCircuit | None): The UCCSD ansatz circuit.
                   Returns (None, None) if any step fails.
        """
        print(f"\n--- Building Quantum Model for: {pdb_path} ---")
        print(f"Initial input: charge={input_charge}, spin(2S)={input_spin_2S}, basis={self.basis}")

        corrected_charge = input_charge
        corrected_spin_2S = input_spin_2S
        mol = None # Initialize molecule object variable

        try:
            # --- 1. Read Atoms & Perform Spin Correction ---
            atoms = self._get_atoms_from_pdb(pdb_path)
            if atoms is None: return None, None # Error handled in helper

            atomic_numbers = qn_settings.dict_atomic_numbers
            num_neutral_electrons = sum(atomic_numbers[symbol.upper()] for symbol, coords in atoms)
            num_total_electrons = num_neutral_electrons - input_charge
            print(f"Info: System atoms imply {num_neutral_electrons} neutral electrons.")
            print(f"Info: With input charge {input_charge}, target total electrons = {num_total_electrons}")

            is_consistent = (num_total_electrons % 2) == (input_spin_2S % 2)
            if not is_consistent:
                corrected_spin_2S = num_total_electrons % 2
                warnings.warn(
                    f"Spin Correction: Input charge={input_charge} implies {num_total_electrons} electrons. "
                    f"Input spin(2S)={input_spin_2S} has INCONSISTENT parity. "
                    f"Automatically correcting spin(2S) to {corrected_spin_2S} for calculation.",
                    UserWarning
                )
            else:
                print(f"Info: Input charge/spin are consistent.")
            print(f"Info: Using charge={corrected_charge}, spin(2S)={corrected_spin_2S} for modeling.")

            # --- 2. Build Molecule Object (using PDBSystemBuilder or directly) ---
            # Using PDBSystemBuilder (if available)
            print("Building molecule object...")
            builder = PDBSystemBuilder(pdb_path, charge=corrected_charge, spin=corrected_spin_2S, basis=self.basis)
            mol = builder.build_mole() # Assumes build_mole returns a PySCF compatible Mole object or similar
            if mol is None:
                raise ValueError("PDBSystemBuilder failed to build the molecule object.")

            # --- 3. Run SCF & Active Space Selection ---
            print("Running SCF...")
            # Instantiate selector here if it's a class needing config per run
            selector = ActiveSpaceSelector(threshold=self.active_space_config.get('threshold', 0.6))
            mf = selector.run_scf(mol)
            if mf is None:
                raise RuntimeError("SCF calculation failed.")

            print("Selecting active space...")
            active_space_method = self.active_space_config.get('method', 'full')
            if active_space_method == 'energy':
                n_before = self.active_space_config.get('n_before_homo', 1)
                n_after = self.active_space_config.get('n_after_lumo', 1)
                active_e, active_o, _, _ = selector.select_active_space_with_energy(mf, n_before_homo=n_before, n_after_lumo=n_after)
                print(f"Selected Active space => electrons={active_e}, spatial_orbitals={active_o}")
                if active_e is None or active_o is None or active_e < 0 or active_o <= 0:
                     raise ValueError("Invalid active space determined by energy method.")
                transformer: Optional[BaseTransformer] = ActiveSpaceTransformer(num_electrons=active_e, num_spatial_orbitals=active_o)
            elif active_space_method == 'manual':
                active_e = self.active_space_config['num_electrons']
                active_o = self.active_space_config['num_spatial_orbitals']
                print(f"Manual Active space => electrons={active_e}, spatial_orbitals={active_o}")
                if active_e < 0 or active_o <= 0:
                     raise ValueError("Invalid manual active space dimensions.")
                transformer = ActiveSpaceTransformer(num_electrons=active_e, num_spatial_orbitals=active_o)
            elif active_space_method == 'full':
                 print("Info: Using full active space (no transformation).")
                 transformer = None # No transformation needed
            else:
                 raise ValueError(f"Unknown active space selection method: {active_space_method}")

            # --- 4. Create Electronic Structure Problem ---
            print("Creating Qiskit Nature problem...")
            # Reconstruct atom string list from parsed atoms for driver
            atom_str_list = [f"{sym} {x} {y} {z}" for sym, (x, y, z) in atoms]
            driver = PySCFDriver(
                atom=atom_str_list, basis=self.basis, charge=corrected_charge, spin=corrected_spin_2S,
                unit=DistanceUnit.ANGSTROM
            )
            es_problem: ElectronicStructureProblem = driver.run()

            # --- 5. Apply Active Space Transformation (if applicable) ---
            if transformer:
                print("Applying active space transformation...")
                try:
                     red_problem = transformer.transform(es_problem)
                     if red_problem.num_particles is None: # Check if transform was successful
                          raise ValueError("Active space transformation resulted in an invalid problem.")
                     problem_to_solve = red_problem
                     print(f"Transformed problem: Particles={problem_to_solve.num_particles}, "
                           f"Spatial Orbitals={problem_to_solve.num_spatial_orbitals}")
                except Exception as transform_err:
                     print(f"Error during active space transformation: {transform_err}")
                     # Fallback to full problem? Or raise error? Let's raise error.
                     raise RuntimeError("Active space transformation failed.") from transform_err
            else:
                problem_to_solve = es_problem # Solve the full problem
                print(f"Full problem: Particles={problem_to_solve.num_particles}, "
                      f"Spatial Orbitals={problem_to_solve.num_spatial_orbitals}")


            # --- 6. Map to Qubits ---
            print("Mapping Hamiltonian to qubits...")
            qubit_converter = QubitConverter(self.mapper)
            second_q_op = problem_to_solve.hamiltonian.second_q_op()
            # Ensure second_q_op is not None before converting
            if second_q_op is None:
                raise ValueError("Failed to obtain second quantized Hamiltonian operator.")
            qubit_op = qubit_converter.convert(second_q_op, num_particles=problem_to_solve.num_particles)
            print(f"Qubit Hamiltonian created with {qubit_op.num_qubits} qubits.")

            # --- 7. Create Ansatz ---
            print("Creating UCCSD Ansatz...")
            n_so = problem_to_solve.num_spatial_orbitals
            alpha = problem_to_solve.num_alpha
            beta = problem_to_solve.num_beta
            if n_so is None or alpha is None or beta is None:
                 raise ValueError("Cannot determine problem dimensions for Ansatz creation.")

            hf_init = HartreeFock(n_so, (alpha, beta), self.mapper)
            ansatz = UCCSD(n_so, (alpha, beta), self.mapper, initial_state=hf_init)
            print("UCCSD Ansatz created.")

            print(f"--- Model building successful for {pdb_path} ---")
            return qubit_op, ansatz

        except Exception as e:
            print(f"\n!!! Error building quantum model for {pdb_path}: {e} !!!")
            import traceback
            traceback.print_exc()
            return None, None # Return None tuple on any failure