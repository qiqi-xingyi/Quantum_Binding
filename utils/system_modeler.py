# --*-- conding:utf-8 --*--
# @time:4/23/25 15:40
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:system_modeler.py

import warnings
from typing import Dict, Any, Optional, Tuple, List

# Imports from the base code snippet and necessary types
from qiskit_nature.units import DistanceUnit # Keep this one as it's used as a value
from qiskit_nature.second_q.drivers import PySCFDriver
# Import specific Mapper class
from qiskit_nature.second_q.mappers import ParityMapper # Import concrete mapper needed
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer, BaseTransformer
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.settings import settings as qn_settings
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit

# Assume custom utils are available as defined in the base code context
try:
    from .pdb_system_builder import PDBSystemBuilder
    from .active_space_selector import ActiveSpaceSelector
except ImportError:
    print("Warning: Could not import custom utils (PDBSystemBuilder, ActiveSpaceSelector).")
    print("Please ensure these files exist in the 'utils' directory or integrate their logic.")
    # Define dummy classes if needed for the script to load
    class PDBSystemBuilder:
         def __init__(self, pdb_path, charge, spin, basis): pass
         def build_mole(self): print("Dummy build_mole called"); return None
    class ActiveSpaceSelector:
         def __init__(self, threshold): pass
         def run_scf(self, mol): print("Dummy run_scf called"); return None
         def select_active_space_with_energy(self, mf, n_before_homo, n_after_lumo): print("Dummy select_active_space called"); return (None, None, None, None)

# --- Class for Modeling Quantum System from PDB ---
class SystemModeler:
    """
    Reads a PDB file, performs quantum chemistry modeling based on the
    original code's logic (PySCFDriver, ActiveSpaceTransformer, etc.),
    includes automatic spin correction, and generates VQE inputs.
    Uses the modern direct mapping approach (mapper.map()) and removes specific type hints.
    """

    def __init__(self,
                 basis: str = "sto3g", # Use basic type hint: str
                 active_space_config: Optional[Dict[str, Any]] = None,
                 mapper: Optional[Any] = None): # Use Any or remove hint if causing issues
        """
        Initializes the System Modeler.

        Args:
            basis (str): The basis set to use (e.g., "sto3g").
            active_space_config (dict, optional): Configuration for active space selection.
                Defaults to a simple full space if None.
            mapper (object, optional): The fermion-to-qubit mapper instance
                (e.g., ParityMapper(), JordanWignerMapper()). Defaults to ParityMapper().
        """
        self.basis: str = basis # Store with basic type hint

        default_active_space_config = {'method': 'full'}
        self.active_space_config = default_active_space_config
        if active_space_config:
            self.active_space_config.update(active_space_config)

        if mapper is None:
            self.mapper = ParityMapper() # Default to ParityMapper instance
            print("Info: SystemModeler using default ParityMapper.")
        else:
             # You might want to add a check here if needed, e.g., hasattr(mapper, 'map')
             self.mapper = mapper # Store the provided mapper instance

        # Assuming ActiveSpaceSelector can be instantiated here or used functionally
        self.selector = ActiveSpaceSelector(threshold=self.active_space_config.get('threshold', 0.6))

    def _get_atoms_from_pdb(self, pdb_path: str) -> Optional[List[Tuple[str, Tuple[float, float, float]]]]:
        """
        Parses a PDB file to extract atom symbols and coordinates (simple parser).
        """
        # (Same simple parser as before)
        atoms = []
        try:
            with open(pdb_path, 'r') as f:
                for line in f:
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        try:
                            symbol = line[76:78].strip().upper()
                            if not symbol: symbol = line[12:16].strip()[0].upper()
                            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                            atoms.append((symbol, (x, y, z)))
                        except (ValueError, IndexError) as parse_err:
                            print(f"Warning: Skipping line due to parse error in {pdb_path}: {line.strip()} - Error: {parse_err}")
            if not atoms: print(f"Error: No ATOM/HETATM records found/parsed in {pdb_path}."); return None
            return atoms
        except FileNotFoundError: print(f"Error: PDB file not found at {pdb_path}"); return None
        except Exception as e: print(f"Error reading PDB file {pdb_path}: {e}"); return None

    def create_model(self, pdb_path: str, input_charge: int, input_spin_2S: int) -> Tuple[Optional[SparsePauliOp], Optional[QuantumCircuit]]:
        """
        Creates the qubit Hamiltonian and UCCSD ansatz based on the base code's logic,
        using direct mapper.map() and including spin correction.

        Args:
            pdb_path (str): Path to the PDB file.
            input_charge (int): Desired total charge.
            input_spin_2S (int): Desired total spin (2*S).

        Returns:
            tuple: (qubit_op, ansatz) or (None, None) on failure.
        """
        print(f"\n--- Creating Quantum Model for: {pdb_path} ---")
        print(f"Basis: {self.basis}, Initial input: charge={input_charge}, spin(2S)={input_spin_2S}")

        mol = None # Ensure mol is defined

        try:
            # --- Step 1: Read Atoms & Perform Spin Correction ---
            atoms = self._get_atoms_from_pdb(pdb_path)
            if atoms is None: return None, None

            atomic_numbers = qn_settings.dict_atomic_numbers
            num_neutral_electrons = sum(atomic_numbers[symbol.upper()] for symbol, coords in atoms)
            num_total_electrons = num_neutral_electrons - input_charge
            print(f"Info: Calculated total electrons = {num_total_electrons} (based on {len(atoms)} atoms and charge {input_charge})")

            corrected_charge = input_charge
            corrected_spin_2S = input_spin_2S
            is_consistent = (num_total_electrons % 2) == (input_spin_2S % 2)
            if not is_consistent:
                corrected_spin_2S = num_total_electrons % 2
                warnings.warn(
                    f"Spin Correction: Electron count ({num_total_electrons}) and input spin(2S) ({input_spin_2S}) have different parity. "
                    f"Correcting spin(2S) to {corrected_spin_2S}.", UserWarning)
            else:
                print("Info: Input charge/spin parity consistent with electron count.")
            print(f"Info: Using charge={corrected_charge}, spin(2S)={corrected_spin_2S} for Mol/Driver.")

            # --- Step 2: Create PySCF Mole ---
            print("Building molecule object via PDBSystemBuilder...")
            builder = PDBSystemBuilder(pdb_path, charge=corrected_charge, spin=corrected_spin_2S, basis=self.basis)
            mol = builder.build_mole()
            if mol is None: raise ValueError("PDBSystemBuilder failed to build the molecule object.")

            # --- Step 3: Run SCF + Active Space Selection ---
            print("Running SCF via ActiveSpaceSelector...")
            mf = self.selector.run_scf(mol)
            if mf is None: raise RuntimeError("SCF calculation failed.")

            print("Selecting active space via ActiveSpaceSelector...")
            active_space_method = self.active_space_config.get('method', 'full')
            transformer: Optional[BaseTransformer] = None
            active_e: Optional[int] = None
            active_o: Optional[int] = None

            if active_space_method == 'energy':
                active_e, active_o, _, _ = self.selector.select_active_space_with_energy(
                    mf, n_before_homo=self.n_before_homo, n_after_lumo=self.n_after_lumo)
                print(f"Selected Active space => electrons={active_e}, spatial_orbitals={active_o}")
                if active_e is None or active_o is None or active_e < 0 or active_o <= 0:
                     raise ValueError("Invalid active space determined by energy method.")
                transformer = ActiveSpaceTransformer(num_electrons=active_e, num_spatial_orbitals=active_o)
            elif active_space_method == 'manual':
                # Ensure keys exist before accessing if method is manual
                if 'num_electrons' not in self.active_space_config or 'num_spatial_orbitals' not in self.active_space_config:
                     raise ValueError("Manual active space requires 'num_electrons' and 'num_spatial_orbitals' in config.")
                active_e = self.active_space_config['num_electrons']
                active_o = self.active_space_config['num_spatial_orbitals']
                print(f"Manual Active space => electrons={active_e}, spatial_orbitals={active_o}")
                if not isinstance(active_e, int) or not isinstance(active_o, int) or active_e < 0 or active_o <= 0:
                     raise ValueError("Invalid manual active space dimensions.")
                transformer = ActiveSpaceTransformer(num_electrons=active_e, num_spatial_orbitals=active_o)
            elif active_space_method == 'full':
                 print("Info: Using full active space (no transformation).")
            else:
                 raise ValueError(f"Unknown active space selection method: {active_space_method}")

            # --- Step 4: Create Electronic Structure Problem ---
            print("Creating Qiskit Nature problem via PySCFDriver...")
            atom_str_list = [f"{sym} {x} {y} {z}" for sym, (x, y, z) in mol.atom]
            # PySCFDriver expects simple strings for basis and unit
            driver = PySCFDriver(
                atom=atom_str_list, basis=str(mol.basis), charge=mol.charge, spin=mol.spin,
                unit=DistanceUnit.ANGSTROM # Use the enum value here
            )
            es_problem = driver.run()
            print(f"Electronic Structure Problem created. Num particles: {es_problem.num_particles}, Num orbitals: {es_problem.num_spatial_orbitals}")


            # --- Step 5: Apply Active Space Transformation (if applicable) ---
            problem_to_solve = es_problem
            if transformer:
                print("Applying active space transformation...")
                try:
                     red_problem = transformer.transform(es_problem)
                     if red_problem.num_particles is None: raise ValueError("Active space transformation resulted in an invalid problem.")
                     problem_to_solve = red_problem
                     print(f"Transformed problem: Particles={problem_to_solve.num_particles}, Spatial Orbitals={problem_to_solve.num_spatial_orbitals}")
                except Exception as transform_err:
                     raise RuntimeError("Active space transformation failed.") from transform_err
            else:
                print("Info: Proceeding with full problem space.")


            # --- Step 6: Map to Qubits (Corrected - No QubitConverter) ---
            print(f"Mapping Hamiltonian to qubits using {self.mapper.__class__.__name__}...")
            hamiltonian_op = problem_to_solve.hamiltonian.second_q_op()
            if hamiltonian_op is None: raise ValueError("Hamiltonian operator is None after transformation.")
            qubit_op: SparsePauliOp = self.mapper.map(hamiltonian_op) # Direct mapping
            print(f"Qubit Hamiltonian created with {qubit_op.num_qubits} qubits.")


            # --- Step 7: Create Ansatz ---
            print("Creating UCCSD Ansatz...")
            n_so = problem_to_solve.num_spatial_orbitals
            alpha = problem_to_solve.num_alpha
            beta = problem_to_solve.num_beta
            if n_so is None or alpha is None or beta is None:
                 raise ValueError("Cannot determine problem dimensions for Ansatz creation.")

            hf_init = HartreeFock(n_so, (alpha, beta), self.mapper)
            ansatz = UCCSD(
                num_spatial_orbitals=n_so,
                num_particles=(alpha, beta),
                qubit_mapper=self.mapper, # Pass the mapper instance
                initial_state=hf_init
            )
            print("UCCSD Ansatz created.")

            print(f"--- Model creation successful for {pdb_path} ---")
            return qubit_op, ansatz

        except Exception as e:
            print(f"\n!!! Error creating quantum model for {pdb_path}: {e} !!!")
            import traceback
            traceback.print_exc()
            return None, None