# --*-- conding:utf-8 --*--
# @time:4/20/25 22:11
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:bench.py

import os
import json
import numpy as np
import warnings

# Qiskit Imports
from qiskit.transpiler import Target
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.algorithms.optimizers import SPSA
# Qiskit Runtime Imports
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options
try:
    # qiskit-ibm-runtime >= 0.23.0
    from qiskit_ibm_runtime import EstimatorV2 as Estimator
    print("Using qiskit_ibm_runtime.EstimatorV2")
except ImportError:
    try:
        # qiskit-ibm-runtime < 0.23.0 (using qiskit.primitives)
        from qiskit.primitives import Estimator # type: ignore
        print("Using qiskit.primitives.Estimator")
    except ImportError:
        raise ImportError("Could not find a suitable Estimator class.")

# Qiskit Nature Imports
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper, QubitConverter
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.units import DistanceUnit
from qiskit_nature.settings import settings as qn_settings # For atomic numbers

# Custom Utils (Assume these exist in the 'utils' directory)
try:
    from utils.pdb_system_builder import PDBSystemBuilder
    from utils.active_space_selector import ActiveSpaceSelector
except ImportError:
    print("Warning: Could not import custom utils (PDBSystemBuilder, ActiveSpaceSelector).")
    print("Please ensure these files exist or integrate their logic.")
    # Define dummy classes if needed for the script to load, but it won't run
    class PDBSystemBuilder: pass
    class ActiveSpaceSelector: pass

# --- Helper Function: Prepare VQE Inputs (with Spin Correction) ---
def prepare_vqe_inputs(pdb_path, input_charge, input_spin_2S, basis, active_space_config, mapper):
    """
    Prepares qubit Hamiltonian and ansatz for VQE, including spin correction.

    Args:
        pdb_path (str): Path to PDB file.
        input_charge (int): User-provided charge.
        input_spin_2S (int): User-provided spin (2S).
        basis (str): Basis set (e.g., "sto3g").
        active_space_config (dict): Configuration for active space.
        mapper (SecondQuantizedOpMapper): Fermion-to-qubit mapper (e.g., ParityMapper).

    Returns:
        tuple: (SparsePauliOp or None, QuantumCircuit or None) - Qubit Hamiltonian and Ansatz.
               Returns (None, None) if preparation fails.
    """
    print(f"\n--- Preparing VQE inputs for: {pdb_path} ---")
    print(f"Initial input: charge={input_charge}, spin(2S)={input_spin_2S}, basis={basis}")

    corrected_charge = input_charge
    corrected_spin_2S = input_spin_2S

    try:
        # 1. Build initial molecule object to get atom list (can potentially fail)
        #    We might need PDBSystemBuilder to just give us atoms first, or read PDB manually here.
        #    Let's assume PDBSystemBuilder can give us the atom list OR build mol first.
        #    Simplification: Build mol first, then check. If build fails, we can't proceed anyway.
        temp_builder = PDBSystemBuilder(pdb_path, charge=input_charge, spin=input_spin_2S, basis=basis)
        mol_initial_try = temp_builder.build_mole() # Try building with initial params
        if mol_initial_try is None:
             print(f"Error: Failed to build initial molecule object from {pdb_path}. Cannot check/correct spin.")
             return None, None
        atoms = mol_initial_try.atom # Get atom list: [(symbol, (x, y, z)), ...]

        # 2. Calculate total electrons based on atoms and INPUT charge
        atomic_numbers = qn_settings.dict_atomic_numbers
        num_neutral_electrons = sum(atomic_numbers[symbol.upper()] for symbol, coords in atoms)
        num_total_electrons = num_neutral_electrons - input_charge
        print(f"Info: System atoms imply {num_neutral_electrons} neutral electrons.")
        print(f"Info: With input charge {input_charge}, target total electrons = {num_total_electrons}")

        # 3. Check & Correct Spin Parity
        is_consistent = (num_total_electrons % 2) == (input_spin_2S % 2)
        if not is_consistent:
            corrected_spin_2S = num_total_electrons % 2
            warnings.warn( # Use warnings module for better visibility
                f"Spin Correction: Input charge={input_charge} implies {num_total_electrons} electrons. "
                f"Input spin(2S)={input_spin_2S} has INCONSISTENT parity. "
                f"Automatically correcting spin(2S) to {corrected_spin_2S} for calculation.",
                UserWarning
            )
        else:
            print(f"Info: Input charge={input_charge} and spin(2S)={input_spin_2S} are consistent.")

        print(f"Info: Using charge={corrected_charge}, spin(2S)={corrected_spin_2S} for calculation.")

        # 4. Build definitive molecule object with corrected parameters (if needed)
        if not is_consistent: # Rebuild ONLY if correction occurred
             builder = PDBSystemBuilder(pdb_path, charge=corrected_charge, spin=corrected_spin_2S, basis=basis)
             mol = builder.build_mole()
        else:
             mol = mol_initial_try # Use the already built one

        if mol is None:
             print(f"Error: Failed to build molecule object from {pdb_path} even after potential spin correction.")
             return None, None

        # 5. Run SCF and Active Space Selection (using custom utils)
        #    These should use the 'mol' object built with corrected parameters.
        print("Running SCF...")
        selector = ActiveSpaceSelector(threshold=active_space_config.get('threshold', 0.6))
        mf = selector.run_scf(mol)
        if mf is None:
             print("Error: SCF calculation failed.")
             return None, None

        print("Selecting active space...")
        if active_space_config.get('method') == 'energy':
            n_before = active_space_config.get('n_before_homo', 1)
            n_after = active_space_config.get('n_after_lumo', 1)
            active_e, active_o, _, _ = selector.select_active_space_with_energy(
                mf, n_before_homo=n_before, n_after_lumo=n_after
            )
        elif active_space_config.get('method') == 'manual':
            active_e = active_space_config['num_electrons']
            active_o = active_space_config['num_spatial_orbitals']
        else:
            print("Error: Unknown active space selection method in config.")
            return None, None

        print(f"Selected Active space => electrons={active_e}, spatial_orbitals={active_o}")
        if active_e is None or active_o is None or active_e < 0 or active_o <= 0:
             print("Error: Invalid active space determined.")
             return None, None

        # 6. Create Qiskit Nature Problem & Transform
        print("Creating Qiskit Nature problem...")
        atom_str_list = [f"{sym} {x} {y} {z}" for sym, (x, y, z) in mol.atom]
        driver = PySCFDriver(
            atom=atom_str_list, basis=mol.basis, charge=mol.charge, spin=mol.spin, # Use final corrected values from mol
            unit=DistanceUnit.ANGSTROM
        )
        es_problem = driver.run()

        # Basic validation before transform
        num_electrons_total = es_problem.num_particles
        if sum(num_electrons_total) != num_total_electrons:
             print(f"Error: Electron count mismatch between check ({num_total_electrons}) and driver result ({sum(num_electrons_total)}). Check driver/SCF.")
             return None, None
        if active_e > num_total_electrons or active_o > es_problem.num_spatial_orbitals:
             print(f"Error: Requested active space ({active_e}e, {active_o}o) invalid for total system ({num_total_electrons}e, {es_problem.num_spatial_orbitals}o).")
             return None, None
        if active_e > 2 * active_o:
             print(f"Error: Cannot fit {active_e} electrons into {active_o} spatial orbitals.")
             return None, None


        print("Applying active space transformation...")
        transformer = ActiveSpaceTransformer(num_electrons=active_e, num_spatial_orbitals=active_o)
        red_problem = transformer.transform(es_problem)
        if red_problem.num_particles is None:
             print("Error: Active space transformation failed.")
             return None, None

        # 7. Map to Qubits
        print("Mapping Hamiltonian to qubits...")
        qubit_converter = QubitConverter(mapper)
        # Get the second quantized operator from the reduced problem
        second_q_op = red_problem.hamiltonian.second_q_op()
        # Convert to qubit operator
        qubit_op = qubit_converter.convert(second_q_op, num_particles=red_problem.num_particles)


        # 8. Create Ansatz
        print("Creating UCCSD Ansatz...")
        n_so = red_problem.num_spatial_orbitals
        alpha, beta = red_problem.num_alpha, red_problem.num_beta
        if n_so is None or alpha is None or beta is None:
             print("Error: Cannot determine reduced problem dimensions for Ansatz.")
             return None, None

        hf_init = HartreeFock(n_so, (alpha, beta), mapper)
        ansatz = UCCSD(n_so, (alpha, beta), mapper, initial_state=hf_init)

        print(f"Successfully prepared inputs: {qubit_op.num_qubits} qubits.")
        return qubit_op, ansatz

    except Exception as e:
        print(f"Error during VQE input preparation for {pdb_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# --- VQE Solver Class (using SPSA) ---
class QCVQESolver_SPSA:
    def __init__(self, maxiter=100, optimization_level=1):
        """
        VQE Solver using SPSA optimizer and an external Estimator.

        Args:
            maxiter (int): Maximum iterations for SPSA.
            optimization_level (int): Qiskit transpiler optimization level for circuits.
        """
        self.maxiter = maxiter
        self.optimization_level = optimization_level
        self.energy_list = [] # Stores energy history for one run_vqe call
        self._current_estimator = None
        self._ansatz_isa = None
        self._hamiltonian_isa = None
        # Store SPSA callback info if needed outside the run
        self.spsa_intermediate_info = {}

    def _cost_func(self, params):
        """Internal cost function using the stored estimator and circuits."""
        if self._current_estimator is None:
            raise RuntimeError("Estimator not set. Call run_vqe first.")

        pub = (self._ansatz_isa, [self._hamiltonian_isa], [params])
        try:
            job = self._current_estimator.run(pubs=[pub])
            result = job.result()
            if hasattr(result[0].data, 'evs'): energy = result[0].data.evs[0]
            elif hasattr(result, 'values'): energy = result.values[0]
            else: raise AttributeError("Could not extract energy.")
        except Exception as e:
            print(f"Error during estimator run (cost_func): {e}")
            energy = float('inf') # Return high value on error

        self.energy_list.append(energy)
        # Note: SPSA calls this twice per step, printing here might be verbose
        # Printing is handled better by the SPSA callback in run_vqe
        return energy

    def run_vqe(self, qubit_op, ansatz, estimator, target):
        """
        Runs the VQE optimization loop using SPSA.

        Args:
            qubit_op (SparsePauliOp): The qubit Hamiltonian.
            ansatz (QuantumCircuit): The parameterized ansatz circuit.
            estimator (Estimator): Pre-configured Estimator instance (in a Session).
            target (Target): Qiskit Target object for the backend.

        Returns:
            tuple: (list of energies during optimization, final parameters)
                   Returns ([], None) if optimization fails early.
        """
        if not isinstance(target, Target):
             raise TypeError("Target must be a qiskit.transpiler.Target object.")

        print(f"\nRunning VQE with SPSA (maxiter={self.maxiter})")
        self.energy_list = [] # Reset history for this run
        self.spsa_intermediate_info = {'nevals': 0, 'params': [], 'energy': [], 'step': []} # Reset SPSA info
        self._current_estimator = estimator

        # --- Transpilation ---
        try:
            print("Transpiling circuit...")
            pm = generate_preset_pass_manager(target=target, optimization_level=self.optimization_level)
            self._ansatz_isa = pm.run(ansatz)
            print("Applying layout to Hamiltonian...")
            if self._ansatz_isa.layout:
                 self._hamiltonian_isa = qubit_op.apply_layout(layout=self._ansatz_isa.layout)
            else:
                 print("Warning: Ansatz has no layout after transpilation. Using original Hamiltonian operator.")
                 self._hamiltonian_isa = qubit_op
            print(f"Transpiled Ansatz parameters: {self._ansatz_isa.num_parameters}")
            if self._ansatz_isa.num_parameters == 0:
                 print("Error: Ansatz has no parameters after transpilation. Cannot optimize.")
                 self._current_estimator = None # Cleanup
                 return [], None
        except Exception as e:
             print(f"Error during transpilation: {e}")
             self._current_estimator = None # Cleanup
             return [], None


        # --- SPSA Optimization ---
        optimizer = SPSA(maxiter=self.maxiter)

        # SPSA callback function for logging
        def callback_spsa(nfev, parameters, value, step, accepted):
            self.spsa_intermediate_info['nevals'] = nfev
            # Avoid storing huge parameter lists if not needed:
            # self.spsa_intermediate_info['params'].append(parameters)
            self.spsa_intermediate_info['energy'].append(value)
            self.spsa_intermediate_info['step'].append(step)
            # Get latest energy from internal list (more accurate representation of progress)
            latest_energy = self.energy_list[-1] if self.energy_list else value
            print(f"SPSA Iter: {len(self.spsa_intermediate_info['step']):>3} (NFEV: {nfev}), "
                  f"Energy: {latest_energy:.6f}, Step: {step:.4f}, Accepted: {accepted}")

        optimizer.callback = callback_spsa
        x0 = np.random.random(self._ansatz_isa.num_parameters) # Initial guess

        try:
            print("Starting SPSA optimizer...")
            result = optimizer.minimize(fun=self._cost_func, x0=x0)
            final_params = result.x
            print(f"SPSA optimization finished.")
            print(f"Final parameters found.")
        except Exception as e:
            print(f"Error during SPSA optimization: {e}")
            final_params = None # Indicate failure

        # Cleanup
        self._current_estimator = None
        self._ansatz_isa = None
        self._hamiltonian_isa = None

        return self.energy_list, final_params

# --- Main Binding Energy Workflow  ---
def calculate_binding_energy_session_spsa(
    system_id, base_path, file_conventions, system_params,
    active_space_config, vqe_options, # Contains maxiter, optimization_level for VQE
    service, session_options, estimator_options # Backend, Estimator settings (shots, resilience)
    ):
    """
    Orchestrates binding energy calculation within a Session using SPSA.
    """
    results = {'complex': None, 'protein': None, 'ligand': None, 'binding': None}
    basis = "sto3g" # Or make configurable
    mapper = ParityMapper() # Use ParityMapper consistently

    print(f"\n===== Calculating Binding Energy for System: {system_id} (Session + SPSA) =====")

    # Instantiate the modified solver ONCE per system
    qcvqe_solver = QCVQESolver_SPSA(
        maxiter=vqe_options.get('maxiter', 100),
        optimization_level=vqe_options.get('optimization_level', 1) # Transpilation level
    )

    # Create Estimator Options from input dict
    est_options = Options()
    est_options.execution.shots = estimator_options.get('shots', 1024)
    est_options.optimization_level = estimator_options.get('transpilation_opt_level', 1) # Runtime transpilation level
    est_options.resilience_level = estimator_options.get('resilience_level', 0) # Error mitigation level (0=off)
    # Add other options if needed e.g., est_options.environment.log_level = 'INFO'

    active_session = None # Keep track of session for potential early close
    try:
        with Session(service=service, backend=session_options.get("backend", None)) as session:
            active_session = session # Store session reference
            print(f"Session opened (ID: {session.session_id}) for {system_id}")
            backend_name = session.backend()
            print(f"Session using backend: {backend_name}")

            estimator = Estimator(session=session, options=est_options)

            # Get Target for transpilation
            if not backend_name:
                 raise ValueError("Session did not return a backend name. Cannot determine target.")
            backend_obj = service.get_backend(backend_name)
            target = backend_obj.target
            print(f"Target obtained for backend: {target.name if hasattr(target,'name') else 'Custom Target'}")

            # --- Calculate for each component ---
            for component in ['complex', 'protein', 'ligand']:
                print(f"\n--- Processing Component: {component} ---")
                pdb_file = file_conventions[component].format(id=system_id)
                pdb_path = os.path.join(base_path, pdb_file)
                params = system_params[component] # Get charge/spin for this component

                if not os.path.exists(pdb_path):
                     print(f"Error: PDB file not found: {pdb_path}")
                     results[component] = None
                     raise IOError(f"PDB not found: {pdb_path}") # Abort this system

                # 1. Prepare inputs (includes spin correction)
                qubit_op, ansatz = prepare_vqe_inputs(
                    pdb_path, params['charge'], params['spin'], basis,
                    active_space_config, mapper
                )

                if qubit_op is None or ansatz is None:
                    print(f"Error: Failed to prepare VQE inputs for {component}. Skipping component.")
                    results[component] = None
                    # Decide whether to abort whole system or just skip component
                    raise ValueError(f"Input prep failed for {component}") # Abort system

                # 2. Run VQE using the solver instance
                energies, final_params = qcvqe_solver.run_vqe(
                    qubit_op, ansatz, estimator, target # Pass same estimator/target
                )

                if energies and final_params is not None:
                    results[component] = energies[-1] # Store final energy
                    # Optionally save params/history for this component here
                else:
                    print(f"Error: VQE optimization failed for {component}.")
                    results[component] = None
                    raise RuntimeError(f"VQE failed for {component}") # Abort system

            print(f"\nSession closing (ID: {session.session_id}) for {system_id}")
            # Session closes automatically here

    except Exception as e:
        print(f"\n!!!! Critical Error during binding energy calculation for {system_id}: {e} !!!!")
        if active_session:
             print(f"Attempting to close session {active_session.session_id} due to error.")
             try:
                  active_session.close()
             except Exception as close_err:
                  print(f"Error closing session: {close_err}")
        # Ensure results dict indicates failure
        results = {k: v if v is not None else 'Error' for k, v in results.items()}
        results['binding'] = 'Error'
        return results # Return results dict indicating error state

    # --- Calculate Binding Energy ---
    if all(results[comp] is not None for comp in ['complex', 'protein', 'ligand']):
        results['binding'] = results['complex'] - (results['protein'] + results['ligand'])
        print(f"\n--- Binding Energy Calculation Summary for {system_id} ---")
        print(f"  E_complex: {results['complex']:.6f}")
        print(f"  E_protein: {results['protein']:.6f}")
        print(f"  E_ligand:  {results['ligand']:.6f}")
        print(f"  E_binding: {results['binding']:.6f}")
        print("==========================================================")
    else:
        print(f"\n--- Binding Energy Calculation FAILED for {system_id} (component error) ---")
        results['binding'] = None # Indicate calculation couldn't complete

    return results


# --- Main Execution Block ---
if __name__ == "__main__":

    print("--- Quantum Binding Energy Calculation Script ---")
    # 1. IBM Quantum Credentials & Service Initialization
    try:
        # Option 1: Load from saved account (if configured)
        # service = QiskitRuntimeService(channel='ibm_quantum')
        # print("Loaded QiskitRuntimeService from saved account.")

        # Option 2: Load from environment variables (More secure than hardcoding)
        token = os.environ.get('QISKIT_IBM_TOKEN')
        instance = os.environ.get('QISKIT_IBM_INSTANCE') # e.g., 'ibm-q/open/main'
        if not token or not instance:
            raise ValueError("Environment variables QISKIT_IBM_TOKEN and QISKIT_IBM_INSTANCE must be set.")
        service = QiskitRuntimeService(channel='ibm_quantum', instance=instance, token=token)
        print(f"Initialized QiskitRuntimeService for instance: {instance}")

    except Exception as e:
        print(f"Error initializing QiskitRuntimeService: {e}")
        print("Please ensure your IBM Quantum credentials are set correctly (e.g., environment variables).")
        exit(1)

    # 2. Define Systems for Batch Processing
    systems_to_process = {
        "1c5z": {
            "base_path": "./data_set/1c5z/", # Base directory for this system
            "file_conventions": { # Pattern for finding PDB files
                'complex': '{id}_Binding_mode.pdb',
                'protein': '{id}_protein_only.pdb', # MAKE SURE THIS FILE EXISTS!
                'ligand': '{id}_ligand_only.pdb'   # MAKE SURE THIS FILE EXISTS!
            },
            "params": { # Initial charge and spin(2S) - will be checked/corrected
                'complex': {'charge': 1, 'spin': 0},
                'protein': {'charge': 1, 'spin': 0}, # **VERIFY CHEMICAL CORRECTNESS**
                'ligand':  {'charge': 0, 'spin': 0}  # **VERIFY CHEMICAL CORRECTNESS**
            }
        },
        # Add more systems here following the same structure
        # "system2_id": { ... },
    }

    # 3. Define Global Calculation Settings
    active_space_config = {
        'method': 'energy',    # 'energy' or 'manual'
        'threshold': 0.6,
        'n_before_homo': 1,
        'n_after_lumo': 1,
        # 'num_electrons': X,  # Use if method is 'manual'
        # 'num_spatial_orbitals': Y # Use if method is 'manual'
    }
    vqe_options = { # Options for the QCVQESolver_SPSA class itself
        'maxiter': 100,         # Max iterations for SPSA
        'optimization_level': 1 # Transpilation level for VQE circuit preparation
    }
    estimator_options = { # Options passed to the Qiskit Runtime Estimator
        'shots': 2048,             # Number of shots per expectation value estimation
        'resilience_level': 1,     # Level of error mitigation (0=off, 1=basic, ...)
        'transpilation_opt_level': 1 # Transpilation level applied by runtime (can differ from VQE prep)
    }
    session_options = { # Options for the Runtime Session
        "backend": None # None = let runtime choose; or specify like "ibm_brisbane"
    }

    # 4. Setup Output Directory
    output_dir = "binding_energy_results_spsa_session"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in: {output_dir}")

    # 5. Run Batch Processing
    all_system_results = {}
    print("\n--- Starting Batch Processing ---")

    for system_id, config in systems_to_process.items():
        print(f"\n>>> Processing System: {system_id} <<<")
        system_results = calculate_binding_energy_session_spsa(
            system_id=system_id,
            base_path=config['base_path'],
            file_conventions=config['file_conventions'],
            system_params=config['params'],
            active_space_config=active_space_config,
            vqe_options=vqe_options,
            service=service,
            session_options=session_options,
            estimator_options=estimator_options
        )

        all_system_results[system_id] = system_results # Store results even if None/Error

        # Save individual results immediately
        if system_results:
            output_file = os.path.join(output_dir, f"{system_id}_binding_energy.json")
            try:
                with open(output_file, "w") as f:
                    # Convert numpy arrays (if any leak out) to lists for JSON
                    json.dump(system_results, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
                print(f"Results for {system_id} saved to {output_file}")
            except Exception as e:
                print(f"Error saving results for {system_id} to JSON: {e}")
        else:
            print(f"Calculation FAILED for system: {system_id}. No results file generated.")

    # 6. Save Summary of All Results
    summary_file = os.path.join(output_dir, "summary_binding_energies.json")
    print(f"\nSaving summary of all results to {summary_file}...")
    try:
        with open(summary_file, "w") as f:
             json.dump(all_system_results, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        print("Summary saved.")
    except Exception as e:
        print(f"Error saving summary JSON: {e}")

    print("\n--- Batch Processing Complete ---")