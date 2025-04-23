# --*-- conding:utf-8 --*--
# @time:4/20/25 22:11
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:bench.py

# -*- coding: utf-8 -*-
# --- Main Script ---

import os
import json
import numpy as np
import time
import warnings
from typing import Dict, Any, Optional, Tuple
from qiskit_nature.second_q.mappers import ParityMapper
from utils import ConfigManager

# --- Qiskit & Dependencies ---
# (Assume necessary Qiskit imports are handled within the classes below)
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options
from qiskit.transpiler import Target
try:
    # Alias for Estimator based on availability
    from qiskit_ibm_runtime import EstimatorV2 as Estimator
except ImportError:
    from qiskit.primitives import Estimator # type: ignore

# --- Import the Custom Classes We Defined ---
# Make sure the files containing these classes are accessible (e.g., in the same directory or PYTHONPATH)
try:
    # Assuming the VQE solver class is named QuantumVQE_Solver as defined previously
    from vqe import VQE_Solver # Replace 'quantum_vqe_solver' with the actual filename if different
except ImportError:
    print("Error: Could not import QuantumVQE_Solver class. Make sure the file exists and is accessible.")
    # Define dummy class so script can be parsed, but it will fail at runtime
    class VQE_Solver:
        def __init__(self, optimizer_options=None, transpilation_options=None, callback=None): pass
        def run(self, hamiltonian, ansatz, estimator, target, initial_point=None): return {'error': 'Dummy Solver'}

try:
    # Assuming the modeling class is named SystemModeler as defined previously
    from utils import SystemModeler
except ImportError:
    print("Error: Could not import SystemModeler class. Make sure the file exists and is accessible.")
    # Define dummy class
    class SystemModeler:
        def __init__(self, basis="sto3g", active_space_config=None, mapper=None): pass
        def create_model(self, pdb_path, input_charge, input_spin_2S): return (None, None)

# --- Helper Function: Calculate Binding Energy for One System ---
def calculate_binding_energy_for_system(
    system_id: str,
    system_config: Dict[str, Any],
    modeler_config: Dict[str, Any],
    solver_config: Dict[str, Any],
    service: QiskitRuntimeService,
    session_options: Dict[str, Any],
    estimator_options: Dict[str, Any]
    ) -> Dict[str, Any]:
    """
    Orchestrates the calculation of binding energy for one system using
    SystemModeler and QuantumVQE_Solver within a Qiskit Runtime Session.

    Args:
        system_id (str): Unique identifier for the system.
        system_config (dict): Configuration for the system, including:
            'base_path': Directory containing PDB files.
            'file_conventions': Dict mapping component name to PDB filename pattern.
            'params': Dict mapping component name to {'charge': int, 'spin': int}.
        modeler_config (dict): Configuration for the SystemModeler.
        solver_config (dict): Configuration for the QuantumVQE_Solver.
        service (QiskitRuntimeService): Initialized Qiskit Runtime service.
        session_options (dict): Options for the Session (e.g., backend).
        estimator_options (dict): Options for the Estimator (e.g., shots, resilience).

    Returns:
        dict: Dictionary containing results for 'complex', 'protein', 'ligand', and 'binding'.
              Values will be energies (float) or None/Error string on failure.
    """
    results = {'complex': None, 'protein': None, 'ligand': None, 'binding': None}
    component_energies = {} # Store successful energy values

    print(f"\n===== Processing System: {system_id} =====")

    # --- Instantiate Modeler and Solver ONCE per system ---
    try:
        modeler = SystemModeler(**modeler_config) # Pass config dict directly
        solver = VQE_Solver(**solver_config) # Pass config dict directly
    except Exception as init_err:
        print(f"Error initializing Modeler or Solver: {init_err}")
        results = {k: 'Init Error' for k in results}
        return results

    # --- Setup Estimator Options ---
    est_options = Options()
    est_options.execution.shots = estimator_options.get('shots', 1024)
    est_options.optimization_level = estimator_options.get('transpilation_opt_level', 1) # Runtime transpilation
    est_options.resilience_level = estimator_options.get('resilience_level', 0)
    # Add other options if needed

    active_session = None
    try:
        # --- Start Session ---
        with Session(service=service, backend=session_options.get("backend", None)) as session:
            active_session = session
            print(f"Session opened (ID: {session.session_id}) for {system_id}")
            backend_name = session.backend()
            if not backend_name:
                raise ValueError("Session could not determine a backend.")
            print(f"Session using backend: {backend_name}")

            # --- Create Estimator ---
            estimator = Estimator(session=session, options=est_options)

            # --- Get Target ---
            print(f"Fetching target information for backend: {backend_name}...")
            backend_obj = service.get_backend(backend_name)
            target = backend_obj.target
            if not target:
                 raise ValueError(f"Could not obtain target information for backend {backend_name}.")
            print(f"Target obtained.")

            # --- Loop through Components (Complex, Protein, Ligand) ---
            for component in ['complex', 'protein', 'ligand']:
                print(f"\n--- Processing Component: {component} ---")
                component_start_time = time.time()

                # Get specific config for this component
                pdb_file = system_config['file_conventions'][component].format(id=system_id)
                pdb_path = os.path.join(system_config['base_path'], pdb_file)
                params = system_config['params'][component]
                input_charge = params['charge']
                input_spin_2S = params['spin'] # Initial spin (2S)

                if not os.path.exists(pdb_path):
                     print(f"Error: PDB file not found: {pdb_path}")
                     raise IOError(f"PDB not found for {component}: {pdb_path}") # Abort system

                # 1. Create Quantum Model (calls modeler.create_model)
                #    This step includes reading PDB, spin correction, SCF, Active Space, Mapping, Ansatz creation
                qubit_op, ansatz = modeler.create_model(
                    pdb_path=pdb_path,
                    input_charge=input_charge,
                    input_spin_2S=input_spin_2S
                )

                if qubit_op is None or ansatz is None:
                    print(f"Error: Failed to create quantum model for {component}.")
                    raise ValueError(f"Model creation failed for {component}")

                # 2. Run VQE (calls solver.run)
                vqe_result = solver.run(
                    hamiltonian=qubit_op,
                    ansatz=ansatz,
                    estimator=estimator, # Pass the SAME estimator
                    target=target,       # Pass the SAME target
                    initial_point=None   # Use random start
                )

                # 3. Store Result
                if 'error' in vqe_result or vqe_result.get('optimal_value') is None:
                    error_msg = vqe_result.get('error', 'Unknown VQE Error')
                    print(f"Error during VQE run for {component}: {error_msg}")
                    raise RuntimeError(f"VQE run failed for {component}")
                else:
                    component_energies[component] = vqe_result['optimal_value']
                    print(f"Component {component} VQE successful. Final Energy: {component_energies[component]:.6f}")
                    print(f"Component {component} wall time: {vqe_result.get('wall_time', 'N/A'):.2f} s")
                    # Optionally save detailed vqe_result dict for this component
                    # component_result_file = os.path.join(output_dir, f"{system_id}_{component}_vqe_details.json")
                    # with open(component_result_file, "w") as f_comp:
                    #     json.dump(vqe_result, f_comp, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))


            print(f"\nSession closing (ID: {session.session_id}) for {system_id}")
            # Session closes automatically here

    except Exception as e:
        print(f"\n!!!! Critical Error during calculation for system {system_id}: {e} !!!!")
        if active_session:
            print(f"Attempting to close session {active_session.session_id} due to error.")
            try: active_session.close()
            except Exception as close_err: print(f"Error closing session: {close_err}")
        results = {k: v if v is not None else 'Error' for k, v in results.items()} # Mark failed components
        results['binding'] = 'Error'
        return results

    # --- Calculate Binding Energy ---
    if all(comp in component_energies for comp in ['complex', 'protein', 'ligand']):
        E_complex = component_energies['complex']
        E_protein = component_energies['protein']
        E_ligand = component_energies['ligand']
        binding_energy = E_complex - (E_protein + E_ligand)
        results.update({'complex': E_complex, 'protein': E_protein, 'ligand': E_ligand, 'binding': binding_energy})
        print(f"\n--- Binding Energy Calculation Summary for {system_id} ---")
        print(f"  E_complex: {E_complex:.6f}")
        print(f"  E_protein: {E_protein:.6f}")
        print(f"  E_ligand:  {E_ligand:.6f}")
        print(f"  E_binding: {binding_energy:.6f}")
        print("==========================================================")
    else:
        print(f"\n--- Binding Energy Calculation FAILED for {system_id} (missing component energy) ---")
        # Update results dict with successfully calculated energies
        results.update(component_energies)
        results['binding'] = None # Indicate binding energy could not be calculated

    return results


# --- Main Execution Block ---
if __name__ == "__main__":

    print("--- Quantum Binding Energy Calculation Script ---")
    start_run_time = time.time()

    # 1. IBM Quantum Credentials & Service Initialization
    try:
        print("Initializing Qiskit Runtime Service...")

        cfg = ConfigManager("config.txt")

        token = token=cfg.get("TOKEN")
        instance = cfg.get("INSTANCE")

        if not token or not instance:
            raise ValueError("Set environment variables QISKIT_IBM_TOKEN and QISKIT_IBM_INSTANCE")
        service = QiskitRuntimeService(channel='ibm_quantum', instance=instance, token=token)
        print(f"Service initialized for instance: {instance}")
    except Exception as e:
        print(f"Error initializing QiskitRuntimeService: {e}")
        exit(1)

    # 2. Define Systems for Batch Processing
    # !! IMPORTANT: Make sure the specified PDB files actually exist !!

    systems_to_process = {
        "1c5z_example": { # Using a descriptive ID
            "base_path": "./data_set/1c5z/", # Directory for this system's PDBs
            "file_conventions": {
                'complex': '{id}_Binding_mode.pdb', # e.g., 1c5z_example_Binding_mode.pdb
                'protein': '{id}_protein_part.pdb', # e.g., 1c5z_example_protein_only.pdb
                'ligand': '{id}_ligand_part.pdb'   # e.g., 1c5z_example_ligand_only.pdb
            },
            "params": { # Initial charge and spin(2S) - will be checked/corrected by SystemModeler
                'complex': {'charge': 1, 'spin': 0}, # Example values - VERIFY THESE!
                'protein': {'charge': 1, 'spin': 0}, # Example values - VERIFY THESE!
                'ligand':  {'charge': 0, 'spin': 0}  # Example values - VERIFY THESE!
            }
        },
    }

    # 3. Define Global Calculation Settings
    # Passed to SystemModeler constructor
    modeler_config = {
        'basis': "sto3g",
        'active_space_config': {
            'method': 'energy', # 'energy', 'manual', or 'full'
            'threshold': 0.6,   # Used if method='energy'
            'n_before_homo': 1, # Used if method='energy'
            'n_after_lumo': 1,  # Used if method='energy'
            # 'num_electrons': X, # Used if method='manual'
            # 'num_spatial_orbitals': Y # Used if method='manual'
        },
        'mapper': ParityMapper() # Instantiate mapper here
    }
    # Passed to QuantumVQE_Solver constructor
    solver_config = {
        'optimizer_options': {
            'name': 'SPSA',
            'maxiter': 100,
            # Add other SPSA params like c0, c1 if needed
        },
        'transpilation_options': {
            'optimization_level': 1 # Level for VQE circuit prep
        }
        # 'callback': my_global_callback # Optional global callback
    }
    # Passed to Estimator constructor
    estimator_options = {
        'shots': 2048,
        'resilience_level': 0, # 0: No mitigation, 1: T-REx (Readout error), 2: ZNE, 3: T-REx+ZNE
        'transpilation_opt_level': 1 # Level for runtime transpilation
    }
    # Passed to Session constructor
    session_options = {
        "backend": None # None: Let runtime choose; Or e.g., "ibm_simulator" or "ibm_brisbane"
    }

    # 4. Setup Output Directory
    output_dir = f"binding_energy_results_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # 5. Run Batch Processing
    all_system_results = {}
    print("\n--- Starting Batch Processing ---")

    for system_id, config in systems_to_process.items():
        print(f"\n>>> Processing System: {system_id} <<<")
        system_start_time = time.time()

        # Prepare system_config dict for the function
        system_config = {
            "base_path": config['base_path'],
            "file_conventions": config['file_conventions'],
            "params": config['params']
        }

        # Call the main calculation function for this system
        system_results = calculate_binding_energy_for_system(
            system_id=system_id,
            system_config=system_config,
            modeler_config=modeler_config,
            solver_config=solver_config,
            service=service,
            session_options=session_options,
            estimator_options=estimator_options
        )

        system_wall_time = time.time() - system_start_time
        print(f">>> System {system_id} finished in {system_wall_time:.2f} seconds <<<")

        all_system_results[system_id] = system_results

        # Save individual results immediately
        output_file = os.path.join(output_dir, f"{system_id}_binding_energy.json")
        try:
            # Convert numpy arrays (if any remain) to lists for JSON compatibility
            def json_converter(o):
                if isinstance(o, np.ndarray): return o.tolist()
                if isinstance(o, np.generic): return o.item() # Handle numpy scalars
                # Add more type conversions if needed (e.g., complex numbers)
                return str(o) # Fallback to string for unknown types

            with open(output_file, "w") as f:
                json.dump(system_results, f, indent=4, default=json_converter)
            print(f"Results for {system_id} saved to {output_file}")
        except Exception as e:
            print(f"Error saving results for {system_id} to JSON: {e}")

    # 6. Save Summary of All Results
    summary_file = os.path.join(output_dir, "summary_binding_energies.json")
    print(f"\nSaving summary of all results to {summary_file}...")
    try:
        with open(summary_file, "w") as f:
             json.dump(all_system_results, f, indent=4, default=json_converter) # Use converter here too
        print("Summary saved.")
    except Exception as e:
        print(f"Error saving summary JSON: {e}")

    total_run_time = time.time() - start_run_time
    print(f"\n--- Batch Processing Complete ---")
    print(f"Total wall time: {total_run_time:.2f} seconds.")


