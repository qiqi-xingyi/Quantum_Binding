# -*- coding: utf-8 -*-
# @time:4/23/25 14:31
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:vqesolver.py


import numpy as np
import time
import warnings
from typing import Dict, Any, Optional, Tuple, List, Callable

# Qiskit Imports
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_algorithms.optimizers import SPSA, Optimizer, OptimizerResult, OptimizerSupportLevel

# Qiskit Runtime / Primitives Imports
try:
    # qiskit-ibm-runtime >= 0.23.0
    from qiskit_ibm_runtime import EstimatorV2 as Estimator
    print("Info: Using qiskit_ibm_runtime.EstimatorV2")
except ImportError:
    try:
        # qiskit-ibm-runtime < 0.23.0 (using qiskit.primitives)
        from qiskit.primitives import Estimator # type: ignore
        print("Info: Using qiskit.primitives.Estimator")
    except ImportError:
        raise ImportError("Could not find a suitable Estimator class.")


class VQE_Solver:
    """
    A class to execute the VQE algorithm, designed to work with an external Session/Estimator.
    """

    def __init__(self,
                 optimizer_options: Optional[Dict[str, Any]] = None,
                 transpilation_options: Optional[Dict[str, Any]] = None,
                 callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None):
        """
        Initializes the VQE Solver.

        Args:
            optimizer_options (dict, optional): Configuration options for the optimizer.
                Example: {'name': 'SPSA', 'maxiter': 100, 'c0': 0.1, ...}
                Defaults to SPSA with maxiter=100.
            transpilation_options (dict, optional): Options for circuit transpilation.
                Example: {'optimization_level': 1}
                Defaults to optimization_level=1.
            callback (callable, optional): Callback function during the optimization process.
                A common signature might be (iteration, current_params, current_energy),
                but the exact signature depends on how it's called (e.g., from the
                optimizer's callback support).
        """
        # --- Configuration storage ---
        # Set default optimizer options
        default_optimizer_options = {'name': 'SPSA', 'maxiter': 100}
        self.optimizer_options = default_optimizer_options
        if optimizer_options:
            self.optimizer_options.update(optimizer_options) # Merge user options

        # Set default transpilation options
        default_transpilation_options = {'optimization_level': 1}
        self.transpilation_options = default_transpilation_options
        if transpilation_options:
            self.transpilation_options.update(transpilation_options)

        self.callback = callback

        # --- Internal state (reset in run method) ---
        self._optimizer: Optional[Optimizer] = None
        self.energy_history: List[float] = []
        self.params_history: List[np.ndarray] = [] # Store parameter history (optional)
        self._eval_count: int = 0 # Store evaluation count

    def _build_optimizer(self) -> Optimizer:
        """Builds the optimizer instance based on options."""
        name = self.optimizer_options.get('name', 'SPSA').upper()
        maxiter = int(self.optimizer_options.get('maxiter', 100)) # Ensure integer

        if name == 'SPSA':
            # Can extract more SPSA-specific parameters from optimizer_options
            spsa_params = {k: v for k, v in self.optimizer_options.items() if k not in ['name', 'maxiter']}
            optimizer = SPSA(maxiter=maxiter, **spsa_params)
            # SPSA usually requires special callback handling as it calls fun internally multiple times
            # Callback logic will be handled within the run method
        # elif name == 'L_BFGS_B': # Example: if support for other optimizers is desired
        #     from qiskit.algorithms.optimizers import L_BFGS_B
        #     optimizer = L_BFGS_B(maxiter=maxiter)
        else:
            warnings.warn(f"Optimizer '{name}' not explicitly supported, defaulting to SPSA.", UserWarning)
            optimizer = SPSA(maxiter=maxiter)

        # Check if optimizer supports callbacks (most Qiskit optimizers do)
        if hasattr(optimizer, 'callback') and callable(self.callback):
             # Qiskit optimizer callbacks often accept specific arguments, which might not
             # perfectly match the user-provided callback signature.
             # An adapter might be needed, or the user should provide a callback
             # matching the optimizer's expectation.
             # Simplify: Assume for now the user callback works with SPSA or is not set via this mechanism.
             pass
        elif self.callback:
             print(f"Warning: Optimizer {name} might not support the provided callback format.")

        return optimizer

    def run(self,
            hamiltonian: SparsePauliOp,
            ansatz: QuantumCircuit,
            estimator: Estimator,
            target: Target,
            initial_point: Optional[np.ndarray] = None
           ) -> Dict[str, Any]:
        """
        Executes the VQE optimization procedure.

        Args:
            hamiltonian (SparsePauliOp): Qubit Hamiltonian (layout application might be needed or handled internally).
            ansatz (QuantumCircuit): Parameterized ansatz circuit (before transpilation).
            estimator (Estimator): Pre-configured Estimator instance associated with a Session.
            target (Target): Target object for the backend, used for circuit transpilation.
            initial_point (np.ndarray, optional): Initial parameter point for the optimization. If None, generated randomly.

        Returns:
            dict: A dictionary containing results, e.g.,
                  {'optimal_point': np.ndarray,  # Optimal parameters
                   'optimal_value': float,      # Optimal energy value
                   'energy_history': list[float], # Energy history
                   'params_history': list[np.ndarray], # Parameter history (if recorded)
                   'optimizer_result': OptimizerResult # Optimizer's raw result object
                  }
                  May return a dictionary with error information or raise an exception on failure.
        """
        print("\n--- Starting QuantumVQE_Solver run ---")
        start_time = time.time()

        # --- 0. Reset internal state ---
        self.energy_history = []
        self.params_history = []
        self._eval_count = 0

        # --- 1. Transpile circuit and Hamiltonian ---
        if not isinstance(target, Target):
            raise TypeError("Target must be a qiskit.transpiler.Target object.")
        print(f"Transpiling for target: {target.name if hasattr(target,'name') else 'Custom Target'} "
              f"with level {self.transpilation_options['optimization_level']}...")
        try:
            pm = generate_preset_pass_manager(
                target=target,
                optimization_level=self.transpilation_options['optimization_level']
            )
            ansatz_isa = pm.run(ansatz) # ISA = Instruction Set Architecture (transpiled)

            # Apply layout to Hamiltonian
            if ansatz_isa.layout:
                print("Applying layout to Hamiltonian...")
                hamiltonian_isa = hamiltonian.apply_layout(layout=ansatz_isa.layout)
            else:
                print("Warning: Ansatz has no layout after transpilation. Using original Hamiltonian operator.")
                hamiltonian_isa = hamiltonian # May require further checks

            num_params = ansatz_isa.num_parameters
            print(f"Transpilation complete. Ansatz parameters: {num_params}")
            if num_params == 0:
                print("Warning: Ansatz has no parameters after transpilation. Returning classical value if possible, or error.")
                # If the Hamiltonian is diagonal, the 0-parameter expectation value can be computed directly.
                # Otherwise, this is an error or special case.
                # Simplify: Treat as an error for now.
                raise ValueError("Ansatz has no parameters after transpilation, cannot optimize.")

        except Exception as e:
            print(f"Error during transpilation: {e}")
            raise RuntimeError("VQE failed during circuit transpilation.") from e

        # --- 2. Define energy evaluation function (Cost Function Evaluator) ---
        # This internal function is called by the optimizer whenever an energy value is needed.
        # It captures the current estimator, ansatz_isa, hamiltonian_isa.
        def cost_function_evaluator(params: np.ndarray) -> float:
            self._eval_count += 1
            pub = (ansatz_isa, [hamiltonian_isa], [params]) # PUB format
            try:
                job = estimator.run(pubs=[pub])
                result = job.result() # Get the result

                # Extract energy value (compatible with different Estimator versions)
                if hasattr(result[0].data, 'evs'):
                     energy = result[0].data.evs[0]
                elif hasattr(result, 'values'):
                     energy = result.values[0]
                else:
                     raise AttributeError("Could not extract energy from estimator result.")

                # Record history (ensure thread-safety if optimizer calls in parallel, but scipy/SPSA usually sequential)
                self.energy_history.append(energy)
                self.params_history.append(np.copy(params)) # Record a copy of the parameters

                # print(f"  Eval #{self._eval_count}, Energy: {energy:.6f}") # Print in callback instead

            except Exception as e:
                print(f"Error during estimator run (eval #{self._eval_count}): {e}")
                energy = float('inf') # Return a large value on error, signaling to the optimizer this is not a good direction
                # Could also record None or NaN, but optimizer needs to handle it
                self.energy_history.append(energy)
                self.params_history.append(np.copy(params))

            return energy

        # --- 3. Build optimizer ---
        self._optimizer = self._build_optimizer()
        print(f"Optimizer: {self.optimizer_options['name']} with maxiter={self._optimizer.settings.get('maxiter')}")

        # --- 4. Set initial point ---
        if initial_point is None:
            print("Generating random initial point.")
            initial_point = np.random.random(num_params)
        elif len(initial_point) != num_params:
            raise ValueError(f"Initial point dimension ({len(initial_point)}) does not match "
                             f"ansatz parameter count ({num_params}).")
        else:
            print("Using provided initial point.")

        # --- 5. Define SPSA callback function (if using SPSA) ---
        if isinstance(self._optimizer, SPSA):
            print("Setting up SPSA callback...")
            spsa_step_count = [0] # Use list to allow modification inside callback
            def spsa_callback_wrapper(nfev, parameters, value, step, accepted):
                # SPSA's 'value' is from one of its internal evaluations;
                # getting the latest from history is often more representative.
                latest_energy = self.energy_history[-1] if self.energy_history else value
                spsa_step_count[0] += 1
                print(f"SPSA Iter: {spsa_step_count[0]:>3} (NFEV: {nfev}), "
                      f"Energy: {latest_energy:.6f}, Step: {step:.4f}, Accepted: {accepted}")
                # Call the user-provided generic callback (if it exists)
                if self.callback:
                    try:
                        # Try passing generic info to the user callback
                        self.callback(spsa_step_count[0], parameters, latest_energy, step)
                    except Exception as cb_err:
                        print(f"Warning: Error in user callback: {cb_err}")

            self._optimizer.callback = spsa_callback_wrapper
        elif self.callback:
            # For non-SPSA optimizers, a different callback adapter might be needed
             print("Warning: Custom callback handling for non-SPSA optimizers is basic.")
             # Could try setting self.callback directly here, but arguments might mismatch
             # self._optimizer.callback = self.callback # Might fail

        # --- 6. Execute optimization ---
        print(f"Starting optimization at {time.strftime('%Y-%m-%d %H:%M:%S')}...")
        try:
            opt_result = self._optimizer.minimize(
                fun=cost_function_evaluator, # Energy evaluation function
                x0=initial_point           # Initial parameters
                # jac=... # If using a gradient optimizer, provide gradient function here
                # bounds=... # If optimizer supports bounds
            )
            print(f"Optimization finished successfully after {time.time() - start_time:.2f} seconds.")
            if hasattr(opt_result, 'message'): # Check if message attribute exists
                 print(f"Optimizer message: {opt_result.message}")

        except Exception as e:
            print(f"Error during optimization: {e}")
            import traceback
            traceback.print_exc()
            # Return partial results or error info
            return {
                'error': str(e),
                'optimal_point': None,
                'optimal_value': None,
                'energy_history': self.energy_history,
                'params_history': self.params_history,
                'optimizer_result': None
            }

        # --- 7. Prepare and return results ---
        results = {
            'optimal_point': opt_result.x,
            'optimal_value': opt_result.fun, # Final function value
            'energy_history': self.energy_history,
            'params_history': self.params_history,
            'optimizer_result': opt_result, # Keep the original result object
            'eval_count': self._eval_count,
            'wall_time': time.time() - start_time
        }
        print(f"Final Energy: {results['optimal_value']:.6f}")
        print("--- QuantumVQE_Solver run finished ---")
        return results