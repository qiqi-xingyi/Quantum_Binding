# --*-- conding:utf-8 --*--
# @time:4/28/25 18:03
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:vqev2.py

import os
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple

from scipy.optimize import minimize
from qiskit_ibm_runtime import Session, EstimatorV2


class MultiVQEPipeline:
    """
    Run (standard) VQE or Adapt-VQE for multiple problems in **one** IBM Runtime
    session, while recording detailed timing information.
    """

    def __init__(
        self,
        service,
        shots: int = 1024,
        maxiter: int = 100,
        optimization_level: int = 3,
        result_dir: str = "results_vqe",
    ):
        self.service = service
        self.shots = shots
        self.maxiter = maxiter
        self.optimization_level = optimization_level
        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)

    # ---------------------------------------------------------------------
    # public entry
    # ---------------------------------------------------------------------
    def run(self, problems: Dict[str, Tuple[Any, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Args
        ----
        problems
            mapping  {label: (qubit_op, ansatz_circuit | AdaptVQE_solver)}

        Returns
        -------
        dict[label] -> {"energies": [...], "ground_energy": float}
        """
        # -----------------------------------------------------------------
        # choose backend / open session / estimator
        # -----------------------------------------------------------------
        max_qubits = max(qop.num_qubits for qop, _ in problems.values())
        backend = self.service.least_busy(
            operational=True, simulator=False, min_num_qubits=max_qubits
        )

        session = Session(backend=backend)
        estimator = EstimatorV2(mode=session)
        estimator.options.default_shots = self.shots

        # -----------------------------------------------------------------
        results: Dict[str, Dict[str, Any]] = {}
        # -----------------------------------------------------------------
        for label, (qop, solver) in problems.items():
            timeline: List[Dict[str, Any]] = []

            # -------------------------------------------------------------
            # Adapt-VQE branch
            # -------------------------------------------------------------
            if hasattr(solver, "compute_minimum_eigenvalue"):
                adapt_res = solver.compute_minimum_eigenvalue(qop)
                e0 = adapt_res.eigenvalue

                results[label] = {"energies": [e0], "ground_energy": e0}
                with open(
                    os.path.join(self.result_dir, f"{label}_adapt_summary.json"), "w"
                ) as f:
                    json.dump(
                        {
                            "ground_energy": e0,
                            "num_iterations": adapt_res.num_iterations,
                            "final_max_gradient": adapt_res.final_maximum_gradient,
                        },
                        f,
                        indent=2,
                    )
                continue

            # -------------------------------------------------------------
            # Standard VQE branch (solver is a QuantumCircuit ansatz)
            # -------------------------------------------------------------
            circ = solver  # keep original circuit → qubit count matches qop
            n_params = circ.num_parameters
            energy_hist: List[float] = []

            def cost_grad(x):
                pub = (circ, [qop], [x])  # single PUB

                # --- quantum run
                tq0 = time.monotonic()
                e_job = estimator.run([pub])
                e_res = e_job.result()[0]
                tq1 = time.monotonic()

                md = e_res.metadata or {}
                qpu_time = md.get("execution_time", tq1 - tq0)

                timeline.append(
                    {
                        "iter": len(energy_hist) + 1,
                        "stage": "quantum",
                        "job_id": e_job.job_id(),
                        "num_qubits": qop.num_qubits,
                        "shots": self.shots,
                        "depth": circ.depth(),
                        "duration_s": qpu_time,
                        "arrival_time": md.get("created_at"),
                    }
                )

                # --- gradient run
                tc0 = time.monotonic()
                g_job = estimator.run_gradient([pub])
                grad = g_job.result().gradients[0]
                tc1 = time.monotonic()
                timeline.append(
                    {
                        "iter": len(energy_hist) + 1,
                        "stage": "classical",
                        "duration_s": tc1 - tc0,
                    }
                )

                energy = e_res.data.evs[0]
                energy_hist.append(energy)

                # live-update timeline file
                with open(
                    os.path.join(self.result_dir, f"{label}_timeline.json"), "w"
                ) as lf:
                    json.dump(timeline, lf, indent=2)

                print(
                    f"{label} iter {len(energy_hist)} "
                    f"E={energy:.6f}  QPU={qpu_time:.3f}s  CPU={(tc1-tc0):.3f}s"
                )
                return energy, np.array(grad)

            # BFGS optimisation
            x0 = np.zeros(n_params)
            minimize(
                fun=lambda x: cost_grad(x)[0],
                x0=x0,
                jac=lambda x: cost_grad(x)[1],
                method="BFGS",
                options={"maxiter": self.maxiter},
            )

            g_e = min(energy_hist)
            results[label] = {"energies": energy_hist, "ground_energy": g_e}

            np.savetxt(
                os.path.join(self.result_dir, f"{label}_energies.csv"),
                np.c_[np.arange(1, len(energy_hist) + 1), energy_hist],
                header="Iter,Energy",
                delimiter=",",
                comments="",
            )
            with open(
                os.path.join(self.result_dir, f"{label}_ground_energy.txt"), "w"
            ) as gf:
                gf.write(f"{g_e}\n")

        session.close()
        return results
