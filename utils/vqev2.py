# --*-- conding:utf-8 --*--
# @time:4/28/25 18:03
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:vqev2.py

import os, time, json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple
from scipy.optimize import minimize

from qiskit_ibm_runtime import Session, EstimatorV2
from qiskit import transpile


class MultiVQEPipeline:
    """
    VQE / Adapt-VQE runner with correct hardware transpilation & layout sync.
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
        os.makedirs(result_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def run(self, problems: Dict[str, Tuple[Any, Any]]) -> Dict[str, Dict[str, Any]]:
        max_q = max(qop.num_qubits for qop, _ in problems.values())
        backend = self.service.least_busy(
            simulator=False, operational=True, min_num_qubits=max_q
        )

        session = Session(backend=backend)
        estimator = EstimatorV2(mode=session)
        estimator.options.default_shots = self.shots

        results: Dict[str, Dict[str, Any]] = {}

        for label, (qop, solver) in problems.items():
            # ------------ Adapt-VQE branch -------------
            if hasattr(solver, "compute_minimum_eigenvalue"):
                res = solver.compute_minimum_eigenvalue(qop)
                g_e = res.eigenvalue
                results[label] = {"energies": [g_e], "ground_energy": g_e}
                with open(
                    os.path.join(self.result_dir, f"{label}_adapt_summary.json"), "w"
                ) as f:
                    json.dump(
                        {
                            "ground_energy": g_e,
                            "num_iterations": res.num_iterations,
                            "final_max_gradient": res.final_maximum_gradient,
                        },
                        f,
                        indent=2,
                    )
                continue

            # ------------ Standard VQE branch ----------
            raw_circ = solver                     # QuantumCircuit ansatz
            n_q = raw_circ.num_qubits

            # ① transpile to hardware gate set, keep logical lines 0..n_q-1
            circ_t = transpile(
                raw_circ,
                backend=backend,
                optimization_level=self.optimization_level,
                initial_layout=list(range(n_q)),
                routing_method="basic",
            )

            # ② apply same layout to observable
            qop_t = qop.apply_layout(circ_t.layout)

            # optimisation loop
            energy_hist: List[float] = []
            timeline: List[Dict[str, Any]] = []

            def cost_grad(pars):
                pub = (circ_t, [qop_t], [pars])

                t0 = time.monotonic()
                job_e = estimator.run([pub])
                ev = job_e.result()[0]
                t1 = time.monotonic()

                md = ev.metadata or {}
                qpu_dt = md.get("execution_time", t1 - t0)

                # gradient
                t2 = time.monotonic()
                job_g = estimator.run_gradient([pub])
                grad = job_g.result().gradients[0]
                t3 = time.monotonic()

                idx = len(energy_hist) + 1
                energy = ev.data.evs[0]
                energy_hist.append(energy)

                timeline.extend(
                    [
                        {
                            "iter": idx,
                            "stage": "quantum",
                            "job_id": job_e.job_id(),
                            "duration_s": qpu_dt,
                        },
                        {
                            "iter": idx,
                            "stage": "classical",
                            "duration_s": t3 - t2,
                        },
                    ]
                )
                with open(
                    os.path.join(self.result_dir, f"{label}_timeline.json"), "w"
                ) as lf:
                    json.dump(timeline, lf, indent=2)

                print(
                    f"{label} iter {idx}: E={energy:.6f}  "
                    f"QPU={qpu_dt:.3f}s  CPU={(t3 - t2):.3f}s"
                )
                return energy, np.array(grad)

            x0 = np.zeros(circ_t.num_parameters)
            minimize(
                fun=lambda p: cost_grad(p)[0],
                x0=x0,
                jac=lambda p: cost_grad(p)[1],
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

