# --*-- conding:utf-8 --*--
# @time:4/28/25 18:03
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:vqev2.py


import os
import json
import time
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
from scipy.optimize import minimize

from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

class MultiVQEPipeline:
    """
    Sequentially run VQE for multiple problems, each in its own Session.
    Record per-iteration quantum/classical timing in a timeline.
    problems: dict[label -> (qubit_operator, ansatz_circuit)]
    Returns dict[label -> {
        "energies": [...],
        "ground_energy": float,
        "parameters": [...],
        "timeline": [...]
    }]
    """

    def __init__(
        self,
        service,
        optimization_level: int = 3,
        shots: int = 200,
        min_qubit_num: int = 100,
        maxiter: int = 20,
        result_dir: str = "results"
    ):
        self.service = service
        self.optimization_level = optimization_level
        self.shots = shots
        self.min_qubit_num = min_qubit_num
        self.maxiter = maxiter
        self.result_dir = result_dir
        os.makedirs(result_dir, exist_ok=True)

    def _select_backend(self, required_qubits: int):
        """
        Select a least-busy IBMQ backend with at least `required_qubits`.
        """
        return self.service.least_busy(
            simulator=False,
            operational=True,
            min_num_qubits=max(required_qubits, self.min_qubit_num)
        )

    def _generate_pass_manager(self, backend):
        """
        Generate a preset pass manager for compiling circuits.
        """
        return generate_preset_pass_manager(
            target=backend.target,
            optimization_level=self.optimization_level
        )

    def run(self, problems: Dict[str, Tuple['SparsePauliOp', 'QuantumCircuit']]):
        """
        Run VQE sequentially for each (label, (qubit_op, ansatz)) in `problems`.
        Returns:
            results: { label: {
                "energies": [...],
                "ground_energy": float,
                "parameters": [...],
                "timeline": [...]
            } }
        """
        results: Dict[str, dict] = {}

        for label, (hamiltonian, ansatz) in problems.items():
            print(f"\nSolving problem: {label}")
            backend = self._select_backend(hamiltonian.num_qubits)

            pass_manager = self._generate_pass_manager(backend)
            compiled_ansatz = pass_manager.run(ansatz)
            compiled_hamiltonian = hamiltonian.apply_layout(compiled_ansatz.layout)

            energies: List[float] = []
            timeline: List[dict] = []

            with Session(backend=backend) as session:
                opts = Estimator.Options()
                opts.default_shots = self.shots
                estimator = Estimator(mode=session, options=opts)

                def cost_fn(params: np.ndarray) -> float:
                    """
                    Cost function for a given parameter vector `params`.
                    Runs the compiled ansatz and measures expectation of compiled_hamiltonian.
                    Records energy, QPU execution time, and queue delay.
                    """
                    pub = (compiled_ansatz, [compiled_hamiltonian], [params.tolist()])
                    job = estimator.run(pubs=[pub])
                    result = job.result()[0]

                    energy = float(result.data.evs[0])
                    energies.append(energy)

                    md = result.metadata or {}
                    qpu_time = md.get("execution_time", 0.0)
                    queue_delay = None
                    if md.get("queued_at") and md.get("started_at"):
                        queued_at  = datetime.fromisoformat(md["queued_at"].replace("Z", "+00:00"))
                        started_at = datetime.fromisoformat(md["started_at"].replace("Z", "+00:00"))
                        queue_delay = (started_at - queued_at).total_seconds()

                    timeline.append({
                        "iter": len(energies),
                        "stage": "quantum",
                        "qpu_time_s": qpu_time,
                        "queue_delay_s": queue_delay
                    })
                    print(f"{label} iter {len(energies):3d} | E = {energy:.6f} | QPU time = {qpu_time:.3f}s | Queue delay = {queue_delay}")
                    return energy

                def callback(_xk: np.ndarray):
                    """
                    Called after each optimization step. Records a classical-timing entry.
                    """
                    t0 = time.time()
                    t1 = time.time()
                    timeline.append({
                        "iter": len(energies),
                        "stage": "classical",
                        "cpu_time_s": t1 - t0
                    })
                    # Write intermediate timeline to disk
                    with open(f"{self.result_dir}/{label}_timeline.json", "w") as fp:
                        json.dump(timeline, fp, indent=2)

                x0 = np.random.random(compiled_ansatz.num_parameters)
                res = minimize(
                    fun=cost_fn,
                    x0=x0,
                    method="COBYLA",
                    callback=callback,
                    options={"maxiter": self.maxiter}
                )

            # After optimization, write energies.csv and final timeline.json
            np.savetxt(
                os.path.join(self.result_dir, f"{label}_energies.csv"),
                np.column_stack((np.arange(1, len(energies) + 1), energies)),
                header="iter,energy",
                delimiter=",",
                comments="",
                fmt="%.6f"
            )
            with open(os.path.join(self.result_dir, f"{label}_timeline.json"), "w") as fp:
                json.dump(timeline, fp, indent=2)

            results[label] = {
                "energies": energies,
                "ground_energy": min(energies) if energies else None,
                "parameters": res.x.tolist(),
                "timeline": timeline
            }

        return results





