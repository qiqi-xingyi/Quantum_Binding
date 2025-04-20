# --*-- conding:utf-8 --*--
# @time:4/19/25 21:38
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:vqe.py

from qiskit import IBMQ, QuantumCircuit, transpile
from qiskit.utils import QuantumInstance
from qiskit.tools.monitor import job_monitor

# Step 1: Load IBM Quantum account (replace with your API token if not saved)
IBMQ.load_account()  # Assumes API token is saved; else use: IBMQ.save_account('YOUR_API_TOKEN')

# Step 2: Select a provider and backend
provider = IBMQ.get_provider(hub='ibm-q')  # Choose appropriate hub
backend = provider.get_backend('ibm_brisbane')  # Replace with desired backend (e.g., ibmq_manila)

# Step 3: Create a simple 2-qubit Bell state circuit
qc = QuantumCircuit(2, 2)  # 2 qubits, 2 classical bits
qc.h(0)  # Apply Hadamard gate to qubit 0
qc.cx(0, 1)  # Apply CNOT gate (qubit 0 controls qubit 1)
qc.measure([0, 1], [0, 1])  # Measure both qubits

# Step 4: Transpile the circuit for the backend
transpiled_circuit = transpile(qc, backend=backend, optimization_level=1)

# Step 5: Set up quantum instance with the backend
quantum_instance = QuantumInstance(backend=backend, shots=8192)

# Step 6: Run the circuit on the IBM Quantum backend
job = quantum_instance.execute(transpiled_circuit)

# Step 7: Monitor the job
job_monitor(job)

# Step 8: Retrieve and print results
result = job.result()
counts = result.get_counts()
print(f"Measurement counts: {counts}")

# Optional: Plot histogram of results (requires matplotlib)
from qiskit.visualization import plot_histogram
plot_histogram(counts).show()


from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
from qiskit import QuantumCircuit

service = QiskitRuntimeService()

backend_name = "ibm_lagos"

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure(range(2), range(2))

with Session(service=service, backend=backend_name) as session:
    sampler = Sampler(session=session)
    job = sampler.run(circuits=qc, shots=1024)
    result = job.result()

print("Counts:", result.quasi_dists())
