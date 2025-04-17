# --*-- conding:utf-8 --*--
# @Time : 3/27/25 5:11 PM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : quantum_service.py

from utils.config_manager import ConfigManager
from qiskit_ibm_runtime import QiskitRuntimeService

def test_ibm_quantum_service():
    try:

        # 1) read config
        cfg = ConfigManager("config.txt")

        service = QiskitRuntimeService(
            channel='ibm_quantum',
            instance=cfg.get("INSTANCE"),
            token=cfg.get("TOKEN")
        )

        # Retrieve the list of available backends.
        backends = service.backends()
        if backends:
            print("IBM Quantum Service connected successfully!")
            print("Available backends:")
            for backend in backends:
                print(" -", backend.name)
        else:
            print("No backends available. Please check your service connection.")

    except Exception as e:
        print("Failed to connect to IBM Quantum Service:")
        print(e)


if __name__ == "__main__":
    test_ibm_quantum_service()
