import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
from utils import array_f4

# //＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
n_qubits = 3
params = np.array(
    [
        0.456,
        0.789,
    ],
    dtype=float,
)

# //ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
hamiltonian = [
    ("XII", 1.0),
    ("IXI", 1.0),
    ("IIX", 1.0),
]

theta_list = [
    Parameter("θ0"),
    Parameter("θ1"),
]

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(0)
qc.rx(0.5, 1)
qc.rx(theta_list[0], 0)
qc.ry(0.123, 1)
qc.rz(theta_list[1], 2)

H = SparsePauliOp.from_list(hamiltonian)

# //ーーーーーーーーーーーーーーーーーーーーー
estimator = StatevectorEstimator()
gradient = ParamShiftEstimatorGradient(estimator)
job = gradient.run(circuits=[qc], observables=[H], parameter_values=[params.tolist()])
result = job.result()
print("Qiskit Simulator:", array_f4(result.gradients[0]))
