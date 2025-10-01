import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
from utils import array_f4

# //＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
n_qubits = 2
params = np.array(
    [
        np.pi / 4,
        np.pi / 2,
        np.pi / 4,
        np.pi / 2,
    ],
    dtype=float,
)

# //ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
hamiltonian = [
    ("XI", 1.0),
    ("IX", 1.0),
]

theta_list = [
    Parameter("θ0"),
    Parameter("θ1"),
    Parameter("θ2"),
    Parameter("θ3"),
]

qc = QuantumCircuit(n_qubits, n_qubits)
qc.ry(theta_list[0], 0)
qc.rz(theta_list[1], 0)
qc.rz(theta_list[2], 1)
qc.ry(theta_list[3], 1)

H = SparsePauliOp.from_list(hamiltonian)

# //ーーーーーーーーーーーーーーーーーーーーー
estimator = StatevectorEstimator()
gradient = ParamShiftEstimatorGradient(estimator)
job = gradient.run(circuits=[qc], observables=[H], parameter_values=[params.tolist()])
result = job.result()
print("Qiskit Simulator:", array_f4(result.gradients[0]))
