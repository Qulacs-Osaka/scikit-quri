import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
from utils import array_f4

# //＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
n_qubits = 1
params = np.array(
    [
        np.pi / 3,
        np.pi / 6,
        # np.pi / 4,
        # np.pi / 2,
    ],
    dtype=float,
)

# //ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
hamiltonian = [("Y", 1.0)]

theta_list = [
    Parameter("θ0"),
    Parameter("θ1"),
]

qc = QuantumCircuit(n_qubits, n_qubits)
qc.rx(theta_list[0], 0)
qc.rz(theta_list[1], 0)

H = SparsePauliOp.from_list(hamiltonian)

# //ーーーーーーーーーーーーーーーーーーーーー
estimator = StatevectorEstimator()
gradient = ParamShiftEstimatorGradient(estimator)
job = gradient.run(circuits=[qc], observables=[H], parameter_values=[params.tolist()])
result = job.result()
print("Qiskit Simulator:", array_f4(result.gradients[0]))
