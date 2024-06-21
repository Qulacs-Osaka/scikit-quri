from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

# import better_exceptions
from scikit_quri.qnn.regressor import QNNRegressor
import numpy as np
from typing import Optional
from quri_parts.circuit.utils.circuit_drawer import draw_circuit
from quri_parts.circuit import UnboundParametricQuantumCircuit,LinearMappedUnboundParametricQuantumCircuit
from numpy.random import default_rng

from qulacs import Observable
from quri_parts.core.operator import Operator, pauli_label
from quri_parts.core.estimator.gradient import create_parameter_shift_gradient_estimator
from quri_parts.qulacs.estimator import (
    create_qulacs_vector_parametric_estimator,
    create_qulacs_vector_concurrent_parametric_estimator,
)
from quri_parts.algo.optimizer import Adam, LBFGS

def create_farhi_neven_ansatz(
    n_qubit: int, c_depth: int, seed: Optional[int] = 0
) -> UnboundParametricQuantumCircuit:
    circuit = UnboundParametricQuantumCircuit(n_qubit)
    zyu = list(range(n_qubit))
    rng = default_rng(seed)
    for _ in range(c_depth):
        rng.shuffle(zyu)
        for i in range(0, n_qubit - 1, 2):
            circuit.add_CNOT_gate(zyu[i + 1], zyu[i])
            circuit.add_ParametricRX_gate(zyu[i])
            circuit.add_ParametricRY_gate(zyu[i])
            circuit.add_CNOT_gate(zyu[i + 1], zyu[i])
            circuit.add_ParametricRY_gate(zyu[i])
            circuit.add_ParametricRX_gate(zyu[i])
    return circuit

def generate_noisy_sine(x_min, x_max, num_x):
    rng = default_rng(0)
    x_train = [[rng.uniform(x_min, x_max)] for _ in range(num_x)]
    y_train = [np.sin(np.pi * x[0]) for x in x_train]
    mag_noise = 0.01
    y_train += mag_noise * rng.random(num_x)
    # return np.array(x_train), np.array(y_train)
    return np.array(x_train).flatten(), np.array(y_train)

if __name__ == "__main__":
    from scikit_quri.circuit import LearningCircuit
    n_qubits = 2
    parametric_circuit = create_farhi_neven_ansatz(n_qubits,2)
    # parametric_circuit = LearningCircuit(n_qubits)
    
    # parametric_circuit.bind_parameters()
    op = Operator({
        pauli_label("X0 Y1"): 0.5 + 0.5j,
        pauli_label("Z0 X1"): 0.2,
    })
    op = Operator()
    for i in range(n_qubits):
        op.add_term(pauli_label(f"Z {i}"),1.0)
    estimator = create_qulacs_vector_parametric_estimator()
    concurrent_estimator = create_qulacs_vector_concurrent_parametric_estimator()
    gradient_estimator = create_parameter_shift_gradient_estimator(concurrent_estimator)
    adam = Adam()
    # Create Instance
    qnn = QNNRegressor(n_qubits,parametric_circuit,estimator,gradient_estimator,adam,op)

    x_train,y_train = generate_noisy_sine(-1.,1.,80)
    qnn.fit(x_train,y_train,len(x_train))
