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
from scikit_quri.circuit import LearningCircuit


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

def preprocess_x(x: np.ndarray, i: int) -> float:
    xa = x[i % len(x)]
    clamped = min(1,max(-1,xa))
    return clamped

def create_farhi_neven_ansatz(n_qubit:int,c_depth:int):
    linear_param_circuit = LinearMappedUnboundParametricQuantumCircuit(n_qubit)
    theta,x = linear_param_circuit.add_parameters("theta","x")
    for i in range(n_qubit):
        linear_param_circuit.add_ParametricRY_gate(i,{theta:1.})
        linear_param_circuit.add_ParametricRZ_gate(i,lambda x,i=i: np.arccos(preprocess_x(x,i) * preprocess_x(x,i)))
    zyu = list(range(n_qubit))
    rng = default_rng(0)
    for _ in range(c_depth):
        rng.shuffle(zyu)
        for i in range(0, n_qubit - 1, 2):
            pass
def _create_farhi_neven_ansatz(
    n_qubit: int, c_depth: int, seed: Optional[int] = 0
) -> LearningCircuit:
    circuit = LearningCircuit(n_qubit)
    rng = default_rng(seed)
    for i in range(n_qubit):
        circuit.add_input_RY_gate(i,lambda x,i=i: np.arcsin(preprocess_x(x,i)))
        circuit.add_input_RZ_gate(i,lambda x,i=i: np.arccos(preprocess_x(x,i) * preprocess_x(x,i)))
    
    zyu = list(range(n_qubit))

    for _ in range(c_depth):
        rng.shuffle(zyu)
        for i in range(0, n_qubit - 1, 2):
            circuit.circuit.add_CNOT_gate(zyu[i+1],zyu[i])
            circuit.add_parametric_RX_gate(zyu[i])
            circuit.add_parametric_RY_gate(zyu[i])
            circuit.circuit.add_CNOT_gate(zyu[i+1],zyu[i])
            circuit.add_parametric_RY_gate(zyu[i])
            circuit.add_parametric_RX_gate(zyu[i])
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
    from quri_parts.circuit.utils.circuit_drawer import draw_circuit
    import matplotlib.pyplot as plt
    n_qubits = 2
    n_outputs = 1
    parametric_circuit = _create_farhi_neven_ansatz(n_qubits,2,1)
    print(f"{parametric_circuit.circuit.gates_and_params=}")
    # parametric_circuit = LearningCircuit(n_qubits)
    
    # parametric_circuit.bind_parameters()
    op = Operator()
    for i in range(n_outputs):
        op.add_term(pauli_label(f"Z {i}"),1.0)
    draw_circuit(parametric_circuit.circuit)
    estimator = create_qulacs_vector_parametric_estimator()
    concurrent_estimator = create_qulacs_vector_concurrent_parametric_estimator()
    gradient_estimator = create_parameter_shift_gradient_estimator(concurrent_estimator)
    adam = Adam()
    # Create Instance
    qnn = QNNRegressor(n_qubits,parametric_circuit,estimator,gradient_estimator,adam,op)
    x_train,y_train = generate_noisy_sine(-1.,1.,80)
    x_test,y_test = generate_noisy_sine(-1.,1.,80)
    print(f"{x_train=}")
    qnn.fit(x_train,y_train,len(x_train))
    y_pred = qnn.predict(x_test)
    print(f"{y_test.reshape(-1,1)=}")
    print(f"{y_pred=}")

    y_pred = y_pred.flatten()
    plt.figure(figsize=(5,5))

    # plt.plot(x_test,y_test,marker="o",color="orange",label="Test")
    plt.plot(
        np.sort(x_test),
        y_test[np.argsort(x_test.flatten())],label="Test")
    
    x_true = np.linspace(-1,1,100)
    y_true = np.sin(np.pi * x_true)
    plt.plot(x_true,y_true,label="True")
    plt.plot(
        np.sort(x_test.flatten()),
        y_pred[np.argsort(x_test.flatten())],
        label="Prediction")
    plt.legend()
    
    cost = np.mean((y_test- y_pred) ** 2)
    print(f"{cost=}")
    # plt.show()
    plt.savefig("quri-parts.png")
