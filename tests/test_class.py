
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scikit_quri.qnn.new_classifier import QNNClassifier
# import better_exceptions
import numpy as np
from typing import Optional
from quri_parts.circuit.utils.circuit_drawer import draw_circuit
from quri_parts.circuit import UnboundParametricQuantumCircuit,LinearMappedUnboundParametricQuantumCircuit
from numpy.random import default_rng

from qulacs import Observable
from quri_parts.core.operator import Operator, pauli_label
from quri_parts.core.estimator.gradient import (
    create_parameter_shift_gradient_estimator,
    create_numerical_gradient_estimator,)
from quri_parts.qulacs.estimator import (
    create_qulacs_vector_concurrent_estimator,
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
            circuit
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

    import pandas as pd
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    x = df.loc[:, ["petal length (cm)", "petal width (cm)"]]
    x_train,x_test,y_train,y_test = train_test_split(x,iris.target,test_size=0.25,random_state=0)

    x_train = x_train.to_numpy()

    from scikit_quri.circuit import LearningCircuit
    from quri_parts.circuit.utils.circuit_drawer import draw_circuit
    import matplotlib.pyplot as plt
    n_qubits = 5
    num_class = 3
    parametric_circuit = _create_farhi_neven_ansatz(n_qubits,1,)
    # print(f"{parametric_circuit.circuit.gates_and_params=}")
    # parametric_circuit = LearningCircuit(n_qubits)
    
    # parametric_circuit.bind_parameters()
    ops = []
    for i in range(num_class):
        op = Operator({pauli_label(f"Z {i}"):1.0})
        ops.append(op)
    # draw_circuit(parametric_circuit.circuit)
    estimator = create_qulacs_vector_parametric_estimator()
    concurrent_estimator = create_qulacs_vector_concurrent_estimator()
    # gradient_estimator = create_parameter_shift_gradient_estimator(concurrent_estimator)
    gradient_estimator = create_parameter_shift_gradient_estimator(create_qulacs_vector_concurrent_parametric_estimator())
    adam = Adam()
    # Create Instance
    qnn = QNNClassifier(parametric_circuit,num_class,concurrent_estimator,gradient_estimator,adam,ops,y_exp_ratio=4.0)
    print(f"{x_train.shape=}")
    qnn.fit(x_train,y_train,maxiter=500)
    # y_pred = qnn.predict(x_test)
    # print(f"{y_test.reshape(-1,1)=}")
    # print(f"{y_pred=}")

