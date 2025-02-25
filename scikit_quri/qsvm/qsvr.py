# mypy: ignore-errors
from typing import List

import numpy as np
from numpy.typing import NDArray
from ..circuit import LearningCircuit
from sklearn import svm
from quri_parts.core.state import QuantumState, quantum_state
from ..state.overlap_estimator import overlap_estimator


class QSVR:
    def __init__(self, circuit: LearningCircuit):
        self.svc = svm.SVR(kernel="precomputed")
        self.circuit = circuit
        self.data_states: List[QuantumState] = []
        self.n_qubit: int = circuit.n_qubits

    def run_circuit(self, x: NDArray[np.float64]):
        # ここにはparametrizeされたcircuitは入ってこないはず...
        circuit = self.circuit.bind_input_and_parameters(x, [])
        state = quantum_state(n_qubits=self.n_qubit, circuit=circuit)
        return state

    def fit(self, x: NDArray[np.float64], y: NDArray[np.int_]):
        # self.n_qubit = len(x[0])
        kar = np.zeros((len(x), len(x)))
        for i in range(len(x)):
            self.data_states.append(self.run_circuit(x[i]))
        self.estimator = overlap_estimator(self.data_states.copy())
        for i in range(len(x)):
            for j in range(len(x)):
                kar[i][j] = self.estimator.estimate(i, j)
        self.svc.fit(kar, y)

    def predict(self, xs: NDArray[np.float64]) -> NDArray[np.float64]:
        kar = np.zeros((len(xs), len(self.data_states)))
        new_states = []
        for i in range(len(xs)):
            x_qc = self.run_circuit(xs[i])
            new_states.append(x_qc)
        self.estimator.add_data(new_states)
        offset = len(self.data_states)
        for i in range(len(xs)):
            for j in range(len(self.data_states)):
                kar[i][j] = self.estimator.estimate(offset + i, j)

        pred: NDArray[np.float64] = self.svc.predict(kar)
        return pred
