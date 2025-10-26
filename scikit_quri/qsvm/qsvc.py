from typing import List

import numpy as np
from numpy.typing import NDArray
from sklearn import svm
from quri_parts.circuit import QuantumCircuit
from quri_parts.rust.circuit.circuit_parametric import ImmutableBoundParametricQuantumCircuit
from quri_parts.core.sampling import Sampler
from scikit_quri.state import overlap_estimator
from scikit_quri.circuit import LearningCircuit


class QSVC:
    def __init__(self, circuit: LearningCircuit):
        self.svc = svm.SVC(kernel="precomputed", verbose=True)
        self.circuit = circuit
        self.data_circuits: List[QuantumCircuit] = []
        self.n_qubit: int = circuit.n_qubits
        self.estimator = None

    def run_circuit(self, x: NDArray[np.float64]) -> ImmutableBoundParametricQuantumCircuit:
        # ここにはparametrizeされたcircuitは入ってこないはず...
        circuit = self.circuit.bind_input_and_parameters(x, np.array([]))
        return circuit

    def fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.int64],
        sampler: Sampler,
        n_shots: int = 1000,
    ):
        """
        Args:
            x: 学習データの特徴量(二次元配列)
            y: 学習データのラベル
            sampler: 期待値計算に用いるSampler
            n_shots: ショット数. Defaults to 1000.
        """
        kar = np.zeros((len(x), len(x)))
        for i in range(len(x)):
            circuit = self.run_circuit(x[i])
            self.data_circuits.append(circuit.get_mutable_copy())

        self.estimator = overlap_estimator(sampler, n_shots)

        for i in range(len(x)):
            for j in range(len(x)):
                ket_circuit = self.data_circuits[i]
                bra_circuit = self.data_circuits[j]
                kar[i][j] = self.estimator.estimate(ket_circuit, bra_circuit)
            print("\r", f"{i}/{len(x)}", end="")
        print("")
        print(kar.shape)
        print("fitting SVC...")
        self.svc.fit(kar, y)

    def predict(self, xs: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.estimator is None:
            raise ValueError("run fit() before predict")
        kar = np.zeros((len(xs), len(self.data_circuits)))
        new_circuits = []
        for i in range(len(xs)):
            x_qc = self.run_circuit(xs[i])
            new_circuits.append(x_qc.get_mutable_copy())
        for i in range(len(xs)):
            for j in range(len(self.data_circuits)):
                kar[i][j] = self.estimator.estimate(new_circuits[i], self.data_circuits[j])
            print("\r", f"{i}/{len(xs)}", end="")
        print("")

        pred: NDArray[np.float64] = self.svc.predict(kar)
        return pred
