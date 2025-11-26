from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray
from quri_parts.circuit import QuantumCircuit
from quri_parts.core.sampling import Sampler
from sklearn import svm

from scikit_quri.circuit import LearningCircuit
from scikit_quri.state import overlap_estimator


class SVMethodType(Enum):
    SVC = 1
    SVR = 2


class BaseQSV:
    """Quantum Support Vector Machineの基底クラス."""

    def __init__(self, circuit: LearningCircuit, sv_method_type: SVMethodType) -> None:
        self.circuit = circuit
        self.sv_method_type = sv_method_type
        self.data_circuits: list[QuantumCircuit] = []
        self.n_qubit = circuit.n_qubits

    def fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        sampler: Sampler,
        n_shots: int = 1000,
        max_iter: int = int(1e7),
    ) -> None:
        """trainデータから学習を行う.

        Args:
            x: 学習データの特徴量(二次元配列)
            y: 学習データのラベル
            sampler: sampler function
            n_shots: ショット数. Defaults to 1000.
            max_iter: svmの最大反復回数. Defaults to 1e6.

        """
        n_x = len(x)
        gram_train = np.zeros((n_x, n_x))
        self.data_circuits = [self._run_circuit(x[i]) for i in range(n_x)]
        self.estimator = overlap_estimator(sampler, n_shots)
        for i in range(n_x):
            for j in range(n_x):
                ket_circuit = self.data_circuits[i]
                bra_circuit = self.data_circuits[j]
                gram_train[i][j] = self.estimator.estimate(ket_circuit, bra_circuit)
            print("\r", f"{i}/{len(x)}", end="")
        print()
        print("fitting SV...")
        print(gram_train[0][1], gram_train[1][0])
        if self.sv_method_type == SVMethodType.SVC:
            self.sv_method = svm.SVC(kernel="precomputed", max_iter=max_iter, verbose=True)
        elif self.sv_method_type == SVMethodType.SVR:
            self.sv_method = svm.SVR(kernel="precomputed", max_iter=max_iter, verbose=True)
        self.sv_method.fit(gram_train, y)

    def predict(self, xs: NDArray[np.float64]) -> NDArray[np.float64]:
        """testデータから予測を行う.

        Args:
            xs: テストデータの特徴量(二次元配列)

        Returns:
            pred: 予測結果

        """
        if not self.estimator:
            raise ValueError("run fit() before predict")
        n_x = len(xs)
        gram_test = np.zeros((n_x, len(self.data_circuits)))
        test_circuits = [self._run_circuit(xs[i]) for i in range(n_x)]
        for i in range(n_x):
            for j in range(len(self.data_circuits)):
                ket_circuit = test_circuits[i]
                bra_circuit = self.data_circuits[j]
                gram_test[i][j] = self.estimator.estimate(ket_circuit, bra_circuit)
            print("\r", f"{i}/{n_x}", end="")
        print()
        pred: NDArray[np.float64] = self.sv_method.predict(gram_test)
        return pred

    def _run_circuit(self, x: NDArray[np.float64]) -> QuantumCircuit:
        """inputデータを回路にapplyした回路を返す."""
        return self.circuit.bind_input_and_parameters(x, np.array([])).get_mutable_copy()


class QSVC(BaseQSV):
    """Quantum Support Vector Classifier.

    Args:
        circuit: LearningCircuit

    """

    def __init__(self, circuit: LearningCircuit) -> None:
        super().__init__(circuit, SVMethodType.SVC)


class QSVR(BaseQSV):
    """Quantum Support Vector Regressor.

    Args:
        circuit: LearningCircuit

    """

    def __init__(self, circuit: LearningCircuit) -> None:
        super().__init__(circuit, SVMethodType.SVR)
