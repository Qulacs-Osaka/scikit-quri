from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray
from quri_parts.circuit import QuantumCircuit
from quri_parts.core.sampling import ConcurrentSampler
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
        sampler: ConcurrentSampler,
        n_shots: int = 1000,
        max_iter: int = int(1e7),
        verbose: bool = False,
    ) -> None:
        """trainデータから学習を行う.

        Args:
            x: 学習データの特徴量(二次元配列)
            y: 学習データのラベル
            sampler: sampler function
            n_shots: ショット数. Defaults to 1000.
            max_iter: svmの最大反復回数. Defaults to 1e6.
            verbose: svmの学習過程の出力の有無. Defaults to False.

        """
        n_x = len(x)
        gram_train = np.zeros((n_x, n_x))
        self.data_circuits = [self._run_circuit(x[i]) for i in range(n_x)]
        self.estimator = overlap_estimator(sampler, n_shots)
        gram_train = self.estimator.estimate_concurrent(
            self.data_circuits, self.data_circuits
        ).reshape(n_x, n_x)
        # svmのmax_iterはインスタンス作成時に指定する必要があるためここで作成
        if self.sv_method_type == SVMethodType.SVC:
            self.sv_method = svm.SVC(kernel="precomputed", max_iter=max_iter, verbose=verbose)
        elif self.sv_method_type == SVMethodType.SVR:
            self.sv_method = svm.SVR(kernel="precomputed", max_iter=max_iter, verbose=verbose)
        self.sv_method.fit(gram_train, y)
        self.gram_train = gram_train

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
        gram_test = self.estimator.estimate_concurrent(test_circuits, self.data_circuits).reshape(
            n_x, len(self.data_circuits)
        )
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
