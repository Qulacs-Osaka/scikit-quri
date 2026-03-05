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
    """Base class for Quantum Support Vector Machine."""

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
        """Fit the model to the training data.

        Args:
            x: Training feature matrix of shape (n_samples, n_features).
            y: Training labels.
            sampler: Concurrent sampler function.
            n_shots: Number of shots per circuit execution. Defaults to 1000.
            max_iter: Maximum number of iterations for the SVM solver. Defaults to 1e7.
            verbose: Whether to print the SVM training progress. Defaults to False.

        """
        n_x = len(x)
        gram_train = np.zeros((n_x, n_x))
        self.data_circuits = [self._run_circuit(x[i]) for i in range(n_x)]
        self.estimator = overlap_estimator(sampler, n_shots)
        gram_train = self.estimator.estimate_concurrent(
            self.data_circuits, self.data_circuits
        ).reshape(n_x, n_x)
        # max_iter must be specified at instantiation time for sklearn SVM
        if self.sv_method_type == SVMethodType.SVC:
            self.sv_method = svm.SVC(kernel="precomputed", max_iter=max_iter, verbose=verbose)
        elif self.sv_method_type == SVMethodType.SVR:
            self.sv_method = svm.SVR(kernel="precomputed", max_iter=max_iter, verbose=verbose)
        self.sv_method.fit(gram_train, y)
        self.gram_train = gram_train

    def predict(self, xs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict outcomes for the given test data.

        Args:
            xs: Test feature matrix of shape (n_samples, n_features).

        Returns:
            pred: Predicted values of shape (n_samples,).

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
        """Return a bound circuit with the input data applied."""
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
