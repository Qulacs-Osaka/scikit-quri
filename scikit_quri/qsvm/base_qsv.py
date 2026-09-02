from __future__ import annotations

import warnings
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from quri_parts.circuit import QuantumCircuit
from sklearn import svm

from scikit_quri.backend import BaseSampler
from scikit_quri.circuit import LearningCircuit
from scikit_quri.state import OverlapEstimator


class SVMethodType(Enum):
    SVC = 1
    SVR = 2


class BaseQSV:
    """Base class for Quantum Support Vector Machine."""

    def __init__(
        self,
        circuit: LearningCircuit,
        sv_method_type: SVMethodType,
        sampler: Optional[BaseSampler] = None,
        n_shots: int = 1000,
        max_iter: int = int(1e7),
        verbose: bool = False,
    ) -> None:
        if circuit.learning_params_count:
            raise ValueError(
                f"{type(self).__name__} is a kernel method: it evaluates the feature map "
                f"only and never optimizes parameters, so the circuit must have no "
                f"learnable parameters (got {circuit.learning_params_count}). Use a "
                f"data-encoding circuit such as create_ibm_embedding_circuit. Passing a "
                f"trainable ansatz used to fail later with an unrelated IndexError."
            )
        self.circuit = circuit
        self.sv_method_type = sv_method_type
        # The backend is a property of the model, not training data. Taking it in fit()
        # kept this estimator out of Pipeline and every cross-validation helper, which
        # all call fit(X, y).
        self.sampler = sampler
        self.n_shots = n_shots
        self.max_iter = max_iter
        self.verbose = verbose
        self.data_circuits: list[QuantumCircuit] = []
        self.n_qubit = circuit.n_qubits
        self.estimator = None

    def fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        sampler: Optional[BaseSampler] = None,
        n_shots: Optional[int] = None,
        max_iter: Optional[int] = None,
        verbose: Optional[bool] = None,
    ) -> None:
        """Fit the model to the training data.

        Args:
            x: Training feature matrix of shape (n_samples, n_features).
            y: Training labels.
            sampler: Deprecated. Pass the sampler to ``__init__`` instead.
            n_shots: Deprecated. Pass it to ``__init__`` instead.
            max_iter: Deprecated. Pass it to ``__init__`` instead.
            verbose: Deprecated. Pass it to ``__init__`` instead.

        """
        for name, value in (
            ("sampler", sampler),
            ("n_shots", n_shots),
            ("max_iter", max_iter),
            ("verbose", verbose),
        ):
            if value is not None:
                warnings.warn(
                    f"Passing {name} to fit() is deprecated; pass it to the constructor "
                    "instead. Configuration in fit() prevents use with scikit-learn's "
                    "Pipeline and cross-validation helpers, which call fit(X, y).",
                    DeprecationWarning,
                    stacklevel=2,
                )
                setattr(self, name, value)

        if self.sampler is None:
            raise ValueError(
                "No sampler configured. Pass one to the constructor, e.g. "
                f"{type(self).__name__}(circuit, sampler=QulacsSampler())."
            )

        n_x = len(x)
        gram_train = np.zeros((n_x, n_x))
        self.data_circuits = [self._run_circuit(x[i]) for i in range(n_x)]
        self.estimator = OverlapEstimator(self.sampler, self.n_shots)
        gram_train = self.estimator.estimate_concurrent(
            self.data_circuits, self.data_circuits
        ).reshape(n_x, n_x)
        # max_iter must be specified at instantiation time for sklearn SVM
        if self.sv_method_type == SVMethodType.SVC:
            self.sv_method = svm.SVC(
                kernel="precomputed", max_iter=self.max_iter, verbose=self.verbose
            )
        elif self.sv_method_type == SVMethodType.SVR:
            self.sv_method = svm.SVR(
                kernel="precomputed", max_iter=self.max_iter, verbose=self.verbose
            )
        self.sv_method.fit(gram_train, y)
        self.gram_train = gram_train

    def predict(self, xs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict outcomes for the given test data.

        Args:
            xs: Test feature matrix of shape (n_samples, n_features).

        Returns:
            pred: Predicted values of shape (n_samples,).

        """
        if self.estimator is None:
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
        sampler: Sampling backend, e.g. ``QulacsSampler()`` or ``OqtopusSampler(...)``.
        n_shots: Shots per circuit execution.
        max_iter: Iteration cap for the underlying scikit-learn solver.
        verbose: Forwarded to the scikit-learn solver.

    """

    def __init__(
        self,
        circuit: LearningCircuit,
        sampler: Optional[BaseSampler] = None,
        n_shots: int = 1000,
        max_iter: int = int(1e7),
        verbose: bool = False,
    ) -> None:
        super().__init__(
            circuit,
            SVMethodType.SVC,
            sampler=sampler,
            n_shots=n_shots,
            max_iter=max_iter,
            verbose=verbose,
        )


class QSVR(BaseQSV):
    """Quantum Support Vector Regressor.

    Args:
        circuit: LearningCircuit
        sampler: Sampling backend, e.g. ``QulacsSampler()`` or ``OqtopusSampler(...)``.
        n_shots: Shots per circuit execution.
        max_iter: Iteration cap for the underlying scikit-learn solver.
        verbose: Forwarded to the scikit-learn solver.

    """

    def __init__(
        self,
        circuit: LearningCircuit,
        sampler: Optional[BaseSampler] = None,
        n_shots: int = 1000,
        max_iter: int = int(1e7),
        verbose: bool = False,
    ) -> None:
        super().__init__(
            circuit,
            SVMethodType.SVR,
            sampler=sampler,
            n_shots=n_shots,
            max_iter=max_iter,
            verbose=verbose,
        )
