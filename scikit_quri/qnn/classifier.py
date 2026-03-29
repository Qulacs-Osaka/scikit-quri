# mypy: ignore-errors
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from quri_parts.algo.optimizer import Optimizer, Params, OptimizerStatus
from quri_parts.core.estimator import Estimatable
from scikit_quri.circuit import LearningCircuit
from scikit_quri.backend import BaseEstimator
from typing import List, Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss
from quri_parts.core.operator import Operator, pauli_label

from ._qnn_common import GradientEstimatorType, predict_inner, estimate_grad


@dataclass
class QNNClassifier:
    """Class to solve classification problems by quantum neural networks.
    The prediction is made by making a vector which predicts one-hot encoding of labels.
    The prediction is made by
    1. taking expectation values of Pauli Z operator of each qubit ``<Z_i>``,
    2. taking softmax function of the vector (``<Z_0>, <Z_1>, ..., <Z_{n-1}>``).

    Args:
        ansatz: Circuit to use in the learning.
        num_class: The number of classes; the number of qubits to measure. must be n_qubits >= num_class .
        estimator: Estimator to use. It must be a concurrent estimator.
        gradient_estimator: Gradient estimator to use.
        optimizer: Solver to use. use :py:class:`~quri_parts.algo.optimizer.Adam` or :py:class:`~quri_parts.algo.optimizer.LBFGS` method.

    Example:
        >>> from scikit_quri.qnn.classifier import QNNClassifier
        >>> from scikit_quri.circuit import create_qcl_ansatz
        >>> from quri_parts.core.estimator.gradient import (
        >>>     create_numerical_gradient_estimator,
        >>> )
        >>> from quri_parts.qulacs.estimator import (
        >>>     create_qulacs_vector_concurrent_estimator,
        >>>     create_qulacs_vector_concurrent_parametric_estimator,
        >>> )
        >>> from quri_parts.algo.optimizer import Adam
        >>> num_class = 3
        >>> nqubit = 5
        >>> c_depth = 3
        >>> time_step = 0.5
        >>> circuit = create_qcl_ansatz(nqubit, c_depth, time_step, 0)
        >>> adam = Adam()
        >>> estimator = create_qulacs_vector_concurrent_estimator()
        >>> gradient_estimator = create_numerical_gradient_estimator(
        >>>    create_qulacs_vector_concurrent_parametric_estimator(), delta=1e-10
        >>> )
        >>> qnn = QNNClassifier(circuit, num_class, estimator, gradient_estimator, adam)
        >>> qnn.fit(x_train, y_train, maxiter)
        >>> y_pred = qnn.predict(x_test).argmax(axis=1)
    """

    ansatz: LearningCircuit
    num_class: int
    estimator: BaseEstimator
    gradient_estimator: GradientEstimatorType
    optimizer: Optimizer

    operator: List[Estimatable] = field(default_factory=list)

    x_norm_range: float = field(default=1.0)
    y_norm_range: float = field(default=0.7)

    do_x_scale: bool = field(default=True)
    do_y_scale: bool = field(default=True)
    n_outputs: int = field(default=1)
    y_exp_ratio: float = field(default=2.2)

    trained_param: Optional[Params] = field(default=None)

    n_qubit: int = field(init=False)

    def __post_init__(self) -> None:
        if not issubclass(type(self.estimator), BaseEstimator):
            raise TypeError("estimator must be a subclass of BaseEstimator")
        self.n_qubit = self.ansatz.n_qubits
        if self.do_x_scale:
            self.scale_x_scaler = MinMaxScaler(
                feature_range=(-self.x_norm_range, self.x_norm_range)  # type: ignore
            )

    def _softmax(self, x: NDArray[np.float64], axis=None) -> NDArray[np.float64]:
        x_max = np.amax(x, axis=axis, keepdims=True)
        exp_x_shifted = np.exp(x - x_max)
        return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)

    def _cost(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        params: NDArray[np.float64],
    ):
        if x_train.ndim == 1:
            x_train = x_train.reshape(-1, 1)

        if self.do_x_scale:
            x_scaled = self.scale_x_scaler.fit_transform(x_train)
        else:
            x_scaled = x_train

        cost = self.cost_func(x_scaled, y_train, params)
        return cost

    def fit(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        maxiter: int = 100,
    ):
        """
        Args:
            x_train: List of training data inputs whose shape is (n_samples, n_features).
            y_train: List of labels to fit. Labels must be represented as integers. Shape is (n_samples,).
            maxiter: The number of maximum iterations for the optimizer.
        Returns:
            None
        """
        if x_train.ndim == 1:
            x_train = x_train.reshape(-1, 1)

        if self.do_x_scale:
            x_scaled = self.scale_x_scaler.fit_transform(x_train)
        else:
            x_scaled = x_train

        # operator設定
        operators = []
        for i in range(self.num_class):
            operators.append(Operator({pauli_label(f"Z {i}"): 1.0}))
        self.operator = operators

        parameter_count = self.ansatz.learning_params_count
        if self.trained_param is None:
            init_params = 2 * np.pi * np.random.random(parameter_count)
        else:
            init_params = self.trained_param
        # print(f"{init_params=}")
        optimizer_state = self.optimizer.get_init_state(init_params)

        def cost_func(params):
            return self.cost_func(x_scaled, y_train, params)

        # cost_func = partial(self.cost_func, x_scaled=x_scaled, y_train=y_train)
        def grad_func(params):
            return self.cost_func_grad(x_scaled, y_train, params)

        # grad_func = partial(self.cost_func_grad, x_scaled=x_scaled, y_train=y_train)
        c = 0
        while maxiter > c:
            optimizer_state = self.optimizer.step(optimizer_state, cost_func, grad_func)
            print("\r", f"iter:{c}/{maxiter} cost:{optimizer_state.cost=}", end="")

            if optimizer_state.status == OptimizerStatus.CONVERGED:
                break
            if optimizer_state.status == OptimizerStatus.FAILED:
                break

            c += 1
        print("")
        self.trained_param = optimizer_state.params

    def predict(self, x_test: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict outcome for each input data in ``x_test``. This method returns the predicted outcome as a vector of probabilities for each class.
        Args:
            x_test: Input data whose shape is ``(n_samples, n_features)``.
        Returns:
            y_pred: Predicted outcome whose shape is ``(n_samples, num_class)``.

        """
        if self.trained_param is None:
            raise ValueError("Model is not trained.")

        if x_test.ndim == 1:
            x_test = x_test.reshape(-1, 1)
        if self.do_x_scale:
            x_scaled = self.scale_x_scaler.transform(x_test)
        else:
            x_scaled = x_test
        y_pred = self._predict_inner(x_scaled, self.trained_param)  # .argmax(axis=1)
        return y_pred

    def _predict_inner(
        self, x_scaled: NDArray[np.float64], params: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return predict_inner(
            self.ansatz,
            self.estimator,
            self.operator,
            x_scaled,
            params,
            self.y_exp_ratio,
        )

    def cost_func(
        self,
        x_scaled: NDArray[np.float64],
        y_train: NDArray[np.int64],
        params: NDArray[np.float64],
    ) -> float:
        y_pred = self._predict_inner(x_scaled, params)
        # Case of log_logg
        # softmax
        y_pred_sm = self._softmax(y_pred, axis=1)
        loss = float(log_loss(y_train, y_pred_sm))
        # print(f"{params[:4]=}")
        return loss

    def cost_func_grad(
        self, x_scaled: NDArray[np.float64], y_train: NDArray[np.int64], params: Params
    ) -> NDArray[np.float64]:
        # start = time.perf_counter()
        y_pred = self._predict_inner(x_scaled, params)
        y_pred_sm = self._softmax(y_pred, axis=1)
        raw_grads = self._estimate_grad(x_scaled, params)
        # print(f"{raw_grads.shape=}")
        grads = np.zeros(self.ansatz.learning_params_count)
        # print(f"{grads.shape=}")
        # print(f"{raw_grads=}")
        for sample_index in range(len(x_scaled)):
            for current_class in range(self.num_class):
                expected = 1.0 if current_class == y_train[sample_index] else 0.0
                coef = self.y_exp_ratio * (-expected + y_pred_sm[sample_index][current_class])
                grads += coef * raw_grads[sample_index][current_class]
        grads /= len(x_scaled)
        # print(f"{time.perf_counter()-start=}")

        return grads

    def _estimate_grad(
        self, x_scaled: NDArray[np.float64], params: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        learning_param_indexes = self.ansatz.get_minimum_learning_param_indexes()
        return estimate_grad(
            self.ansatz,
            self.gradient_estimator,
            self.operator,
            x_scaled,
            params,
            learning_param_indexes,
            estimator=self.estimator,
        )
