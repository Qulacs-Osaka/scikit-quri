# mypy: ignore-errors
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray
from quri_parts.algo.optimizer import Optimizer, OptimizerStatus, Params
from quri_parts.core.estimator import Estimatable
from quri_parts.core.operator import Operator, pauli_label
from sklearn.preprocessing import MinMaxScaler

from scikit_quri.backend import BaseEstimator
from scikit_quri.circuit import LearningCircuit

from ._qnn_common import (
    GradientEstimatorType,
    predict_inner_cached,
    estimate_grad,
)


def mean_squared_error(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """
    Calculate the mean squared error between true and predicted values.

    Parameters:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        mse: Mean squared error.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")
    return float(np.mean((y_true - y_pred) ** 2))


@dataclass
class QNNRegressor:
    """
    Class to solve regression problems with quantum neural networks.
    The out is taken as expectation values of ``Pauli Z`` operators acting on the first qubit. i.e., output is ``<Z_0>``.

    Args:
        ansatz: Circuit to use in the learning.
        estimator: Estimator to use. use :py:func:`~quri_parts.qulacs.estimator.create_qulacs_vector_concurrent_estimator` method.
        gradient_estimator: Gradient estimator to use. use :py:func:`~quri_parts.core.estimator.gradient.create_parameter_shift_gradient_estimator` or :py:func:`~quri_parts.core.estimator.gradient.create_parameter_shift_gradient_estimator` method.
        optimizer: Optimizer to use. use :py:class:`~quri_parts.algo.optimizer.Adam` or :py:class:`~quri_parts.algo.optimizer.LBFGS` method.

    Example:
        >>> from quri_parts.qulacs.estimator import (
        >>>     create_qulacs_vector_concurrent_estimator,
        >>>     create_qulacs_vector_concurrent_parametric_estimator,
        >>> )
        >>> from quri_parts.core.estimator.gradient import (
        >>>     create_numerical_gradient_estimator,
        >>> )
        >>> n_qubit = 3
        >>> depth = 3
        >>> time_step = 0.5
        >>> estimator = create_qulacs_vector_concurrent_estimator()
        >>> gradient_estimator = create_numerical_gradient_estimator(
        >>>     create_qulacs_vector_concurrent_parametric_estimator()
        >>> )
        >>> circuit = create_qcl_ansatz(n_qubit, depth, time_step, 0)
        >>> circuit = create_qcl_ansatz(n_qubit, depth, time_step, 0)
        >>> qnn = QNNRegressor(n_qubit, circuit, estimator, gradient_estimator, solver)
        >>> qnn.fit(x_train, y_train, maxiter)
        >>> y_pred = qnn.predict(x_test)
    """

    ansatz: LearningCircuit
    estimator: BaseEstimator
    gradient_estimator: GradientEstimatorType
    optimizer: Optimizer

    operator: List[Estimatable] = field(default_factory=list)

    x_norm_range: float = field(default=1.0)
    y_norm_range: float = field(default=0.7)

    n_qubit: int = field(init=False)
    do_x_scale: bool = field(default=True)
    do_y_scale: bool = field(default=True)
    n_outputs: int = field(default=1)
    y_exp_ratio: float = field(default=2.2)

    trained_param: Optional[Params] = field(default=None)

    _pred_cache: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not issubclass(type(self.estimator), BaseEstimator):
            raise TypeError("estimator must be a subclass of BaseEstimator")
        self.n_qubit = self.ansatz.n_qubits
        if self.do_x_scale:
            self.scale_x_scaler = MinMaxScaler(
                feature_range=(-self.x_norm_range, self.x_norm_range)  # type: ignore
            )
        if self.do_y_scale:
            self.scale_y_scaler = MinMaxScaler(
                feature_range=(-self.y_norm_range, self.y_norm_range)  # type: ignore
            )

    def fit(self, x_train: NDArray[np.float64], y_train: NDArray[np.float64], maxiter=20) -> None:
        """
        Fit the model to the training data.

        Parameters:
            x_train: Input data whose shape is (n_samples, n_features).
            y_train: Output data whose shape is (n_samples, n_outputs).
            batch_size: The number of samples in each batch.

        """
        if x_train.ndim == 1:
            x_train = x_train.reshape((-1, 1))

        if y_train.ndim == 1:
            y_train = y_train.reshape((-1, 1))

        if self.do_x_scale:
            x_scaled = self.scale_x_scaler.fit_transform(x_train)
        else:
            x_scaled = x_train

        if self.do_y_scale:
            y_scaled = self.scale_y_scaler.fit_transform(y_train)
        else:
            y_scaled = y_train

        self.n_outputs = y_scaled.shape[1]
        # operator設定
        operators = []
        for i in range(self.n_outputs):
            operators.append(Operator({pauli_label(f"Z {i}"): 1.0}))
        self.operator = operators

        self.x_train = x_scaled
        self.y_train = y_scaled

        parameter_count = self.ansatz.learning_params_count

        # set initial learning parameters
        init_params = 2 * np.pi * np.random.random(parameter_count)
        optimizer_state = self.optimizer.get_init_state(init_params)

        c = 0
        while maxiter > c:

            def cost_fn(params):
                return self.cost_fn(self.x_train, self.y_train, params)

            def grad_fn(params):
                return self.grad_fn(self.x_train, self.y_train, params)

            optimizer_state = self.optimizer.step(optimizer_state, cost_fn, grad_fn)
            print("\r", f"iter:{c}/{maxiter} cost:{optimizer_state.cost}", end="")
            # print(f"{optimizer_state.cost=}")
            # break
            if optimizer_state.status == OptimizerStatus.CONVERGED:
                break
            if optimizer_state.status == OptimizerStatus.FAILED:
                break
            c += 1
        print("")

        self.trained_param = optimizer_state.params
        print(f"{optimizer_state.cost=}")

    def cost_fn(
        self, x_scaled: NDArray[np.float64], y_scaled: NDArray[np.float64], params: Params
    ) -> float:
        """
        Calculate the cost function for solver.

        Parameters:
            x_batched: Input data whose shape is (batch_size, n_features).
            y_batched: Output data whose shape is (batch_size, n_outputs).
            params: Parameters for the quantum circuit.

        Returns:
            cost: Cost function value.
        """
        y_pred = self._predict_inner(x_scaled, params)
        # Case of MSE
        cost = mean_squared_error(y_scaled, y_pred)
        return cost

    def predict(self, x_test: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict outcome for each input data in `x_test`.

        Arguments:
            x_test: Input data whose shape is (batch_size, n_features).

        Returns:
            y_pred: Predicted outcome.
        """
        if self.trained_param is None:
            raise ValueError("Model is not trained yet.")

        if x_test.ndim == 1:
            x_test = x_test.reshape((-1, 1))

        if self.do_x_scale:
            x_scaled = self.scale_x_scaler.transform(x_test)
        else:
            x_scaled = x_test

        if self.do_y_scale:
            y_pred: NDArray[np.float64] = self.scale_y_scaler.inverse_transform(
                self._predict_inner(x_scaled, self.trained_param)
            )
        else:
            y_pred = self._predict_inner(x_scaled, self.trained_param)

        return y_pred

    def grad_fn(
        self, x_scaled: NDArray[np.float64], y_scaled: NDArray[np.float64], params: Params
    ) -> NDArray[np.float64]:
        """
        Calculate the gradient of the cost function for solver.

        Parameters:
            x_batched: Input data whose shape is (batch_size, n_features).
            y_batched: Output data whose shape is (batch_size, n_outputs).
            params: Parameters for the quantum circuit.

        Returns:
            grads: Gradient of the cost function.
        """

        # for MSE
        y_pred = self._predict_inner(x_scaled, params)
        y_pred_grads = self._estimate_grad(x_scaled, params)
        diff = y_pred - y_scaled
        n_samples = len(diff)
        # grads[p] = (1/N) * sum_s (1/n_outputs) * sum_o 2 * diff[s,o] * y_pred_grads[s,o,p]
        grads = (2.0 / (n_samples * self.n_outputs)) * np.einsum("so,sop->p", diff, y_pred_grads)
        return grads

    def _estimate_grad(self, x_scaled: NDArray[np.float64], params: Params) -> NDArray[np.float64]:
        return estimate_grad(
            self.ansatz,
            self.gradient_estimator,
            self.operator,
            x_scaled,
            params,
            estimator=self.estimator,
        )

    def _predict_inner(self, x_scaled: NDArray[np.float64], params: Params) -> NDArray[np.float64]:
        return predict_inner_cached(
            self.ansatz,
            self.estimator,
            self.operator,
            x_scaled,
            params,
            self.y_exp_ratio,
            self._pred_cache,
        )
