# mypy: ignore-errors
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from quri_parts.algo.optimizer import Adam
from quri_parts.core.estimator import (
    ConcurrentParametricQuantumEstimator,
    Estimatable,
    ConcurrentQuantumEstimator,
)
from quri_parts.algo.optimizer import OptimizerStatus
from quri_parts.core.state import quantum_state
from quri_parts.qulacs import QulacsParametricStateT, QulacsStateT
from scikit_quri.circuit import LearningCircuit
from typing import List

# ! Will remove
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# class mimMaxScaler:
#     def __init__(self,feature_range:tuple[int,int]=(0, 1)):
#         self.feature_range = feature_range

#     def fit(self,X:NDArray[np.float64]):
#         self.data_min = X.min(axis=0)
#         self.data_max = X.max(axis=0)

#     def transform(self,X:NDArray[np.float64]):
#         X_std = (X - self.data_min) / (self.data_max - self.data_min)
#         X_scaled = X_std * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
#         return X_scaled

#     def fit_transform(self,X:NDArray[np.float64]):
#         self.fit(X)
#         return self.transform(X)


@dataclass
class QNNRegressor:
    n_qubits: int
    ansatz: LearningCircuit
    estimator: ConcurrentQuantumEstimator[QulacsStateT]
    gradient_estimator: ConcurrentParametricQuantumEstimator[QulacsParametricStateT]
    optimizer: Adam
    operator: Estimatable

    x_norm_range: float = field(default=1.0)
    y_norm_range: float = field(default=0.7)

    do_x_scale: bool = field(default=True)
    do_y_scale: bool = field(default=True)
    n_outputs: int = field(default=1)
    y_exp_ratio: float = field(default=2.2)

    trained_param: Sequence[float] = field(default=None)

    def __post_init__(self) -> None:
        if self.do_x_scale:
            self.scale_x_scaler = MinMaxScaler(
                feature_range=(-self.x_norm_range, self.x_norm_range)
            )
        if self.do_y_scale:
            self.scale_y_scaler = MinMaxScaler(
                feature_range=(-self.y_norm_range, self.y_norm_range)
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

        self.x_train = x_scaled
        self.y_train = y_scaled

        parameter_count = self.ansatz.learning_params_count

        # set initial learning parameters
        init_params = 2 * np.pi * np.random.random(parameter_count)
        print(f"{init_params=}")
        optimizer_state = self.optimizer.get_init_state(init_params)

        c = 0
        while maxiter > c:
            cost_fn = lambda params: self.cost_fn(self.x_train, self.y_train, params)
            grad_fn = lambda params: self.grad_fn(self.x_train, self.y_train, params)
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
        self,
        x_scaled: NDArray[np.float64],
        y_scaled: NDArray[np.float64],
        params: Sequence[float],
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
        self,
        x_scaled: NDArray[np.float64],
        y_scaled: NDArray[np.float64],
        params: Sequence[Sequence[float]],
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

        grads = 2 * (y_pred - y_scaled) * y_pred_grads
        grads = np.mean(grads, axis=0)

        grads = np.asarray([g.real for g in grads])
        # print(f"{grads=}")

        return grads

    def _estimate_grad(
        self, x_scaled: NDArray[np.float64], params: Sequence[float]
    ) -> NDArray[np.float64]:
        """
        Estimate the gradient of the cost function.

        Parameters:
            x_scaled: Input data whose shape is (batch_size, n_features).
            params: Parameters for the quantum circuit.

        Returns:
            grads: Gradients of the cost function.
        """
        grads = []
        learning_params_indexes = self.ansatz.get_learning_param_indexes()
        for x in x_scaled:
            circuit_params = self.ansatz.generate_bound_params(x, params)
            circuit = quantum_state(n_qubits=self.n_qubits, circuit=self.ansatz.circuit)
            estimate = self.gradient_estimator(self.operator, circuit, circuit_params)
            # input用のparamsを取り除く
            grad = np.array(estimate.values)[learning_params_indexes]
            grads.append([g.real for g in grad])
        return np.asarray(grads)

    def _predict_inner(
        self, x_scaled: NDArray[np.float64], params: Sequence[float]
    ) -> NDArray[np.float64]:
        """
        Predict inner function.

        Parameters:
            x_scaled: Input data whose shape is (batch_size, n_features).
            params: Parameters for the quantum circuit.

        Returns:
            res: Predicted outcome.
        """
        res = []
        circuit_states: List[QulacsStateT] = []

        for x in x_scaled:
            circuit_params = self.ansatz.generate_bound_params(x, params)
            param_circuit_state = quantum_state(n_qubits=self.n_qubits, circuit=self.ansatz.circuit)
            circuit_state = param_circuit_state.bind_parameters(circuit_params)
            circuit_states.append(circuit_state)

        estimates = self.estimator(self.operator, circuit_states)
        res = [[e.value.real * self.y_exp_ratio] for e in estimates]

        return np.asarray(res)
