from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from quri_parts.algo.optimizer import Adam
from quri_parts.circuit.circuit_parametric import UnboundParametricQuantumCircuitProtocol
from quri_parts.core.estimator import (
    ConcurrentParametricQuantumEstimator,
    Estimatable,
    ParametricQuantumEstimator,
)
from quri_parts.core.state import ParametricCircuitQuantumState
from quri_parts.qulacs import QulacsParametricStateT


@dataclass
class QNNRegressor:
    n_qubits: int
    ansatz: UnboundParametricQuantumCircuitProtocol
    estimator: ParametricQuantumEstimator[QulacsParametricStateT]
    gradient_estimator: ConcurrentParametricQuantumEstimator[QulacsParametricStateT]
    optimizer: Adam
    operator: Estimatable

    def fit(self, x_train: NDArray[np.float_], y_train: NDArray[np.float_]) -> None:
        init_params = np.random.random(self.ansatz.parameter_count)
        optimizer_state = self.optimizer.get_init_state(init_params)
        print(optimizer_state)

        while True:
            optimizer_state = self.optimizer.step(
                optimizer_state,
                self.cost_fn,
                self.grad_fn,
            )
            break

            if optimizer_state.status == "CONVERGED":
                break

            if optimizer_state.status == "FAILED":
                break

    def run(self, x_train: NDArray[np.float_]) -> NDArray[np.float_]:
        # self.ansatz += 1
        pass

    def cost_fn(self, params: Sequence[float]) -> float:
        print(f"{params=}")
        # * init circuit state to |00..0>
        circuit_state = ParametricCircuitQuantumState(self.n_qubits, self.ansatz)
        estimate = self.estimator(
            self.operator,
            circuit_state,
            params,
        )
        return estimate.value.real

    def grad_fn(self, param_values: Sequence[Sequence[float]]) -> np.ndarray:
        circuit_state = ParametricCircuitQuantumState(self.n_qubits, self.ansatz)
        estimate = self.gradient_estimator(
            self.operator,
            circuit_state,
            param_values,
        )
        return np.asarray([g.real for g in estimate.values])


