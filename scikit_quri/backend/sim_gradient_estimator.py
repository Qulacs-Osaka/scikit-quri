from typing import Literal, Sequence, get_args

import numpy as np
from quri_parts.core.estimator import Estimatable, Estimates
from quri_parts.core.estimator.gradient import (
    _ParametricStateT,
    create_numerical_gradient_estimator,
    create_parameter_shift_gradient_estimator,
)
from quri_parts.core.state import ParametricCircuitQuantumState
from quri_parts.qulacs.estimator import create_qulacs_vector_concurrent_parametric_estimator

from scikit_quri.circuit.circuit import LearningCircuit

from .base_gradient_estimator import BaseGradientEstimator

METHOD = Literal["numerical", "parameter_shift"]


class SimGradientEstimator(BaseGradientEstimator):
    """quri-parts-qulacsを用いて勾配を計算するSimulation用Gradient Estimator Class

    Args:
        method: 勾配計算に用いる手法。 "numerical"または"parameter_shift"を指定可能。デフォルトは"parameter_shift"
        delta: 数値微分を行う際の差分。methodが"numerical"の場合にのみ使用される。デフォルトは1e-5

    Raises:
        ValueError: 不正なmethod名が指定された場合に発生
    """

    def __init__(self, method: METHOD = "parameter_shift", delta: float = 1e-5) -> None:
        if method not in get_args(METHOD):
            raise ValueError(f"Invalid method: {method}. Supported methods are {get_args(METHOD)}")
        self.method = method
        self.delta = delta
        if method == "numerical":
            self.estimator = create_numerical_gradient_estimator(
                create_qulacs_vector_concurrent_parametric_estimator(), delta=self.delta
            )
        else:
            self.estimator = create_parameter_shift_gradient_estimator(
                create_qulacs_vector_concurrent_parametric_estimator()
            )

    def estimate_gradient(
        self, operators: Estimatable, state: _ParametricStateT, params: Sequence[float]
    ) -> Estimates[complex]:
        return self.estimator(operators, state, params)

    def estimate_learning_param_gradient(
        self, operators: Estimatable, circuit: LearningCircuit, params: Sequence[float]
    ) -> Sequence[complex]:
        """学習パラメータに対する勾配を計算する"""
        state = ParametricCircuitQuantumState(circuit.n_qubits, circuit.circuit)
        estimate_gradients = self.estimate_gradient(operators, state, params)
        return np.array(estimate_gradients.values)[circuit.get_learning_params_indexes()].tolist()
