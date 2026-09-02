from typing import Literal, Optional, Sequence, get_args

import numpy as np
from numpy.typing import NDArray
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
        self,
        operators: Estimatable,
        circuit: LearningCircuit,
        params: Sequence[float],
        x: Optional[NDArray[np.float64]] = None,
        theta: Optional[NDArray[np.float64]] = None,
    ) -> Sequence[complex]:
        """Gradient w.r.t. the learning parameters (length ``learning_params_count``).

        ``params`` is the *gate-level* bound parameter vector, so the raw estimate is
        ``d<O>/d(angle)`` per gate position. It is aggregated through the circuit's
        registry, which applies ``share_with`` coefficients and sums shared positions.
        Selecting the learning positions directly (as this used to do) returns one
        entry per gate position instead of per learning parameter — a different length
        whenever ``share_with`` is used, with the coefficients dropped.

        Args:
            operators: Observable.
            circuit: The learning circuit.
            params: Gate-level bound parameters (from ``generate_bound_params``).
            x: Input data for the sample. Required when the circuit has
                parametric-input gates, to apply the ``df/dtheta`` chain factor.
            theta: Learning-parameter vector, required together with ``x``.
        """
        state = ParametricCircuitQuantumState(circuit.n_qubits, circuit.circuit)
        estimate_gradients = self.estimate_gradient(operators, state, params)
        gate_gradients = np.real(np.asarray(estimate_gradients.values, dtype=np.complex128))

        chain_factors = None
        if x is not None and theta is not None:
            chain_factors = circuit.input_chain_factors(x, theta)
        return circuit.aggregate_gate_gradients(
            gate_gradients,
            skip_is_input=chain_factors is None,
            chain_factors=chain_factors,
        ).tolist()
