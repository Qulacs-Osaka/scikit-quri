from .base_estimator import BaseEstimator

from quri_parts.qulacs.estimator import create_qulacs_vector_concurrent_estimator
from quri_parts.circuit import ParametricQuantumCircuitProtocol
from quri_parts.core.estimator import Estimatable
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from quri_parts_scaluq.estimator import estimate as scaluq_estimate
from quri_parts_scaluq.estimator import estimate_numerical_gradient as scaluq_grad
from quri_parts_scaluq import _backend


class SimEstimator(BaseEstimator):
    """Simulation estimator that computes expectation values using quri-parts-qulacs.

    Args:
        use_scaluq: If True, use scaluq batched estimation for predict_inner.
            Defaults to False.
    """

    def __init__(self, use_scaluq: bool = False) -> None:
        self.use_scaluq = use_scaluq

    def estimate(self, operators, states):
        estimator = create_qulacs_vector_concurrent_estimator()
        return estimator(operators, states)

    def estimate_scaluq_batched(
        self,
        operators: Sequence[Estimatable],
        circuit: ParametricQuantumCircuitProtocol,
        params: NDArray[np.float64],
    ) -> list[list[float]]:
        """Compute batched expectation values using scaluq backend.

        Args:
            operators: List of measurement operators. Length: n_operators.
            circuit: Parametric quantum circuit (from LearningCircuit.to_batched).
            params: Batched parameters. Shape: (n_samples, n_params).

        Returns:
            List of shape (n_operators, n_samples) containing real expectation values.
        """
        n_qubits = circuit.qubit_count
        state = _backend.StateVectorBatched(len(params), n_qubits)
        state.set_zero_state()
        return scaluq_estimate(state, circuit, operators, params)

    def estimate_grad_scaluq_batched(
        self,
        operators: Sequence[Estimatable],
        circuit: ParametricQuantumCircuitProtocol,
        shifted_params: NDArray[np.float64],
        n_samples: int,
        n_learning_params: int,
        delta: float = 1e-5,
    ) -> NDArray[np.float64]:
        """Compute batched numerical gradient using scaluq backend.

        Args:
            operators: List of measurement operators. Length: n_operators.
            circuit: Parametric quantum circuit (from LearningCircuit.to_batched_for_gradient).
            shifted_params: Shifted parameter array.
                Shape: (n_samples * 2 * n_learning_params, parameter_count).
            n_samples: Number of input samples.
            n_learning_params: Number of learning parameters.
            delta: Finite difference step size.

        Returns:
            Gradient tensor. Shape: (n_samples, n_operators, n_learning_params).
        """
        return scaluq_grad(
            circuit,
            operators,
            shifted_params,
            n_samples,
            n_learning_params,
            delta,
        )
