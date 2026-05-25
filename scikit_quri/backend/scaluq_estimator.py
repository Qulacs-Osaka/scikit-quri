from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from quri_parts.circuit import ParametricQuantumCircuitProtocol
from quri_parts.core.estimator import Estimatable
from quri_parts.qulacs.estimator import create_qulacs_vector_concurrent_estimator

from quri_parts_scaluq import _backend
from quri_parts_scaluq.estimator import estimate as scaluq_estimate
from quri_parts_scaluq.estimator import estimate_numerical_gradient as scaluq_grad

from .base_estimator import BatchedSimEstimator


class ScaluqEstimator(BatchedSimEstimator):
    """Batched expectation-value estimator backed by scaluq.

    The primary API is ``estimate_batched`` / ``estimate_grad_batched``, which
    evaluate a single parametric circuit topology over many parameter vectors
    in one backend call. The non-batched ``estimate(operators, states)`` is
    implemented as a fallback that defers to qulacs; it is provided so that
    consumers expecting the ``BaseEstimator`` contract still work, but the
    batched methods are what give scaluq its speedup.
    """

    def __init__(self) -> None:
        self._concurrent_estimator = create_qulacs_vector_concurrent_estimator()

    def estimate(self, operators, states):
        return self._concurrent_estimator(operators, states)

    def estimate_batched(
        self,
        operators: Sequence[Estimatable],
        circuit: ParametricQuantumCircuitProtocol,
        params: NDArray[np.float64],
    ) -> list[list[float]]:
        state = _backend.StateVectorBatched(len(params), circuit.qubit_count)
        state.set_zero_state()
        return scaluq_estimate(state, circuit, operators, params)

    def estimate_grad_batched(
        self,
        operators: Sequence[Estimatable],
        circuit: ParametricQuantumCircuitProtocol,
        shifted_params: NDArray[np.float64],
        n_samples: int,
        n_learning_params: int,
        delta: float = 1e-5,
    ) -> NDArray[np.float64]:
        return scaluq_grad(circuit, operators, shifted_params, n_samples, n_learning_params, delta)
