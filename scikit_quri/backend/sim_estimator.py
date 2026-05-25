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

    When ``use_scaluq=True`` the batched scaluq backend is used for
    ``predict_inner`` / ``estimate_grad``. The speedup over qulacs is driven by
    two factors verified in the benchmark report:

    1. **Python-C++ bridge overhead is amortized**: scaluq processes the entire
       batch in a single C++ call, instead of qulacs's per-sample loop that
       crosses the Python boundary thousands of times.
    2. **SIMD vectorization**: ``StateVectorBatched`` stores all sample states
       contiguously, so each gate is applied across the batch via SIMD.

    Kokkos thread parallelism and warmup cache effects were measured and shown
    to be irrelevant — speedup is ~22x at ``OMP_NUM_THREADS=1`` and identical
    in cold / warm / subprocess-isolated runs.

    Recommended ``batch_size`` (samples per ``estimate`` call) for scaluq path:

    - **256–512**: sweet spot, 15–22x speedup, fits in P-core L2 cache (16MB
      on Apple Silicon) for ``n_qubits ≤ 10``.
    - **>1024**: speedup plateaus or declines once the total state-vector
      memory (``batch_size × 2^n_qubits × 16 bytes``) exceeds L2 cache and
      the workload becomes memory-bandwidth bound. At ``n_qubits=8`` the peak
      is around batch=1536 (~6 MB); batch=4096 (16 MB) shows ~17% slowdown.
    - **<128**: Python overhead still dominates; speedup is only 2–8x. Use
      ``use_scaluq=False`` (plain qulacs) for very small batches.

    See ``BENCHMARK_REPORT.md`` for the full per-(n_qubits, batch) measurement
    grid and L2-overflow check.

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

        Performance: 256–512 samples per call is the sweet spot. The total
        state-vector memory (``len(params) × 2^circuit.qubit_count × 16 bytes``)
        should fit in L2 cache for best throughput — see :class:`SimEstimator`
        docstring and ``BENCHMARK_REPORT.md`` for the full analysis.

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

        Note: the effective scaluq batch is ``n_samples * 2 * n_learning_params``
        (each learning parameter produces a +/- shifted state per sample).
        For QNN training with n_samples=128 and ~30 learning parameters this is
        already ~7680 rows per call, which is far past L2 cache. Speedup is
        still ~2x over qulacs because qulacs's gradient estimator is per-sample
        Python-looped, but don't expect the 20x seen in ``estimate_scaluq_batched``.

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
