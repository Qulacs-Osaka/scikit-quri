from quri_parts.qulacs.estimator import create_qulacs_vector_concurrent_estimator

from .base_estimator import BaseEstimator


class QulacsEstimator(BaseEstimator):
    """Expectation-value estimator backed by quri-parts-qulacs.

    Evaluates one operator against one quantum state per call (per-sample
    semantics, concurrent across the inputs of a single ``estimate`` call).
    For workloads that re-bind the same parametric circuit across many samples,
    consider ``ScaluqEstimator`` which natively batches that case.
    """

    def __init__(self) -> None:
        self._concurrent_estimator = create_qulacs_vector_concurrent_estimator()

    def estimate(self, operators, states):
        return self._concurrent_estimator(operators, states)
