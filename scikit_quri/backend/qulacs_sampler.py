from quri_parts.qulacs.sampler import create_qulacs_vector_concurrent_sampler

from .base_sampler import BaseSampler


class QulacsSampler(BaseSampler):
    """Computational-basis sampler backed by quri-parts-qulacs (state-vector simulator).

    Concurrent across the input pairs. This is the standard simulation sampler
    used for shot-based estimation, kernel overlap measurement, and generative
    modeling on a classical simulator.
    """

    def __init__(self) -> None:
        self._impl = create_qulacs_vector_concurrent_sampler()

    def sample(self, circuit_shots_tuples):
        return self._impl(circuit_shots_tuples)
