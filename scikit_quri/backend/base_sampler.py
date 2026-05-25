"""ABC for sampling backends.

Mirrors :class:`~scikit_quri.backend.BaseEstimator` so that the two execution
families (expectation values vs computational-basis sampling) share the same
class-based abstraction style.

A ``BaseSampler`` is also a quri-parts ``ConcurrentSampler`` (i.e. callable
with the same signature), so any existing API expecting a ``ConcurrentSampler``
accepts a ``BaseSampler`` instance unchanged. This is enforced by the
``__call__`` slot that delegates to :meth:`sample`.
"""

from abc import ABCMeta, abstractmethod
from typing import Iterable

from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.core.sampling import MeasurementCounts


class BaseSampler(metaclass=ABCMeta):
    """Concurrent sampling abstraction.

    The required method :meth:`sample` takes an iterable of
    ``(circuit, n_shots)`` pairs and returns one ``MeasurementCounts`` per
    pair in input order. Single-circuit sampling is offered as a thin
    convenience via :meth:`sample_one`.
    """

    @abstractmethod
    def sample(
        self,
        circuit_shots_tuples: Iterable[tuple[NonParametricQuantumCircuit, int]],
    ) -> Iterable[MeasurementCounts]:
        """Sample each ``(circuit, n_shots)`` pair and yield the resulting counts.

        Args:
            circuit_shots_tuples: Iterable of ``(circuit, n_shots)`` pairs.

        Returns:
            Iterable of measurement counts, one per input pair in the same order.
        """

    def sample_one(self, circuit: NonParametricQuantumCircuit, n_shots: int) -> MeasurementCounts:
        """Sample a single circuit ``n_shots`` times.

        Default implementation forwards to :meth:`sample` with a one-element list.
        """
        return next(iter(self.sample([(circuit, n_shots)])))

    def __call__(
        self,
        circuit_shots_tuples: Iterable[tuple[NonParametricQuantumCircuit, int]],
    ) -> Iterable[MeasurementCounts]:
        """``ConcurrentSampler`` protocol compatibility — delegates to :meth:`sample`."""
        return self.sample(circuit_shots_tuples)
