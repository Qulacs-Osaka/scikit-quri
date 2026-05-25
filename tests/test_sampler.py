"""Tests for the sampler ABC and the QulacsSampler implementation."""

import pytest
from quri_parts.circuit import QuantumCircuit

from scikit_quri.backend import BaseSampler, OqtopusSampler, QulacsSampler


def _bell_circuit() -> QuantumCircuit:
    """|Φ+⟩ on 2 qubits — measurements should yield only 0b00 and 0b11."""
    circuit = QuantumCircuit(2)
    circuit.add_H_gate(0)
    circuit.add_CNOT_gate(0, 1)
    return circuit


def test_qulacs_sampler_is_base_sampler() -> None:
    assert isinstance(QulacsSampler(), BaseSampler)


def test_qulacs_sampler_concurrent_protocol() -> None:
    """A BaseSampler instance is also a ConcurrentSampler-callable (via __call__)."""
    sampler = QulacsSampler()
    # Doesn't actually verify the runtime_checkable protocol — quri-parts ships
    # ConcurrentSampler as a typing alias rather than a Protocol — but assert it's
    # callable, which is what consumers depend on.
    assert callable(sampler)


def test_sample_returns_one_counts_per_pair() -> None:
    sampler = QulacsSampler()
    circuit = _bell_circuit()
    results = list(sampler.sample([(circuit, 100), (circuit, 200)]))
    assert len(results) == 2
    assert sum(results[0].values()) == 100
    assert sum(results[1].values()) == 200


def test_sample_only_correlated_outcomes() -> None:
    """Bell state measured in Z basis only ever yields 00 or 11."""
    sampler = QulacsSampler()
    counts = sampler.sample_one(_bell_circuit(), 1000)
    # MeasurementCounts is keyed by int representing the bit string
    for bitstring in counts.keys():
        assert bitstring in (0b00, 0b11), f"unexpected outcome {bin(bitstring)}"


def test_call_delegates_to_sample() -> None:
    """Invoking the sampler as a callable must give the same result as .sample()."""
    sampler = QulacsSampler()
    circuit = _bell_circuit()
    counts_via_call = list(sampler([(circuit, 50)]))
    counts_via_method = list(sampler.sample([(circuit, 50)]))
    assert len(counts_via_call) == len(counts_via_method) == 1
    assert sum(counts_via_call[0].values()) == 50


def test_sample_one_default_implementation() -> None:
    """Default sample_one uses sample under the hood — verify it returns one MeasurementCounts."""
    sampler = QulacsSampler()
    counts = sampler.sample_one(_bell_circuit(), 50)
    assert sum(counts.values()) == 50


def test_oqtopus_sampler_inherits_base_sampler() -> None:
    """OqtopusSampler inherits BaseSampler. Doesn't instantiate to avoid auth."""
    assert issubclass(OqtopusSampler, BaseSampler)


def test_oqtopus_concurrent_sample_deprecated() -> None:
    """The old name is kept as a deprecated alias."""
    # Skip if quri_parts_oqtopus needs auth; instantiation alone is cheap
    try:
        sampler = OqtopusSampler(device_id="qulacs", config=None)
    except Exception:
        pytest.skip("OqtopusSampler construction requires backend auth")
    # Calling concurrent_sample should emit DeprecationWarning, even without making a real call
    # We only check the attribute exists and is callable — don't invoke (would hit backend).
    assert hasattr(sampler, "concurrent_sample")
    assert callable(sampler.concurrent_sample)
