"""Expectation-value estimator backends.

This file was disabled with ``pytestmark = pytest.mark.skip(reason="メンテナンス中の
ためスキップ")`` (2026-02-17). The assertions were ``estimate is not None``, which a
backend returning wrong numbers also satisfies, so they are compared against the
analytic value here.
"""

import pytest
from quri_parts.core import QuantumCircuit, quantum_state
from quri_parts.core.operator import pauli_label
from quri_parts.core.state import GeneralCircuitQuantumState

from scikit_quri.backend import QulacsEstimator, ScaluqEstimator, SimEstimator

# X on qubit 0, H on qubit 1, then CNOT(0, 1). Qubit 0 is flipped to |1>, and
# qubit 1 stays in an equal superposition (the CNOT only relabels its branches),
# so the exact values are:
EXPECTED = {"Z 0": -1.0, "Z 1": 0.0, "Z0 Z1": 0.0}


def create_simple_circuit() -> GeneralCircuitQuantumState:
    circuit = QuantumCircuit(2)
    circuit.add_X_gate(0)
    circuit.add_H_gate(1)
    circuit.add_CNOT_gate(0, 1)
    return quantum_state(n_qubits=2, circuit=circuit)


@pytest.mark.parametrize(
    "make_estimator",
    [QulacsEstimator, ScaluqEstimator, SimEstimator],
    ids=["qulacs", "scaluq", "sim_deprecated"],
)
def test_exact_estimators_agree_with_the_analytic_value(make_estimator) -> None:
    state = create_simple_circuit()
    labels = list(EXPECTED)
    estimate = list(make_estimator().estimate([pauli_label(v) for v in labels], [state]))
    assert len(estimate) == len(labels)
    for label, value in zip(labels, estimate):
        assert value.value.real == pytest.approx(EXPECTED[label], abs=1e-10), label


@pytest.mark.oqtopus
def test_oqtopus_estimator() -> None:
    """Needs an OQTOPUS account; deselect with ``-m "not oqtopus"``."""
    from scikit_quri.backend import OqtopusEstimator

    state = create_simple_circuit()
    # <Z 0> = -1 is deterministic, so sampling noise does not enter.
    estimate = list(OqtopusEstimator("qulacs", shots=10000).estimate([pauli_label("Z 0")], [state]))
    assert len(estimate) == 1
    assert estimate[0].value.real == pytest.approx(EXPECTED["Z 0"], abs=0.02)
