"""Gradient estimator backends.

Previously this whole file was disabled with
``pytestmark = pytest.mark.skip(reason="メンテナンス中のためスキップ")`` (2026-02-17),
and the two functions that did the real work were named with a leading underscore, so
pytest never collected them even without the skip. ``test_grad`` called them directly,
which also bypassed the ``oqtopus`` marker and made the only collected test reach out
to the cloud.
"""

import numpy as np
import pytest
from quri_parts.core.operator import Operator, pauli_label
from quri_parts.core.state import (
    GeneralCircuitQuantumState,
    ParametricCircuitQuantumState,
)
from quri_parts.qulacs.estimator import create_qulacs_vector_concurrent_estimator

from scikit_quri.backend import SimGradientEstimator
from scikit_quri.circuit import LearningCircuit
from scikit_quri.circuit.pre_defined import create_qcl_ansatz

ESTIMATOR = create_qulacs_vector_concurrent_estimator()
OPERATOR = Operator({pauli_label("Z0 Z1"): 1.0})


def create_simple_circuit() -> LearningCircuit:
    """Create a simple quantum circuit for testing."""
    return create_qcl_ansatz(2, 1)


def _inputs(circuit: LearningCircuit):
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, max(1, circuit.n_qubits))
    theta = rng.uniform(0, 2 * np.pi, circuit.learning_params_count)
    return x, theta


def _numerical_gradient(circuit, x, theta, operator, h=1e-5):
    def expectation(t):
        state = GeneralCircuitQuantumState(
            circuit.n_qubits, circuit.bind_input_and_parameters(x, t)
        )
        return ESTIMATOR([operator], [state])[0].value.real

    grad = np.zeros(len(theta))
    for j in range(len(theta)):
        plus, minus = theta.copy(), theta.copy()
        plus[j] += h
        minus[j] -= h
        grad[j] = (expectation(plus) - expectation(minus)) / (2 * h)
    return grad


def test_sim_grad() -> None:
    """Per-gate gradient of the underlying quri-parts estimator.

    ``estimate_gradient`` differentiates w.r.t. the gate parameters, so it has one
    entry per parametric slot rather than per learning parameter.
    """
    circuit = create_simple_circuit()
    x, theta = _inputs(circuit)
    params = circuit.generate_bound_params(x, theta)
    state = ParametricCircuitQuantumState(circuit.n_qubits, circuit.circuit)

    estimate = SimGradientEstimator().estimate_gradient(OPERATOR, state, params)

    values = np.asarray(estimate.values)
    assert len(values) == circuit.parameter_count
    assert np.all(np.isfinite(values))


@pytest.mark.parametrize("method", ["parameter_shift", "numerical"])
def test_sim_grad_learning_param(method: str) -> None:
    """Gradient w.r.t. the learning parameters must match finite differences.

    The old version only checked the length. That passed even while the values were
    wrong: ``share_with`` coefficients were dropped and shared gate positions were
    never summed.
    """
    circuit = create_simple_circuit()
    x, theta = _inputs(circuit)
    params = circuit.generate_bound_params(x, theta)

    estimate = SimGradientEstimator(method=method).estimate_learning_param_gradient(
        OPERATOR, circuit, params, x=x, theta=theta
    )

    assert len(estimate) == circuit.learning_params_count
    np.testing.assert_allclose(
        np.real(np.asarray(estimate, dtype=np.complex128)),
        _numerical_gradient(circuit, x, theta, OPERATOR),
        atol=1e-4,
    )


@pytest.mark.oqtopus
def test_oqtopus_grad() -> None:
    """Same gradient computed on OQTOPUS Cloud (needs an account; deselect with
    ``-m "not oqtopus"``)."""
    from scikit_quri.backend import OqtopusGradientEstimator

    circuit = create_simple_circuit()
    x, theta = _inputs(circuit)
    params = circuit.generate_bound_params(x, theta)

    estimate = OqtopusGradientEstimator(
        device_id="qulacs", shots=10000
    ).estimate_learning_param_gradient(OPERATOR, circuit, params, x=x, theta=theta)

    assert len(estimate) == circuit.learning_params_count
    # 10000 shots: each component carries a sampling error of order 0.01, and a
    # learning parameter may sum several gate positions, so keep the tolerance loose.
    np.testing.assert_allclose(
        np.real(np.asarray(estimate, dtype=np.complex128)),
        _numerical_gradient(circuit, x, theta, OPERATOR),
        atol=0.05,
    )
