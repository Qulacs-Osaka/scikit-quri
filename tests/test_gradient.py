import numpy as np
from typing import Sequence
from scikit_quri.backend import SimGradientEstimator
from scikit_quri.circuit.pre_defined import create_qcl_ansatz
from scikit_quri.circuit import LearningCircuit
from quri_parts.core.operator import pauli_label
from quri_parts.core.state import ParametricCircuitQuantumState


def create_simple_circuit() -> LearningCircuit:
    """Create a simple quantum circuit for testing."""
    circuit = create_qcl_ansatz(2, 1)
    return circuit


def test_sim_grad() -> None:
    """Test for SimGradient."""
    circuit = create_simple_circuit()
    params = circuit.generate_bound_params(
        np.array([0.0 for _ in range(circuit.input_params_count)]),
        np.array([0.0 for _ in range(circuit.learning_params_count)]),
    )
    state = ParametricCircuitQuantumState(2, circuit.circuit)
    estimator = SimGradientEstimator()
    estimate = estimator.estimate_gradient(pauli_label("Z0 Z1"), state, params)
    assert estimate is not None


def _test_sim_grad_learning_param() -> Sequence[complex]:
    """Test for SimGradient with learning parameters."""
    circuit = create_simple_circuit()
    params = circuit.generate_bound_params(
        np.array([0.0 for _ in range(circuit.input_params_count)]),
        np.array([0.0 for _ in range(circuit.learning_params_count)]),
    )
    estimator = SimGradientEstimator()
    estimate = estimator.estimate_learning_param_gradient(pauli_label("Z0 Z1"), circuit, params)
    # learning_params_count分の勾配が返ってくることを確認
    assert len(estimate) == circuit.learning_params_count
    return estimate


def _test_oqtopus_grad() -> Sequence[complex]:
    """Test for OqtopusGradient."""
    from scikit_quri.backend import OqtopusGradientEstimator

    circuit = create_simple_circuit()
    estimator = OqtopusGradientEstimator()
    estimate = estimator.estimate_learning_param_gradient(
        pauli_label("Z0 Z1"),
        circuit,
        circuit.generate_bound_params(
            np.array([0.0 for _ in range(circuit.input_params_count)]),
            np.array([0.0 for _ in range(circuit.learning_params_count)]),
        ),
    )
    assert len(estimate) == circuit.learning_params_count
    return estimate


def test_grad() -> None:
    """Compare gradients from SimGradient and OqtopusGradient."""
    sim_grad = _test_sim_grad_learning_param()
    oqtopus_grad = _test_oqtopus_grad()
    np.testing.assert_allclose(sim_grad, oqtopus_grad, atol=1e-5)
