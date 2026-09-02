"""Numerical correctness of the gradient paths.

Every test here compares an analytic/estimated gradient against a finite-difference
gradient of the same expectation value. These are the invariants that were silently
violated before: unregistered gate slots, dropped ``share_with`` coefficients, the
missing ``df/dtheta`` chain rule, and non-self-inverse gates in the Hadamard test.
"""

import numpy as np
import pytest
from quri_parts.circuit import gates as qpg
from quri_parts.core.operator import Operator, pauli_label
from quri_parts.core.state import GeneralCircuitQuantumState
from quri_parts.qulacs.estimator import (
    create_qulacs_vector_concurrent_estimator,
    create_qulacs_vector_concurrent_parametric_estimator,
)
from quri_parts.core.estimator.gradient import create_numerical_gradient_estimator

from scikit_quri.circuit import LearningCircuit
from scikit_quri.circuit.gradient import hadamard_gradient
from scikit_quri.circuit.pre_defined import create_qcnn_ansatz
from scikit_quri.qnn._qnn_common import estimate_grad

ESTIMATOR = create_qulacs_vector_concurrent_estimator()


def _expval(circuit, x, theta, op):
    bound = circuit.bind_input_and_parameters(x, theta)
    state = GeneralCircuitQuantumState(circuit.n_qubits, bound)
    return ESTIMATOR([op], [state])[0].value.real


def _numerical_gradient(circuit, x, theta, op, h=1e-5):
    grad = np.zeros(len(theta))
    for j in range(len(theta)):
        plus, minus = theta.copy(), theta.copy()
        plus[j] += h
        minus[j] -= h
        grad[j] = (_expval(circuit, x, plus, op) - _expval(circuit, x, minus, op)) / (2 * h)
    return grad


def test_every_parametric_slot_is_registered():
    """Each parametric gate slot must belong to some registered parameter.

    An unregistered slot is bound to 0.0 forever and never receives a gradient, so
    the gate silently becomes the identity.
    """
    circuit = create_qcnn_ansatz(4, seed=0)
    registered = {
        pos.gate_pos
        for lp in circuit._registry.learning_parameters
        for pos in lp.positions_in_circuit
    }
    registered |= {ip.gate_pos for ip in circuit._registry.input_parameters}
    assert registered == set(range(circuit.parameter_count))


def test_qcnn_multi_pauli_angles_follow_theta():
    circuit = create_qcnn_ansatz(4, seed=0)
    x = np.array([0.1, 0.2, 0.3, 0.4])
    theta = np.full(circuit.learning_params_count, 0.3)
    bound_a = np.array(circuit.generate_bound_params(x, theta))
    bound_b = np.array(circuit.generate_bound_params(x, theta + 0.7))
    # No slot may be pinned at zero regardless of theta.
    assert np.count_nonzero(bound_a) > 0
    assert np.max(np.abs(bound_a - bound_b)) > 1e-9


@pytest.mark.parametrize("fixed_gate", [None, "S", "T", "SqrtX"])
def test_hadamard_gradient_with_non_self_inverse_gates(fixed_gate):
    """U†_{>j} must be a real inverse; negating params leaves S/T/SqrtX unchanged."""
    circuit = LearningCircuit(2)
    circuit.add_H_gate(0)
    circuit.add_H_gate(1)
    circuit.add_parametric_RX_gate(0)
    circuit.add_parametric_RY_gate(1)
    circuit.add_CNOT_gate(0, 1)
    if fixed_gate == "S":
        circuit.add_gate(qpg.S(1))
    elif fixed_gate == "T":
        circuit.add_gate(qpg.T(0))
    elif fixed_gate == "SqrtX":
        circuit.add_gate(qpg.SqrtX(1))
    circuit.add_parametric_RZ_gate(0)
    circuit.add_CNOT_gate(1, 0)
    circuit.add_parametric_RY_gate(1)

    op = Operator({pauli_label("Z0"): 1.0, pauli_label("Y1"): 0.7, pauli_label("X0 X1"): 0.3})
    theta = np.array([0.31, 0.72, 1.13, 0.44])
    x = np.array([0.0])

    expected = _numerical_gradient(circuit, x, theta, op)
    got = np.real(hadamard_gradient(circuit, x, theta, op, ESTIMATOR))
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_hadamard_gradient_aggregates_shared_parameters():
    """share_with must sum positions and apply the coefficient."""
    circuit = LearningCircuit(2)
    circuit.add_H_gate(0)
    circuit.add_H_gate(1)
    pid = circuit.add_parametric_RX_gate(0)
    circuit.add_parametric_RY_gate(1, share_with=pid, share_with_coef=-1.0)
    circuit.add_CNOT_gate(0, 1)
    circuit.add_parametric_RZ_gate(1)

    op = Operator({pauli_label("Z0"): 1.0, pauli_label("X1"): 0.5})
    theta = np.array([0.63, 0.29])
    x = np.array([0.0])

    expected = _numerical_gradient(circuit, x, theta, op)
    got = np.real(hadamard_gradient(circuit, x, theta, op, ESTIMATOR))
    assert len(got) == circuit.learning_params_count
    np.testing.assert_allclose(got, expected, atol=1e-6)


def _chain_rule_circuit():
    circuit = LearningCircuit(2)
    circuit.add_H_gate(0)
    circuit.add_H_gate(1)
    circuit.add_parametric_RX_gate(0)
    circuit.add_parametric_input_RY_gate(1, lambda theta, x: theta * x[0])
    circuit.add_CNOT_gate(0, 1)
    circuit.add_parametric_RZ_gate(1)
    return circuit


def test_chain_rule_hadamard():
    """angle = f(theta, x) needs the df/dtheta factor (here exactly x[0])."""
    circuit = _chain_rule_circuit()
    op = Operator({pauli_label("Z0"): 1.0, pauli_label("X1"): 0.4})
    theta = np.array([0.41, 0.55, 0.27])
    x = np.array([3.0])

    expected = _numerical_gradient(circuit, x, theta, op)
    got = np.real(hadamard_gradient(circuit, x, theta, op, ESTIMATOR))
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_chain_rule_estimate_grad():
    circuit = _chain_rule_circuit()
    op = Operator({pauli_label("Z0"): 1.0, pauli_label("X1"): 0.4})
    theta = np.array([0.41, 0.55, 0.27])
    x = np.array([3.0])

    gradient_estimator = create_numerical_gradient_estimator(
        create_qulacs_vector_concurrent_parametric_estimator(), delta=1e-6
    )
    expected = _numerical_gradient(circuit, x, theta, op)
    got = np.real(estimate_grad(circuit, gradient_estimator, [op], x.reshape(1, -1), theta)[0][0])
    np.testing.assert_allclose(got, expected, atol=1e-4)


def test_registering_input_param_invalidates_template():
    """Binding, then adding a gate, must not reuse the stale template size."""
    circuit = LearningCircuit(1)
    circuit.add_parametric_RX_gate(0)
    theta = np.array([1.0])
    assert list(circuit.generate_bound_params(np.array([0.5]), theta)) == [1.0]
    circuit.add_input_RZ_gate(0, lambda x: x[0])
    bound = circuit.generate_bound_params(np.array([0.5]), theta)
    assert len(bound) == circuit.parameter_count == 2


def test_regressor_grad_matches_cost_gradient():
    """grad_fn must be the gradient of cost_fn (y_exp_ratio applied to both)."""
    from quri_parts.algo.optimizer import Adam

    from scikit_quri.backend import QulacsEstimator
    from scikit_quri.circuit.pre_defined import create_qcl_ansatz
    from scikit_quri.qnn.regressor import QNNRegressor

    rng = np.random.default_rng(0)
    circuit = create_qcl_ansatz(3, 2, seed=0)
    gradient_estimator = create_numerical_gradient_estimator(
        create_qulacs_vector_concurrent_parametric_estimator(), delta=1e-6
    )
    model = QNNRegressor(circuit, QulacsEstimator(), gradient_estimator, Adam())
    x = rng.uniform(-1, 1, (6, 1))
    y = np.sin(x[:, 0]).reshape(-1, 1)
    model.fit(x, y, maxiter=1)

    theta = model.trained_param.copy()
    analytic = model.grad_fn(model.x_train, model.y_train, theta)
    h = 1e-5
    numeric = np.zeros_like(theta)
    for j in range(len(theta)):
        plus, minus = theta.copy(), theta.copy()
        plus[j] += h
        minus[j] -= h
        numeric[j] = (
            model.cost_fn(model.x_train, model.y_train, plus)
            - model.cost_fn(model.x_train, model.y_train, minus)
        ) / (2 * h)

    scale = max(1e-12, np.max(np.abs(numeric)))
    assert np.max(np.abs(analytic - numeric)) / scale < 1e-4


def test_adjoint_is_not_used_on_hardware_backends():
    """The adjoint method is simulator-only and must never run for a device backend.

    It needs ``|psi>`` and ``O|psi>`` as vectors and replays the circuit backwards;
    on hardware measurement collapses the state. Taking that path for a device would
    quietly compute the gradient on a local simulator instead of the device.
    """
    from scikit_quri.backend import (
        ExactStatevectorEstimator,
        OqtopusEstimator,
        QulacsEstimator,
        ScaluqEstimator,
    )

    assert issubclass(QulacsEstimator, ExactStatevectorEstimator)
    assert issubclass(ScaluqEstimator, ExactStatevectorEstimator)
    assert not issubclass(OqtopusEstimator, ExactStatevectorEstimator)


def test_estimate_grad_falls_back_when_backend_is_not_exact():
    """With a non-exact backend, estimate_grad must use the supplied gradient_estimator."""
    from scikit_quri.backend import BaseEstimator

    class _DeviceLikeEstimator(BaseEstimator):
        def estimate(self, operators, states):  # pragma: no cover - not exercised here
            raise AssertionError("should not be called by the gradient path")

    circuit = _chain_rule_circuit()
    op = Operator({pauli_label("Z0"): 1.0})
    theta = np.array([0.41, 0.55, 0.27])
    x = np.array([3.0]).reshape(1, -1)

    calls = []
    base = create_numerical_gradient_estimator(
        create_qulacs_vector_concurrent_parametric_estimator(), delta=1e-6
    )

    def counting_gradient_estimator(*args, **kwargs):
        calls.append(1)
        return base(*args, **kwargs)

    estimate_grad(
        circuit,
        counting_gradient_estimator,
        [op],
        x,
        theta,
        estimator=_DeviceLikeEstimator(),
        use_adjoint=True,
    )
    assert calls, "gradient_estimator must be used when the backend is not exact"
