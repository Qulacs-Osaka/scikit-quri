import numpy as np
import pytest

from scikit_quri.circuit import LearningCircuit


def test_parametric_gate() -> None:
    circuit = LearningCircuit(2)
    circuit.add_input_RX_gate(1, lambda x: x[0])
    circuit.add_RX_gate(0, 0.5)
    idx = circuit.add_parametric_RX_gate(0)
    circuit.add_parametric_RX_gate(1, idx)
    params = circuit.generate_bound_params(np.array([0.1]), np.array([0.2]))
    assert np.array_equal(params, np.array([0.1, 0.2, 0.2]))


def test_parametric_input_gate() -> None:
    circuit = LearningCircuit(2)
    circuit.add_input_RX_gate(1, lambda x: 0.5 + x[0])
    params = circuit.generate_bound_params(np.array([1.0]), np.array([]))
    assert np.array_equal(params, np.array([1.5]))


def test_parametric_gates_mixed() -> None:
    circuit = LearningCircuit(2)
    circuit.add_parametric_RX_gate(0)
    circuit.add_parametric_input_RX_gate(1, lambda theta, x: theta + x[0])
    circuit.add_input_RX_gate(0, lambda x: 0)
    params = circuit.generate_bound_params(np.array([1.0]), np.array([0.1, 0.5, 0]))
    assert np.array_equal(params, np.array([0.1, 1.5, 0]))
    params = circuit.generate_bound_params(np.array([1.0]), np.array([0.2, 1.0, 0]))
    assert np.array_equal(params, np.array([0.2, 2.0, 0]))


def test_share_learning_parameter() -> None:
    circuit = LearningCircuit(2)
    idx = circuit.add_parametric_RX_gate(0)
    circuit.add_parametric_RY_gate(1, share_with=idx)  # Compute RY gate with shared parameter 0.
    params = circuit.generate_bound_params(np.array([]), np.array([0.1]))
    assert np.array_equal(params, np.array([0.1, 0.1]))  # 2 parameters are shared.


def test_running_shared_parameter() -> None:
    circuit = LearningCircuit(2)
    shared_parameter = circuit.add_parametric_RX_gate(0)
    circuit.add_parametric_RY_gate(1, share_with=shared_parameter)
    params = circuit.generate_bound_params(np.array([]), np.array([0.0]))
    assert np.array_equal(params, np.array([0.0, 0.0]))

    circuit_without_share = LearningCircuit(2)
    circuit_without_share.add_parametric_RX_gate(0)
    circuit_without_share.add_parametric_RY_gate(1)

    params = circuit.generate_bound_params(np.array([]), np.array([0.1]))
    params_without_share = circuit_without_share.generate_bound_params(
        np.array([]), np.array([0.1, 0.1])
    )
    assert np.array_equal(params, params_without_share)


def test_share_coef_input_learning_parameter() -> None:
    circuit = LearningCircuit(2)
    circuit.add_parametric_RX_gate(0)
    shared_parameter = circuit.add_parametric_RX_gate(0)
    circuit.add_parametric_RY_gate(1, share_with=shared_parameter, share_with_coef=2.0)
    params = circuit.generate_bound_params(np.array([]), np.array([0.1, 0.2]))

    circuit_without_share = LearningCircuit(2)
    circuit_without_share.add_parametric_RX_gate(0)
    circuit_without_share.add_parametric_RX_gate(0)
    circuit_without_share.add_parametric_RY_gate(1)
    params_circuit_without_share = circuit_without_share.generate_bound_params(
        np.array([]), np.array([0.1, 0.2, 0.4])
    )
    assert np.array_equal(params, params_circuit_without_share)


def test_zz_data_map_is_symmetric_and_matches_the_paper():
    """phi(x_i, x_j) = (pi - x_i)(pi - x_j).

    The port from scikit-qulacs moved a parenthesis and produced
    `pi - x_i * (pi - x_j)`, which is not symmetric in i and j even though the angle
    multiplies Z_i Z_j. Measured on 4 qubits, the resulting state had a mean fidelity
    of 0.086 against the intended one.
    """
    from scikit_quri.circuit.pre_defined import zz_data_map

    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.uniform(-1.0, 1.0, size=4)
        for i, j in [(0, 1), (1, 2), (3, 0)]:
            expected = (np.pi - x[i]) * (np.pi - x[j])
            assert zz_data_map(i, j)(x) == pytest.approx(expected)
            assert zz_data_map(i, j)(x) == pytest.approx(zz_data_map(j, i)(x))
