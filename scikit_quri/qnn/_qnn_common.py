# mypy: ignore-errors
"""Common quantum circuit execution utilities shared by QNNClassifier and QNNRegressor.

Provides state preparation, expectation value computation, and gradient estimation
logic that is delegated from each QNN class.
"""

from typing import List, Sequence

import numpy as np
from numpy.typing import NDArray
from quri_parts.algo.optimizer import Params
from quri_parts.core.estimator import Estimatable, GradientEstimator
from quri_parts.core.estimator.gradient import _ParametricStateT
from quri_parts.core.state import ParametricCircuitQuantumState, quantum_state
from quri_parts.qulacs import QulacsStateT
from typing_extensions import TypeAlias

from scikit_quri.backend import BaseEstimator
from scikit_quri.backend.sim_estimator import SimEstimator
from scikit_quri.circuit import LearningCircuit

GradientEstimatorType: TypeAlias = GradientEstimator[_ParametricStateT]


def build_circuit_states(
    ansatz: LearningCircuit,
    x_scaled: NDArray[np.float64],
    params: Params,
) -> List[QulacsStateT]:
    """Build a list of parameter-bound quantum states for each input sample.

    Args:
        ansatz: Learning circuit.
        x_scaled: Scaled input data. Shape: (n_samples, n_features).
        params: Learning parameters.

    Returns:
        List of bound quantum states, one per input sample. Length: n_samples.
    """
    circuit_states: List[QulacsStateT] = []
    for x in x_scaled:
        circuit_params = ansatz.generate_bound_params(x, params)
        param_circuit_state: ParametricCircuitQuantumState = quantum_state(  # type: ignore
            n_qubits=ansatz.n_qubits, circuit=ansatz.circuit
        )
        circuit_state = param_circuit_state.bind_parameters(circuit_params)
        circuit_states.append(circuit_state)
    return circuit_states


def compute_expectations(
    estimator: BaseEstimator,
    operators: Sequence[Estimatable],
    circuit_states: List[QulacsStateT],
    y_exp_ratio: float,
) -> NDArray[np.float64]:
    """Compute expectation values of each operator for a list of quantum states.

    Args:
        estimator: Expectation value estimator.
        operators: List of measurement operators. Length: n_operators.
        circuit_states: List of bound quantum states. Length: n_samples.
        y_exp_ratio: Scaling factor applied to expectation values.

    Returns:
        Expectation value matrix. Shape: (n_samples, n_operators).
    """
    n_samples = len(circuit_states)
    n_ops = len(operators)
    res = np.zeros((n_samples, n_ops), dtype=np.float64)
    for i, op in enumerate(operators):
        estimates = estimator.estimate([op], circuit_states)
        res[:, i] = np.array([e.value.real for e in estimates])
    res *= y_exp_ratio
    return res


def predict_inner(
    ansatz: LearningCircuit,
    estimator: BaseEstimator,
    operators: Sequence[Estimatable],
    x_scaled: NDArray[np.float64],
    params: Params,
    y_exp_ratio: float,
) -> NDArray[np.float64]:
    """Compute expectation-based predictions for the given input data.

    Runs build_circuit_states followed by compute_expectations.

    Args:
        ansatz: Learning circuit.
        estimator: Expectation value estimator.
        operators: List of measurement operators. Length: n_operators.
        x_scaled: Scaled input data. Shape: (n_samples, n_features).
        params: Learning parameters.
        y_exp_ratio: Scaling factor applied to expectation values.

    Returns:
        Prediction matrix. Shape: (n_samples, n_operators).
    """
    # Use scaluq batched estimation if enabled
    if isinstance(estimator, SimEstimator) and estimator.use_scaluq:
        circuit, batched_params = ansatz.to_batched(x_scaled, params)
        results = estimator.estimate_scaluq_batched(operators, circuit, batched_params)
        # results: (n_operators, n_samples) -> transpose to (n_samples, n_operators)
        res = np.array(results, dtype=np.float64).T
        res *= y_exp_ratio
        return res

    circuit_states = build_circuit_states(ansatz, x_scaled, params)
    return compute_expectations(estimator, operators, circuit_states, y_exp_ratio)


def estimate_grad(
    ansatz: LearningCircuit,
    gradient_estimator: GradientEstimatorType,
    operators: Sequence[Estimatable],
    x_scaled: NDArray[np.float64],
    params: Params,
    estimator: BaseEstimator | None = None,
    delta: float = 1e-5,
) -> NDArray[np.float64]:
    """Estimate gradients of learning parameters for each input and operator.

    Args:
        ansatz: Learning circuit.
        gradient_estimator: Gradient estimator (used for qulacs path).
        operators: List of measurement operators. Length: n_operators.
        x_scaled: Scaled input data. Shape: (n_samples, n_features).
        params: Learning parameters.
        estimator: Optional estimator. If SimEstimator with use_scaluq=True, uses scaluq batched path.
        delta: Finite difference step size for scaluq numerical gradient.

    Returns:
        Gradient tensor. Shape: (n_samples, n_operators, n_learning_params).
    """
    # Use scaluq batched gradient if enabled
    if isinstance(estimator, SimEstimator) and estimator.use_scaluq:
        n_learning = ansatz.learning_params_count
        circuit, shifted_params = ansatz.to_batched_for_gradient(x_scaled, params, delta)
        return estimator.estimate_grad_scaluq_batched(
            operators,
            circuit,
            shifted_params,
            len(x_scaled),
            n_learning,
            delta,
        )

    n_ops = len(operators)
    n_learning_params = ansatz.learning_params_count
    grads = []

    # Build aggregation map from the circuit: for each learnable parameter,
    # list of (gate_pos, coef) spanning all shared gate positions.
    agg_map = ansatz.get_learning_param_grad_aggregators()

    for x in x_scaled:
        circuit_params = ansatz.generate_bound_params(x, params)
        param_state = quantum_state(n_qubits=ansatz.n_qubits, circuit=ansatz.circuit)
        grad = np.zeros((n_ops, n_learning_params), dtype=np.float64)
        for i, op in enumerate(operators):
            estimate = gradient_estimator(op, param_state, circuit_params)
            values = np.array(estimate.values).real
            for j, param_aggs in enumerate(agg_map):
                total = 0.0
                for gate_pos, coef in param_aggs:
                    total += values[gate_pos] * coef
                grad[i, j] = total
        grads.append(grad)
    return np.asarray(grads)
