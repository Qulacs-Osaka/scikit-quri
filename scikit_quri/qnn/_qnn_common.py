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
    # Hoist parametric state construction out of the per-sample loop:
    # the underlying circuit is identical across samples; only bound params change.
    param_circuit_state: ParametricCircuitQuantumState = quantum_state(  # type: ignore
        n_qubits=ansatz.n_qubits, circuit=ansatz.circuit
    )
    circuit_states: List[QulacsStateT] = []
    for x in x_scaled:
        circuit_params = ansatz.generate_bound_params(x, params)
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
    operators_list = list(operators)
    # Flatten to 1-to-1 (op, state) pairs in row-major order:
    # pair index s*n_ops + i corresponds to (operators[i], circuit_states[s]).
    # A single concurrent estimator call lets the backend parallelize across
    # both axes instead of serializing across operators.
    ops_flat = operators_list * n_samples
    states_flat = [s for s in circuit_states for _ in range(n_ops)]
    estimates = estimator.estimate(ops_flat, states_flat)
    res = np.fromiter(
        (e.value.real for e in estimates), dtype=np.float64, count=n_samples * n_ops
    ).reshape(n_samples, n_ops)
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


def predict_inner_cached(
    ansatz: LearningCircuit,
    estimator: BaseEstimator,
    operators: Sequence[Estimatable],
    x_scaled: NDArray[np.float64],
    params: Params,
    y_exp_ratio: float,
    cache: dict,
) -> NDArray[np.float64]:
    """Compute predictions with caching of the last (params, x_scaled) pair.

    During optimization ``cost_func`` and ``grad_func`` are called back-to-back
    with the same parameters. This cache avoids running the circuit twice per
    step by storing the most recent result keyed on the params content and a
    composite x_scaled fingerprint.

    The x fingerprint combines ``id(x_scaled)`` with ``shape`` and ``dtype`` to
    guard against the case where the previously cached array was garbage
    collected and a new array reuses the same memory address. A params hash is
    used as a fast-fail check before the exact ``np.array_equal`` comparison.

    Args:
        ansatz: Learning circuit.
        estimator: Expectation value estimator.
        operators: List of measurement operators. Shape: (n_operators,).
        x_scaled: Scaled input data. Shape: (n_samples, n_features).
        params: Learning parameters.
        y_exp_ratio: Scaling factor applied to expectation values.
        cache: Mutable dict with keys ``cached_params`` (Optional[NDArray]),
            ``y_pred`` (Optional[NDArray]), ``cached_x_fp`` (Optional[tuple]),
            ``cached_params_hash`` (Optional[int]).

    Returns:
        Prediction matrix. Shape: (n_samples, n_operators).
    """
    params_arr = np.ascontiguousarray(np.asarray(params))
    x_fp = (id(x_scaled), x_scaled.shape, x_scaled.dtype)
    params_hash = hash(params_arr.tobytes())
    cached_params: NDArray[np.float64] | None = cache.get("cached_params")
    cached_y_pred: NDArray[np.float64] | None = cache.get("y_pred")
    cached_x_fp: tuple | None = cache.get("cached_x_fp")
    cached_params_hash: int | None = cache.get("cached_params_hash")
    if (
        cached_params is not None
        and cached_y_pred is not None
        and cached_x_fp == x_fp
        and cached_params_hash == params_hash
        and cached_params.shape == params_arr.shape
        and np.array_equal(cached_params, params_arr)
    ):
        return cached_y_pred
    y_pred = predict_inner(ansatz, estimator, operators, x_scaled, params, y_exp_ratio)
    cache["cached_params"] = params_arr.copy()
    cache["y_pred"] = y_pred
    cache["cached_x_fp"] = x_fp
    cache["cached_params_hash"] = params_hash
    return y_pred


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

    # Build aggregation map from the circuit: for each learnable parameter,
    # list of (gate_pos, coef) spanning all shared gate positions.
    agg_map = ansatz.get_learning_param_grad_aggregators()

    # Compact sparse aggregation matrix using only positions referenced by some
    # learning parameter. Restricting to active positions avoids 0 * inf = NaN
    # contamination when the gradient estimator returns non-finite values at
    # input-only parameter slots (which the old per-element loop never touched).
    active_positions = sorted({gp for aggs in agg_map for gp, _ in aggs})
    pos_to_row = {p: i for i, p in enumerate(active_positions)}
    n_active = len(active_positions)
    A = np.zeros((n_active, n_learning_params), dtype=np.float64)
    for j, param_aggs in enumerate(agg_map):
        for gate_pos, coef in param_aggs:
            A[pos_to_row[gate_pos], j] = coef
    active_idx = np.asarray(active_positions, dtype=np.int64)

    # Hoist parametric state construction out of the per-sample loop.
    param_state = quantum_state(n_qubits=ansatz.n_qubits, circuit=ansatz.circuit)

    grads = []
    values_matrix = np.zeros((n_ops, n_active), dtype=np.float64)
    for x in x_scaled:
        circuit_params = ansatz.generate_bound_params(x, params)
        for i, op in enumerate(operators):
            estimate = gradient_estimator(op, param_state, circuit_params)
            values = np.ascontiguousarray(np.asarray(estimate.values).real, dtype=np.float64)
            values_matrix[i, :] = values[active_idx]
        grads.append(np.einsum("ij,jk->ik", values_matrix, A))
    return np.asarray(grads)
