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

from scikit_quri.backend import (
    BaseEstimator,
    BatchedSimEstimator,
    ExactStatevectorEstimator,
)
from scikit_quri.circuit import LearningCircuit
from scikit_quri.circuit.gradient import adjoint_expectation_gradients

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
    # Capability dispatch: any backend that natively supports batched
    # parametric evaluation (currently ScaluqEstimator) takes the fast path.
    if isinstance(estimator, BatchedSimEstimator):
        circuit, batched_params = ansatz.to_batched(x_scaled, params)
        results = estimator.estimate_batched(operators, circuit, batched_params)
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
    use_adjoint: bool = False,
) -> NDArray[np.float64]:
    """Estimate gradients of learning parameters for each input and operator.

    Args:
        ansatz: Learning circuit.
        gradient_estimator: Gradient estimator (used for qulacs path).
        operators: List of measurement operators. Length: n_operators.
        x_scaled: Scaled input data. Shape: (n_samples, n_features).
        params: Learning parameters.
        estimator: Optional estimator. If it implements BatchedSimEstimator,
            the batched (e.g. scaluq) numerical-gradient path is taken.
        delta: Finite difference step size for the batched numerical gradient.

    Returns:
        Gradient tensor. Shape: (n_samples, n_operators, n_learning_params).
    """
    # Exact statevector backends can use the adjoint method: one forward pass plus
    # one backpropagation per observable, instead of 2 * parameter_count circuit
    # simulations. Same quantity, ~10-140x faster and exact rather than O(delta^2).
    #
    # It is simulator-only: the adjoint method needs |psi> and O|psi> as vectors and
    # replays the circuit backwards, none of which is available on hardware, where
    # measurement collapses the state and recovering it would take exponentially many
    # tomography shots. Backends that cannot provide this (OQTOPUS and any other real
    # device) fall through to the supplied ``gradient_estimator``; the adjoint path is
    # never taken for them, so a hardware run is never silently replaced by a local
    # simulation. Use the parameter-shift rule there - it needs only ordinary circuit
    # executions - e.g. SimGradientEstimator(method="parameter_shift").
    if use_adjoint and isinstance(estimator, ExactStatevectorEstimator):
        return np.asarray(
            [adjoint_expectation_gradients(ansatz, x, params, list(operators)) for x in x_scaled]
        )

    # Capability dispatch: batched-simulation backends take the fast path.
    if isinstance(estimator, BatchedSimEstimator):
        n_learning = ansatz.learning_params_count
        circuit, shifted_params = ansatz.to_batched_for_gradient(x_scaled, params, delta)
        return estimator.estimate_grad_batched(
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

    # Parametric-input gates have angle = f(theta, x), so the per-gate derivative
    # d<O>/d(angle) must be scaled by df/dtheta to become d<O>/d(theta). The factor
    # depends on x, so it is applied per sample. Without it the reported gradient is
    # off by df/dtheta (e.g. exactly x for f = theta * x), which is not even a
    # constant factor across samples and breaks the descent direction.
    has_input_chain = any(
        ip.companion_parameter_id is not None for ip in ansatz._registry.input_parameters
    )

    grads = []
    values_matrix = np.zeros((n_ops, n_active), dtype=np.float64)
    for x in x_scaled:
        circuit_params = ansatz.generate_bound_params(x, params)
        for i, op in enumerate(operators):
            estimate = gradient_estimator(op, param_state, circuit_params)
            values = np.ascontiguousarray(np.asarray(estimate.values).real, dtype=np.float64)
            values_matrix[i, :] = values[active_idx]
        if has_input_chain:
            chain = ansatz.input_chain_factors(x, params)
            scale = np.array([chain.get(p, 1.0) for p in active_positions], dtype=np.float64)
            A_x = A * scale[:, None]
        else:
            A_x = A
        grads.append(np.einsum("ij,jk->ik", values_matrix, A_x))
    return np.asarray(grads)
