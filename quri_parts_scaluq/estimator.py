# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence, Mapping
from typing import TYPE_CHECKING, Callable, NamedTuple

import numpy as np
from numpy.typing import NDArray

from quri_parts.circuit import ParametricQuantumCircuitProtocol
from quri_parts.core.estimator import (
    Estimatable,
    Estimate,
    ParametricQuantumEstimator,
    QuantumEstimator,
)
from quri_parts.core.operator import zero
from quri_parts.core.state import ParametricQuantumStateVector, QuantumStateVector

if TYPE_CHECKING:
    from scaluq.default.f64 import StateVectorBatched

from . import cast_to_list, _backend, scaluqStateT, scaluqParametricStateT
from .circuit import convert_circuit, convert_parametric_circuit
from .operator import convert_operator


class _Estimate(NamedTuple):
    value: complex
    error: float = 0.0


def _create_scaluq_initial_state(state: scaluqStateT) -> _backend.StateVector:
    sq_state = _backend.StateVector(state.qubit_count)
    if isinstance(state, (QuantumStateVector, ParametricQuantumStateVector)):
        sq_state.load(cast_to_list(state.vector))
    return sq_state


def _estimate(operator: Estimatable, state: scaluqStateT) -> Estimate[complex]:
    if operator == zero():
        return _Estimate(value=0.0)

    circuit = convert_circuit(state.circuit)
    sq_state = _create_scaluq_initial_state(state)

    op = convert_operator(operator, state.qubit_count)
    circuit.update_quantum_state(sq_state)
    exp = op.get_expectation_value(sq_state)

    return _Estimate(value=exp)


def _sequential_parametric_estimate(
    op_state: tuple[Estimatable, scaluqParametricStateT],
    params: Sequence[Sequence[float]],
) -> Sequence[Estimate[complex]]:
    operator, state = op_state
    n_qubits = state.qubit_count
    op = convert_operator(operator, n_qubits)
    parametric_circuit = state.parametric_circuit

    scaluq_circuit, param_mapper = convert_parametric_circuit(parametric_circuit)

    estimates = []
    for param in params:
        tmp_params: Mapping[str, float] = {}
        for i in range(len(param)):
            tmp_params[str(i)] = param[i]

        sq_state = _create_scaluq_initial_state(state)
        scaluq_circuit.update_quantum_state(sq_state, tmp_params)
        exp = op.get_expectation_value(sq_state)
        estimates.append(_Estimate(value=exp))

    return estimates


def create_scaluq_vector_estimator() -> QuantumEstimator[scaluqStateT]:
    return _estimate


def create_scaluq_vector_parametric_estimator() -> ParametricQuantumEstimator[
    scaluqParametricStateT
]:
    def estimator(
        operator: Estimatable, state: scaluqParametricStateT, param: Sequence[float]
    ) -> Estimate[complex]:
        ests = _sequential_parametric_estimate((operator, state), [param])
        return ests[0]

    return estimator


def _create_scaluq_initial_state_batched(
    state: scaluqStateT,
    batch_num: int,
) -> _backend.StateVectorBatched:
    sq_state = _backend.StateVectorBatched(batch_num, state.qubit_count)

    if isinstance(state, (QuantumStateVector, ParametricQuantumStateVector)):
        single_state_list = cast_to_list(state.vector)
        batched_states = [single_state_list for _ in range(batch_num)]
        sq_state.load(cast_to_list(batched_states))
    return sq_state


def _batched_parametric_estimate(
    op_state: tuple[Estimatable, scaluqParametricStateT],
    params: Sequence[Sequence[float]],
) -> Sequence[Estimate[complex]]:
    operator, state = op_state
    n_qubits = state.qubit_count
    op = convert_operator(operator, n_qubits)
    parametric_circuit = state.parametric_circuit
    scaluq_circuit, param_mapper = convert_parametric_circuit(parametric_circuit)

    sq_state_batched = _create_scaluq_initial_state_batched(state, len(params))

    batched_params: dict[str, list[float]] = {}
    for i in range(len(params[0])):
        batched_params[str(i)] = [param[i] for param in params]

    scaluq_circuit.update_quantum_state(sq_state_batched, batched_params)
    exp = op.get_expectation_value(sq_state_batched)

    return [_Estimate(value=val) for val in exp]


def create_scaluq_vector_batched_parametric_estimator() -> ParametricQuantumEstimator[
    scaluqParametricStateT
]:
    def estimator(
        operator: Estimatable, state: scaluqParametricStateT, param: Sequence[float]
    ) -> Estimate[complex]:
        ests = _batched_parametric_estimate((operator, state), [param])
        return ests[0]

    return estimator


def _batched_param_mapper(
    param_mapper: Callable[[Sequence[float]], dict[str, list[float]]],
    params_list: NDArray[np.float64],
) -> dict[str, list[float]]:
    mapped = [param_mapper(params) for params in params_list]
    keys = mapped[0].keys()
    return {k: [m[k][0] for m in mapped] for k in keys}


def estimate(
    state: StateVectorBatched,
    circuit: ParametricQuantumCircuitProtocol,
    operators: Sequence[Estimatable],
    params: NDArray[np.float64],
) -> list[list[float]]:
    """Batched estimation of expectation values.

    Args:
        state: StateVectorBatched, will be modified in-place.
        circuit: ParametricQuantumCircuitProtocol to convert and apply.
        operators: Sequence of operators to measure.
        params: Batched parameters, shape (n_samples, n_params).

    Returns:
        List of shape (n_operators, n_samples) containing real expectation values.
    """
    n_qubits = circuit.qubit_count
    scaluq_circuit, param_mapper = convert_parametric_circuit(circuit)
    batched_params = _batched_param_mapper(param_mapper, params)

    scaluq_circuit.update_quantum_state(state, batched_params)

    results: list[list[float]] = []
    for op in operators:
        scaluq_op = convert_operator(op, n_qubits)
        exps = scaluq_op.get_expectation_value(state)
        results.append([v.real for v in exps])

    return results


def estimate_numerical_gradient(
    circuit: ParametricQuantumCircuitProtocol,
    operators: Sequence[Estimatable],
    shifted_params: NDArray[np.float64],
    n_samples: int,
    n_learning_params: int,
    delta: float,
) -> NDArray[np.float64]:
    """Batched numerical gradient estimation using scaluq.

    Args:
        circuit: Parametric quantum circuit.
        operators: List of measurement operators. Length: n_operators.
        shifted_params: Shifted parameter array from to_batched_for_gradient.
            Shape: (n_samples * 2 * n_learning_params, parameter_count).
        n_samples: Number of input samples.
        n_learning_params: Number of learning parameters.
        delta: Finite difference step size used to build shifted_params.

    Returns:
        Gradient tensor. Shape: (n_samples, n_operators, n_learning_params).
    """
    total_batch = shifted_params.shape[0]
    n_qubits = circuit.qubit_count
    n_ops = len(operators)

    scaluq_circuit, param_mapper = convert_parametric_circuit(circuit)
    batched_params = _batched_param_mapper(param_mapper, shifted_params)

    state = _backend.StateVectorBatched(total_batch, n_qubits)
    state.set_zero_state()
    scaluq_circuit.update_quantum_state(state, batched_params)

    grads = np.zeros((n_samples, n_ops, n_learning_params), dtype=np.float64)
    for op_idx, op in enumerate(operators):
        scaluq_op = convert_operator(op, n_qubits)
        exps = scaluq_op.get_expectation_value(state)
        exps_real = np.fromiter((e.real for e in exps), dtype=np.float64, count=total_batch)
        plus_vals = exps_real[0::2]
        minus_vals = exps_real[1::2]
        grads[:, op_idx, :] = ((plus_vals - minus_vals) / delta).reshape(
            n_samples, n_learning_params
        )

    return grads
