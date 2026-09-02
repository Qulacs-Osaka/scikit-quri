"""Adjoint (backpropagation) gradient of expectation values, qulacs-backed.

For a state ``|psi> = U(theta)|0>`` and an observable ``O``::

    d<psi|O|psi>/d(theta_j) = 2 Re( <O psi| dU/d(theta_j) |0> )

so the whole gradient vector needs **one** forward simulation plus one
backpropagation pass per observable, instead of the ``2 * parameter_count``
simulations a finite-difference estimator needs. On ``create_dqn_cl(6, 5, 2)``
(102 gate parameters) this is ~140x faster and, being exact, is also more
accurate than the ``delta=1e-10`` numerical estimator the README recommends.

The circuit structure is identical across samples — only the bound parameters
change — so the qulacs conversion of the circuit and of the observables is done
once per batch rather than once per sample. Profiling ``test_qcnn`` showed
``convert_gate`` being called 1,063,040 times, about 60% of the runtime, purely
from rebuilding the same circuit.

Note on the sign: qulacs' ``backprop_inner_product`` uses the opposite rotation
convention, so the raw result is negated here. This is covered by
``tests/test_gradient_correctness.py``, which compares against finite differences.

It is simulator-only: hardware cannot expose ``|psi>`` as a vector, nor replay the
circuit backwards. Use :mod:`~scikit_quri.circuit.gradient.parameter_shift` there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence

import numpy as np
from numpy.typing import NDArray
from qulacs import QuantumState as QulacsQuantumState
from quri_parts.core.operator import Operator
from quri_parts.qulacs.circuit import convert_parametric_circuit
from quri_parts.qulacs.operator import convert_operator

if TYPE_CHECKING:
    from ..circuit import LearningCircuit


def _prepared(circuit: "LearningCircuit", operators: Sequence[Operator]):
    """Convert the parametric circuit and the observables to qulacs once."""
    qulacs_circuit, param_mapper = convert_parametric_circuit(circuit.circuit)
    qulacs_operators = [convert_operator(op, circuit.n_qubits) for op in operators]
    return qulacs_circuit, param_mapper, qulacs_operators


def adjoint_expectation_gradients_batch(
    circuit: "LearningCircuit",
    x_batch: NDArray[np.float64],
    theta: NDArray[np.float64],
    operators: Sequence[Operator],
) -> NDArray[np.float64]:
    """Gradients for a whole batch of inputs.

    Args:
        circuit: The learning circuit.
        x_batch: Input data of shape ``(n_samples, n_features)``.
        theta: Learning-parameter vector.
        operators: Observables to differentiate.

    Returns:
        Array of shape ``(n_samples, len(operators), circuit.learning_params_count)``.
    """
    n_qubits = circuit.n_qubits
    qulacs_circuit, param_mapper, qulacs_operators = _prepared(circuit, operators)

    psi = QulacsQuantumState(n_qubits)
    work = QulacsQuantumState(n_qubits)
    o_psi = QulacsQuantumState(n_qubits)

    grads = np.empty(
        (len(x_batch), len(operators), circuit.learning_params_count), dtype=np.float64
    )
    for s, x in enumerate(x_batch):
        bound_params = circuit.generate_bound_params(x, theta)
        for i, value in enumerate(param_mapper(bound_params)):
            qulacs_circuit.set_parameter(i, value)

        psi.set_zero_state()
        qulacs_circuit.update_quantum_state(psi)

        chain_factors = circuit.input_chain_factors(x, theta)
        for k, qulacs_operator in enumerate(qulacs_operators):
            o_psi.set_zero_state()
            qulacs_operator.apply_to_state(work, psi, o_psi)
            gate_gradients = -2.0 * np.asarray(
                qulacs_circuit.backprop_inner_product(o_psi), dtype=np.float64
            )
            grads[s, k] = circuit.aggregate_gate_gradients(
                gate_gradients, skip_is_input=False, chain_factors=chain_factors
            )
    return grads


def adjoint_expectation_gradients(
    circuit: "LearningCircuit",
    x: NDArray[np.float64],
    theta: NDArray[np.float64],
    operators: Sequence[Operator],
) -> NDArray[np.float64]:
    """Gradients of ``<O_k>`` w.r.t. the learning parameters, for one sample.

    Returns:
        Array of shape ``(len(operators), circuit.learning_params_count)``.
    """
    return adjoint_expectation_gradients_batch(
        circuit, np.asarray(x).reshape(1, -1), theta, operators
    )[0]


def exact_expectations_batch(
    circuit: "LearningCircuit",
    x_batch: NDArray[np.float64],
    theta: NDArray[np.float64],
    operators: Sequence[Operator],
) -> NDArray[np.float64]:
    """Exact expectation values for a batch, reusing one qulacs circuit conversion.

    Equivalent to binding each sample separately and calling a state-vector
    estimator, but converts the circuit and the observables once instead of once
    per sample.

    Returns:
        Array of shape ``(n_samples, len(operators))``.
    """
    n_qubits = circuit.n_qubits
    qulacs_circuit, param_mapper, qulacs_operators = _prepared(circuit, operators)

    psi = QulacsQuantumState(n_qubits)
    values: List[List[float]] = []
    for x in x_batch:
        bound_params = circuit.generate_bound_params(x, theta)
        for i, value in enumerate(param_mapper(bound_params)):
            qulacs_circuit.set_parameter(i, value)
        psi.set_zero_state()
        qulacs_circuit.update_quantum_state(psi)
        values.append([op.get_expectation_value(psi).real for op in qulacs_operators])
    return np.asarray(values, dtype=np.float64)
