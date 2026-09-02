"""Adjoint (backpropagation) gradient of expectation values, qulacs-backed.

For a state ``|psi> = U(theta)|0>`` and an observable ``O``::

    d<psi|O|psi>/d(theta_j) = 2 Re( <O psi| dU/d(theta_j) |0> )

so the whole gradient vector needs **one** forward simulation plus one
backpropagation pass per observable, instead of the ``2 * parameter_count``
simulations a finite-difference estimator needs. On ``create_dqn_cl(6, 5, 2)``
(102 gate parameters) this is ~140x faster and, being exact, is also more
accurate than the ``delta=1e-10`` numerical estimator the README recommends.

Note on the sign: qulacs' ``backprop_inner_product`` uses the opposite rotation
convention, so the raw result is negated here. This is covered by
``tests/test_gradient_correctness.py``, which compares against finite differences.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.typing import NDArray
from qulacs import QuantumState as QulacsQuantumState
from quri_parts.core.operator import Operator
from quri_parts.qulacs.circuit import convert_parametric_circuit
from quri_parts.qulacs.operator import convert_operator

if TYPE_CHECKING:
    from ..circuit import LearningCircuit


def adjoint_expectation_gradients(
    circuit: "LearningCircuit",
    x: NDArray[np.float64],
    theta: NDArray[np.float64],
    operators: Sequence[Operator],
) -> NDArray[np.float64]:
    """Gradients of ``<O_k>`` w.r.t. the learning parameters, for several observables.

    Args:
        circuit: The learning circuit.
        x: Input data for one sample.
        theta: Learning-parameter vector.
        operators: Observables to differentiate.

    Returns:
        Array of shape ``(len(operators), circuit.learning_params_count)``.
    """
    n_qubits = circuit.n_qubits
    bound_params = circuit.generate_bound_params(x, theta)
    qulacs_circuit, param_mapper = convert_parametric_circuit(circuit.circuit)
    for i, value in enumerate(param_mapper(bound_params)):
        qulacs_circuit.set_parameter(i, value)

    psi = QulacsQuantumState(n_qubits)
    psi.set_zero_state()
    qulacs_circuit.update_quantum_state(psi)

    chain_factors = circuit.input_chain_factors(x, theta)
    work = QulacsQuantumState(n_qubits)
    o_psi = QulacsQuantumState(n_qubits)

    grads = np.empty((len(operators), circuit.learning_params_count), dtype=np.float64)
    for k, operator in enumerate(operators):
        qulacs_operator = convert_operator(operator, n_qubits)
        o_psi.set_zero_state()
        qulacs_operator.apply_to_state(work, psi, o_psi)
        gate_gradients = -2.0 * np.asarray(
            qulacs_circuit.backprop_inner_product(o_psi), dtype=np.float64
        )
        grads[k] = circuit.aggregate_gate_gradients(
            gate_gradients, skip_is_input=False, chain_factors=chain_factors
        )
    return grads
