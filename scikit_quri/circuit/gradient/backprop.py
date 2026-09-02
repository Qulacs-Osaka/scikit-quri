"""Qulacs-backed backpropagation gradient for LearningCircuit.

Converts the parametric circuit to qulacs format, runs
``backprop_inner_product`` against the target state, then maps the per-gate
gradients back to per-learning-parameter gradients via the parameter registry's
share_with aggregation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from qulacs import QuantumState as QulacsQuantumState
from quri_parts.qulacs.circuit import convert_parametric_circuit

if TYPE_CHECKING:
    from ..circuit import LearningCircuit


def backprop_inner_product(
    circuit: "LearningCircuit",
    x: NDArray[np.float64],
    theta: NDArray[np.float64],
    state: QulacsQuantumState,
) -> NDArray[np.float64]:
    """Compute gradients of learnable parameters via qulacs backpropagation using inner product.

    Args:
        circuit: The learning circuit whose parameters are differentiated.
        x: Input data array.
        theta: Learnable parameter vector of length ``circuit.learning_params_count``.
        state: Target qulacs quantum state used in the inner product.

    Returns:
        Gradient array of shape ``(learning_params_count,)``.
    """
    params = circuit.generate_bound_params(x, theta)
    qulacs_circuit, param_mapper = convert_parametric_circuit(circuit.circuit)
    for i, v in enumerate(param_mapper(params)):
        qulacs_circuit.set_parameter(i, v)
    gate_gradients = np.asarray(qulacs_circuit.backprop_inner_product(state), dtype=np.float64)
    # Parametric-input gates need the df/dtheta chain factor; previously they were
    # silently zeroed here while the other gradient paths reported d<O>/d(angle),
    # so the three paths disagreed on the same circuit.
    chain_factors = circuit.input_chain_factors(x, theta)
    return circuit.aggregate_gate_gradients(
        gate_gradients, skip_is_input=False, chain_factors=chain_factors
    )
