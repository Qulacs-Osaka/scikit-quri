"""Parameter-shift gradient of expectation values.

Unlike the adjoint method this needs nothing but ordinary circuit executions, so it
is the gradient rule to use on real hardware.

Every parametric gate emitted by :class:`LearningCircuit` is a Pauli rotation
``exp(-i * phi * P / 2)``, for which::

    d<O>/d(phi) = ( <O>(phi + pi/2) - <O>(phi - pi/2) ) / 2

exactly, for any shot count — no step size to tune. Compare with a finite difference
at ``delta=1e-5``: with 1000 shots the sampling error is divided by 1e-5, so the
estimate is dominated by noise (measured spread was ~5 orders of magnitude larger
than the true value).

The shift is applied per **gate position** and the result is aggregated through the
circuit's parameter registry, so ``share_with`` coefficients and the ``df/dtheta``
factor of parametric-input gates are handled the same way as in every other gradient
path.

Cost: ``2 * (number of learning gate positions)`` circuit executions per observable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np
from numpy.typing import NDArray
from quri_parts.core.estimator import ConcurrentQuantumEstimator
from quri_parts.core.operator import Operator
from quri_parts.core.state import GeneralCircuitQuantumState

if TYPE_CHECKING:
    from ..circuit import LearningCircuit

_SHIFT = np.pi / 2


def parameter_shift_gradient(
    circuit: "LearningCircuit",
    x: NDArray[np.float64],
    theta: NDArray[np.float64],
    operator: Operator,
    estimator: ConcurrentQuantumEstimator,
) -> NDArray[np.float64]:
    """Gradient of ``<O>`` w.r.t. the learning parameters by the parameter-shift rule.

    Args:
        circuit: The learning circuit.
        x: Input data for one sample.
        theta: Learning-parameter vector.
        operator: Observable.
        estimator: Concurrent estimator; may be backed by a simulator or by hardware.

    Returns:
        Array of shape ``(circuit.learning_params_count,)``.
    """
    bound = np.asarray(circuit.generate_bound_params(x, theta), dtype=np.float64)
    positions = sorted(set(circuit.get_learning_params_indexes()))

    states: List[GeneralCircuitQuantumState] = []
    for position in positions:
        for sign in (+1.0, -1.0):
            shifted = bound.copy()
            shifted[position] += sign * _SHIFT
            states.append(
                GeneralCircuitQuantumState(
                    circuit.n_qubits, circuit.circuit.bind_parameters(list(shifted))
                )
            )

    estimates = list(estimator([operator] * len(states), states))

    gate_gradients = np.zeros(circuit.parameter_count, dtype=np.float64)
    for i, position in enumerate(positions):
        plus = np.real(estimates[2 * i].value)
        minus = np.real(estimates[2 * i + 1].value)
        gate_gradients[position] = (plus - minus) / 2.0

    chain_factors = circuit.input_chain_factors(x, theta)
    return circuit.aggregate_gate_gradients(
        gate_gradients, skip_is_input=False, chain_factors=chain_factors
    )
