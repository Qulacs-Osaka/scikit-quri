"""Gradient estimation on OQTOPUS backends.

Uses the parameter-shift rule. The previous implementation took a central difference
with a hardcoded ``delta=1e-5`` on shot-based expectation values: with the default
1000 shots the sampling error of each term is divided by 1e-5, so the result was
noise (measured spread ~5 orders of magnitude above the true value). The
parameter-shift rule is exact for the Pauli rotations this library emits, needs no
step size, and costs the same 2 executions per gate position.
"""

from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from quri_parts.core.estimator import Estimatable, Estimates

from scikit_quri.circuit import LearningCircuit
from scikit_quri.circuit.gradient import parameter_shift_gradient

from .base_gradient_estimator import BaseGradientEstimator
from .oqtopus_estimator import OqtopusEstimator


class OqtopusGradientEstimator(BaseGradientEstimator):
    """Gradient estimator backed by OQTOPUS, using the parameter-shift rule."""

    def __init__(self, device_id: str = "qulacs", shots: int = 1000) -> None:
        self.estimator = OqtopusEstimator(device_id, shots=shots)

    def _concurrent_estimator(self, operators, states):
        return self.estimator.estimate(operators, states)

    def estimate_gradient(
        self, operators: Estimatable, state, params: Sequence[float]
    ) -> Estimates[complex]:
        """Not available.

        The parameter-shift rule is defined per learning parameter of a
        :class:`LearningCircuit`; use :meth:`estimate_learning_param_gradient`.
        """
        raise NotImplementedError(
            "OqtopusGradientEstimator only supports estimate_learning_param_gradient; "
            "the parameter-shift rule needs the circuit's parameter registry to map "
            "gate positions back to learning parameters."
        )

    def estimate_learning_param_gradient(
        self,
        operators: Estimatable,
        circuit: LearningCircuit,
        params: Sequence[float],
        x: Optional[NDArray[np.float64]] = None,
        theta: Optional[NDArray[np.float64]] = None,
    ) -> Sequence[complex]:
        """Gradient w.r.t. the learning parameters, of length ``learning_params_count``.

        Args:
            operators: Observable.
            circuit: The learning circuit.
            params: Unused; kept for interface compatibility. The shifts are applied
                to the gate-level parameters resolved from ``x`` and ``theta``.
            x: Input data for the sample being differentiated. Required.
            theta: Learning-parameter vector. Required.

        Raises:
            ValueError: If ``x`` or ``theta`` is missing. The previous implementation
                sliced the already-resolved gate angles out of ``params`` and fed them
                back in as if they were raw input data and a theta vector, which is a
                different circuit whenever an input function is not the identity, and
                has the wrong length as soon as ``share_with`` is used.
        """
        if x is None or theta is None:
            raise ValueError(
                "estimate_learning_param_gradient needs x and theta: the parameter-shift "
                "rule differentiates w.r.t. the learning parameters, and the chain rule "
                "for parametric-input gates (angle = f(theta, x)) needs both."
            )
        grad = parameter_shift_gradient(
            circuit, np.asarray(x), np.asarray(theta), operators, self._concurrent_estimator
        )
        return grad.tolist()
