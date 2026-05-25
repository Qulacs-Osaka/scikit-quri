"""Parameter registry for LearningCircuit.

Separates parameter bookkeeping (learning / input / parametric-input) from the
gate-builder API. Resolution methods are pure: they read from the user-supplied
``parameters`` array and never mutate any registered parameter object across
calls. This makes resolution safe to share across threads and removes the
through-instance-state coupling that previously existed between
``generate_bound_params`` and ``to_batched``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

# Depends on x only
InputFunc = Callable[[NDArray[np.float64]], float]
# Depends on (theta, x)
InputFuncWithParam = Callable[[float, NDArray[np.float64]], float]


@dataclass
class PositionDetail:
    """A circuit-level gate position with an optional shared-parameter coefficient.

    A learning parameter may occupy several gate positions when share_with is
    used; ``coef`` scales the shared value (angle = shared_value * coef).
    ``coef=None`` means coefficient 1.
    """

    gate_pos: int
    coef: Optional[float]


@dataclass
class LearningParameter:
    """A trainable parameter, possibly occupying multiple gate positions via share_with.

    ``is_input`` marks a parameter that was implicitly created as the companion of a
    parametric-input gate. Such parameters are excluded from gradient backprop.
    """

    parameter_id: int
    positions_in_circuit: List[PositionDetail] = field(default_factory=list)
    is_input: bool = False

    def append_position(self, gate_pos: int, coef: Optional[float]) -> None:
        self.positions_in_circuit.append(PositionDetail(gate_pos, coef))


@dataclass
class InputParameter:
    """A parameter whose value at bind time is derived from input data ``x``.

    When ``companion_parameter_id`` is set the angle also depends on a learning
    parameter (``f(theta, x)``); otherwise it is purely ``f(x)``.
    """

    gate_pos: int
    func: Union[InputFunc, InputFuncWithParam]
    companion_parameter_id: Optional[int] = None

    @property
    def needs_learning_param(self) -> bool:
        return self.companion_parameter_id is not None


class ParameterRegistry:
    """Manages learning/input parameter bookkeeping and value resolution.

    Has no quantum-backend dependency. The owning circuit reports its current
    ``parameter_count`` via ``parameter_count_getter`` so the registry can size
    its output vectors without holding a reference back to the circuit.
    """

    def __init__(self, parameter_count_getter: Callable[[], int]) -> None:
        self._parameter_count_getter = parameter_count_getter
        self._learning_parameters: List[LearningParameter] = []
        self._input_parameters: List[InputParameter] = []
        # Template caches the learning-only contribution to the bound vector;
        # input positions are recomputed every call because they depend on x.
        # During fit the same ``parameters`` is reused across a whole batch,
        # so the template only needs to be rebuilt when parameters changes.
        self._template: Optional[NDArray[np.float64]] = None
        self._template_params: Optional[NDArray[np.float64]] = None

    # --- Registration -------------------------------------------------------

    def register_learning_param(
        self,
        gate_pos: int,
        share_with: Optional[int] = None,
        coef: Optional[float] = None,
        is_input: bool = False,
    ) -> int:
        """Register a new learning-parameter slot.

        If ``share_with`` is given, append ``gate_pos`` to that parameter's positions.
        Otherwise create a fresh learning parameter. Returns the parameter id.
        """
        if share_with is None:
            new_id = len(self._learning_parameters)
            param = LearningParameter(parameter_id=new_id, is_input=is_input)
            param.append_position(gate_pos, coef)
            self._learning_parameters.append(param)
        else:
            self._learning_parameters[share_with].append_position(gate_pos, coef)
        self._invalidate_template()
        return share_with if share_with is not None else new_id

    def register_input_param(
        self,
        gate_pos: int,
        func: Union[InputFunc, InputFuncWithParam],
        companion_parameter_id: Optional[int] = None,
    ) -> None:
        self._input_parameters.append(InputParameter(gate_pos, func, companion_parameter_id))

    # --- Read-only views ----------------------------------------------------

    @property
    def learning_params_count(self) -> int:
        return len(self._learning_parameters)

    @property
    def input_params_count(self) -> int:
        return len(self._input_parameters)

    @property
    def learning_parameters(self) -> Sequence[LearningParameter]:
        return self._learning_parameters

    @property
    def input_parameters(self) -> Sequence[InputParameter]:
        return self._input_parameters

    def learning_param_positions(self) -> List[int]:
        """All circuit-level gate positions covered by some learning parameter.

        Positions belonging to the same parameter via share_with appear separately.
        """
        return [p.gate_pos for lp in self._learning_parameters for p in lp.positions_in_circuit]

    def minimum_learning_param_positions(self) -> List[int]:
        """One representative gate position per unique learning parameter."""
        return [lp.positions_in_circuit[0].gate_pos for lp in self._learning_parameters]

    def input_param_positions(self) -> List[int]:
        return [ip.gate_pos for ip in self._input_parameters]

    # --- Resolution (pure) --------------------------------------------------

    def resolve_bound(
        self, x: NDArray[np.float64], parameters: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute the gate-level bound parameter vector for a single input.

        Args:
            x: Input data array.
            parameters: Learning-parameter vector of length ``learning_params_count``.

        Returns:
            Array of length ``parameter_count`` ready to pass to
            ``UnboundParametricQuantumCircuit.bind_parameters``.
        """
        bound = self._learning_template(parameters).copy()
        for ip in self._input_parameters:
            bound[ip.gate_pos] = self._resolve_input_angle(ip, x, parameters)
        return bound

    def resolve_batched(
        self, data: NDArray[np.float64], parameters: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute per-sample bound parameter vectors for a batch of inputs.

        Args:
            data: Input data of shape ``(n_samples, n_features)``.
            parameters: Learning-parameter vector of length ``learning_params_count``.

        Returns:
            Matrix of shape ``(n_samples, parameter_count)``.
        """
        template = self._learning_template(parameters)
        # Broadcast the learning template across samples; input positions get
        # overwritten per row below.
        batched = np.broadcast_to(template, (len(data), template.size)).copy()
        for i, x in enumerate(data):
            for ip in self._input_parameters:
                batched[i, ip.gate_pos] = self._resolve_input_angle(ip, x, parameters)
        return batched

    # --- Internal helpers ---------------------------------------------------

    def _resolve_input_angle(
        self,
        ip: InputParameter,
        x: NDArray[np.float64],
        parameters: NDArray[np.float64],
    ) -> float:
        if ip.companion_parameter_id is None:
            return ip.func(x)  # type: ignore[arg-type]
        theta_value = float(parameters[ip.companion_parameter_id])
        return ip.func(theta_value, x)  # type: ignore[arg-type]

    def _learning_template(self, parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the learning-only contribution to the bound vector.

        Cached across calls; rebuilt when ``parameters`` changes. The returned
        array is the internal cache — callers must ``.copy()`` before mutating.
        """
        if (
            self._template is not None
            and self._template_params is not None
            and self._template_params.shape == parameters.shape
            and np.array_equal(self._template_params, parameters)
        ):
            return self._template
        n = self._parameter_count_getter()
        template = np.zeros(n, dtype=np.float64)
        for lp in self._learning_parameters:
            value = float(parameters[lp.parameter_id])
            for pos in lp.positions_in_circuit:
                coef = pos.coef if pos.coef is not None else 1.0
                template[pos.gate_pos] = value * coef
        self._template = template
        self._template_params = np.asarray(parameters).copy()
        return template

    def _invalidate_template(self) -> None:
        self._template = None
        self._template_params = None
