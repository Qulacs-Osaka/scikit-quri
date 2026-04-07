from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional, Sequence, Tuple, TypeGuard, Union

import numpy as np
from numpy.typing import NDArray
from qulacs import QuantumState as QulacsQuantumState
from quri_parts.circuit import (
    Parameter,
    ParametricQuantumGate,
    QuantumCircuit,
    QuantumGate,
    UnboundParametricQuantumCircuit,
)
from quri_parts.core.estimator import ConcurrentQuantumEstimator
from quri_parts.core.operator import Operator, commutator, pauli_label
from quri_parts.core.state import GeneralCircuitQuantumState
from quri_parts.qulacs.circuit import convert_parametric_circuit
from quri_parts.rust.circuit.circuit_parametric import ImmutableBoundParametricQuantumCircuit


class _Axis(Enum):
    """Specifying axis. Used in inner private method in LearningCircuit."""

    X = auto()
    Y = auto()
    Z = auto()


# Depends on x
InputFunc = Callable[[NDArray[np.float64]], float]

# Depends on theta, x
InputFuncWithParam = Callable[[float, NDArray[np.float64]], float]


@dataclass
class _PositionDetail:
    """Manage a parameter of `ParametricQuantumCircuit.positions_in_circuit`.
    This class manages indexe and coefficients (optional) of gate.
    Args:
        gate_pos: Indices of a parameter in LearningCircuit._circuit.
        coef: Coefficient of a parameter in LearningCircuit._circuit. It's a optional.
    """

    gate_pos: int
    coef: Optional[float]


@dataclass
class _InputParameter:
    """Manage transformation of an input.
    `func` transforms the given input and the outcome is stored at `pos`-th parameter in `LearningCircuit._circuit`.
    If the `func` needs a learning parameter, supply `companion_parameter_id` with the learning parameter's `parameter_id`.
    """

    pos: int
    func: Union[InputFunc, InputFuncWithParam] = field(compare=False)
    companion_parameter_id: Optional[int]


# TypeGuardでfuncの型を分けるための関数群
def need_learning_parameter_guard(
    func: Union[InputFunc, InputFuncWithParam], companion_parameter_id: Optional[int]
) -> TypeGuard[InputFunc]:
    return companion_parameter_id is None


def not_needed_learning_parameter_guard(
    func: Union[InputFunc, InputFuncWithParam], companion_parameter_id: Optional[int]
) -> TypeGuard[InputFuncWithParam]:
    return companion_parameter_id is not None


@dataclass
class _LearningParameter:
    """Manage a parameter of `ParametricQuantumCircuit`.
    This class manages index and value of parameter.
    There is two member variables to note: `positions_in_circuit` and `parameter_id`.
    `positions_in_circuit` is indices of parameters in `ParametricQuantumCircuit` held by `LearningCircuit`.
    If you change the parameter value of the `_LearningParameter` instance, all of the parameters
    specified in `positions_in_circuit` are also updated with that value.
    And `parameter_id` is an index of a whole set of learning parameters.
    This is used by method of `LearningCircuit` which has "parametric" in its name.

    Args:
        positions_in_circuit: Indices and coefficient of a parameter in LearningCircuit._circuit.
        parameter_id: Index at array of learning parameter(theta).
        value: Current `parameter_id`-th parameter of LearningCircuit._circuit.
        is_input: Whethter this parameter is used with a input parameter.
    """

    positions_in_circuit: List[_PositionDetail]
    parameter_id: int
    value: float
    is_input: bool = field(default=False)

    def __init__(self, parameter_id: int, value: float, is_input: bool = False) -> None:
        self.positions_in_circuit = []
        self.parameter_id = parameter_id
        self.value = value
        self.is_input = is_input

    def append_position(self, position: int, coef: Optional[float]) -> None:
        self.positions_in_circuit.append(_PositionDetail(position, coef))


@dataclass
class LearningCircuit:
    """Parametric quantum circuit for quantum machine learning.

    Manages three types of circuit parameters:

    - **Fixed gates**: Non-parametric gates (X, Y, Z, H, CNOT, etc.).
    - **Input gates** (``add_input_R*``): Rotation angle is computed from input data ``x``
      via a user-supplied function at inference time.
    - **Learnable gates** (``add_parametric_R*``): Rotation angle is a trainable parameter
      updated during optimization. Multiple gates can share a single parameter via
      ``share_with`` / ``share_with_coef``.
    - **Parametric-input gates** (``add_parametric_input_R*``): Angle depends on both a
      learnable parameter and input data (e.g. ``f(theta, x)``).

    Args:
        n_qubits: Number of qubits in the circuit.

    Example:
        >>> circuit = LearningCircuit(n_qubits=4)
        >>> circuit.add_input_RY_gate(0, lambda x: np.arcsin(x[0]))
        >>> param_id = circuit.add_parametric_RX_gate(1)
        >>> bound = circuit.bind_input_and_parameters(x, theta)
    """

    n_qubits: int
    circuit: UnboundParametricQuantumCircuit = field(init=False)
    input_functions: dict[int, Callable[[NDArray[np.float64]], float]] = field(
        init=False, default_factory=dict
    )
    _input_parameter_list: list[_InputParameter] = field(init=False, default_factory=list)
    _learning_parameter_list: List[_LearningParameter] = field(init=False, default_factory=list)
    # _gate_list: List[_GateTypes] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.circuit = UnboundParametricQuantumCircuit(self.n_qubits)

    def _new_parameter_position(self) -> int:
        """
        This function does not actually register a new parameter.
        """
        return self.circuit.parameter_count

    def add_gate(self, gate: QuantumGate) -> None:
        """Add arbitrary gate.

        Args:
            gate: Gate to add.
        """
        self.circuit.add_gate(gate)

    def add_X_gate(self, index: int) -> None:
        """
        Args:
            index: Index of qubit to add X gate.
        """
        self.circuit.add_X_gate(index)

    def add_Y_gate(self, index: int) -> None:
        """
        Args:
            index: Index of qubit to add Y gate.
        """
        self.circuit.add_Y_gate(index)

    def add_Z_gate(self, index: int) -> None:
        """
        Args:
            index: Index of qubit to add Z gate.
        """
        self.circuit.add_Z_gate(index)

    def add_RX_gate(self, index: int, angle: float) -> None:
        """
        Args:
            index: Index of qubit to add RX gate.
            angle: Rotation angle.
        """
        self._add_R_gate_inner(index, angle, _Axis.X)

    def add_RY_gate(self, index: int, parameter: float) -> None:
        """
        Args:
            index: Index of qubit to add RY gate.
            angle: Rotation angle.
        """
        self._add_R_gate_inner(index, parameter, _Axis.Y)

    def add_RZ_gate(self, index: int, parameter: float) -> None:
        """
        Args:
            index: Index of qubit to add RZ gate.
            angle: Rotation angle.
        """
        self._add_R_gate_inner(index, parameter, _Axis.Z)

    def add_CNOT_gate(self, control_index: int, target_index: int) -> None:
        """
        Args:
            control_index: Index of control qubit.
            target_index: Index of target qubit.
        """
        self.circuit.add_CNOT_gate(control_index, target_index)

    def add_H_gate(self, index: int) -> None:
        """
        Args:
            index: Index of qubit to put H gate.
        """
        self.circuit.add_H_gate(index)

    def add_input_RX_gate(
        self, qubit: int, input_function: Callable[[NDArray[np.float64]], float]
    ) -> None:
        """Add an RX gate whose angle is determined by input data at inference time.

        Args:
            qubit: Index of the target qubit.
            input_function: Function ``f(x) -> float`` that maps input data to the rotation angle.
        """
        self._add_input_R_gate_inner(qubit, _Axis.X, input_function)

    def add_input_RY_gate(
        self, qubit: int, input_function: Callable[[NDArray[np.float64]], float]
    ) -> None:
        """Add an RY gate whose angle is determined by input data at inference time.

        Args:
            qubit: Index of the target qubit.
            input_function: Function ``f(x) -> float`` that maps input data to the rotation angle.
        """
        self._add_input_R_gate_inner(qubit, _Axis.Y, input_function)

    def add_input_RZ_gate(
        self, qubit: int, input_function: Callable[[NDArray[np.float64]], float]
    ) -> None:
        """Add an RZ gate whose angle is determined by input data at inference time.

        Args:
            qubit: Index of the target qubit.
            input_function: Function ``f(x) -> float`` that maps input data to the rotation angle.
        """
        self._add_input_R_gate_inner(qubit, _Axis.Z, input_function)

    def _add_R_gate_inner(
        self,
        index: int,
        angle: float,
        target: _Axis,
    ) -> None:
        if target == _Axis.X:
            self.circuit.add_RX_gate(index, angle)
        elif target == _Axis.Y:
            self.circuit.add_RY_gate(index, angle)
        elif target == _Axis.Z:
            self.circuit.add_RZ_gate(index, angle)
        else:
            raise NotImplementedError

    def _add_input_R_gate_inner(
        self,
        index: int,
        target: _Axis,
        input_function: Callable[[NDArray[np.float64]], float],
    ):
        new_gate_pos = self._new_parameter_position()
        self._input_parameter_list.append(_InputParameter(new_gate_pos, input_function, None))
        self.input_functions[new_gate_pos] = input_function

        if target == _Axis.X:
            self.circuit.add_ParametricRX_gate(index)
        elif target == _Axis.Y:
            self.circuit.add_ParametricRY_gate(index)
        elif target == _Axis.Z:
            self.circuit.add_ParametricRZ_gate(index)
        else:
            raise ValueError("Invalid target axis")

    def _add_parametric_R_gate_inner(
        self,
        index: int,
        target: _Axis,
        share_with: Optional[int],
        share_with_coef: Optional[float],
    ) -> int:
        new_gate_pos = self._new_parameter_position()

        if share_with is None:
            parameter_id = len(self._learning_parameter_list)
            learning_parameter = _LearningParameter(
                parameter_id,
                0.0,  # initial value; will be overwritten when parameters are bound
            )
            learning_parameter.append_position(new_gate_pos, None)
            self._learning_parameter_list.append(learning_parameter)
        else:
            parameter_id = share_with
            sharing_parameter = self._learning_parameter_list[parameter_id]
            sharing_parameter.append_position(new_gate_pos, share_with_coef)

        if target == _Axis.X:
            self.circuit.add_ParametricRX_gate(index)
        elif target == _Axis.Y:
            self.circuit.add_ParametricRY_gate(index)
        elif target == _Axis.Z:
            self.circuit.add_ParametricRZ_gate(index)

        return parameter_id

    def add_parametric_RX_gate(
        self, qubit: int, share_with: Optional[int] = None, share_with_coef: Optional[float] = None
    ) -> int:
        """Add a trainable RX gate and return its parameter ID.

        Args:
            qubit: Index of the target qubit.
            share_with: If given, this gate shares the learnable parameter with the gate
                whose ``parameter_id`` equals ``share_with``.
            share_with_coef: Coefficient applied to the shared parameter value
                (angle = shared_value * coef). Only used when ``share_with`` is set.

        Returns:
            parameter_id: Index of the learnable parameter assigned to this gate.
        """
        return self._add_parametric_R_gate_inner(qubit, _Axis.X, share_with, share_with_coef)

    def add_parametric_RY_gate(
        self, qubit: int, share_with: Optional[int] = None, share_with_coef: Optional[float] = None
    ) -> int:
        """Add a trainable RY gate and return its parameter ID.

        Args:
            qubit: Index of the target qubit.
            share_with: If given, this gate shares the learnable parameter with the gate
                whose ``parameter_id`` equals ``share_with``.
            share_with_coef: Coefficient applied to the shared parameter value
                (angle = shared_value * coef). Only used when ``share_with`` is set.

        Returns:
            parameter_id: Index of the learnable parameter assigned to this gate.
        """
        return self._add_parametric_R_gate_inner(qubit, _Axis.Y, share_with, share_with_coef)

    def add_parametric_RZ_gate(
        self, qubit: int, share_with: Optional[int] = None, share_with_coef: Optional[float] = None
    ) -> int:
        """Add a trainable RZ gate and return its parameter ID.

        Args:
            qubit: Index of the target qubit.
            share_with: If given, this gate shares the learnable parameter with the gate
                whose ``parameter_id`` equals ``share_with``.
            share_with_coef: Coefficient applied to the shared parameter value
                (angle = shared_value * coef). Only used when ``share_with`` is set.

        Returns:
            parameter_id: Index of the learnable parameter assigned to this gate.
        """
        return self._add_parametric_R_gate_inner(qubit, _Axis.Z, share_with, share_with_coef)

    def add_parametric_multi_Pauli_rotation_gate(
        self, targets: List[int], pauli_ids: List[int]
    ) -> Parameter:
        """Add a trainable multi-qubit Pauli rotation gate.

        Args:
            targets: List of target qubit indices.
            pauli_ids: List of Pauli operator IDs for each target qubit
                (1=X, 2=Y, 3=Z).

        Returns:
            The Parameter object associated with this gate.
        """
        return self.circuit.add_ParametricPauliRotation_gate(targets, pauli_ids)

    def add_parametric_input_RX_gate(
        self, index: int, input_func: InputFuncWithParam = lambda theta, x: x[0]
    ) -> None:
        """Add an RX gate whose angle depends on both a learnable parameter and input data.

        Args:
            index: Index of the target qubit.
            input_func: Function ``f(theta, x) -> float`` that computes the rotation angle
                from the current learnable parameter value and input data.
                Defaults to ``lambda theta, x: x[0]``.
        """
        self._add_parametric_input_R_gate_inner(index, _Axis.X, input_func)

    def add_parametric_input_RY_gate(
        self, index: int, input_func: InputFuncWithParam = lambda theta, x: x[0]
    ) -> None:
        """Add an RY gate whose angle depends on both a learnable parameter and input data.

        Args:
            index: Index of the target qubit.
            input_func: Function ``f(theta, x) -> float`` that computes the rotation angle
                from the current learnable parameter value and input data.
                Defaults to ``lambda theta, x: x[0]``.
        """
        self._add_parametric_input_R_gate_inner(index, _Axis.Y, input_func)

    def add_parametric_input_RZ_gate(
        self, index: int, input_func: InputFuncWithParam = lambda theta, x: x[0]
    ) -> None:
        """Add an RZ gate whose angle depends on both a learnable parameter and input data.

        Args:
            index: Index of the target qubit.
            input_func: Function ``f(theta, x) -> float`` that computes the rotation angle
                from the current learnable parameter value and input data.
                Defaults to ``lambda theta, x: x[0]``.
        """
        self._add_parametric_input_R_gate_inner(index, _Axis.Z, input_func)

    def _add_parametric_input_R_gate_inner(
        self, index: int, target: _Axis, input_func: InputFuncWithParam
    ) -> None:
        new_gate_pos = self._new_parameter_position()
        learning_parameter = _LearningParameter(len(self._learning_parameter_list), 0.0, True)
        learning_parameter.append_position(new_gate_pos, None)
        self._learning_parameter_list.append(learning_parameter)
        self._input_parameter_list.append(
            _InputParameter(new_gate_pos, input_func, learning_parameter.parameter_id)
        )

        if target == _Axis.X:
            self.circuit.add_ParametricRX_gate(index)
        elif target == _Axis.Y:
            self.circuit.add_ParametricRY_gate(index)
        elif target == _Axis.Z:
            self.circuit.add_ParametricRZ_gate(index)
        else:
            raise NotImplementedError

    @property
    def parameter_count(self) -> int:
        """Total number of parametric slots in the underlying circuit (input + learnable)."""
        return self.circuit.parameter_count

    @property
    def input_params_count(self) -> int:
        """Number of input-data-driven parameters."""
        return len(self._input_parameter_list)

    @property
    def learning_params_count(self) -> int:
        """Number of unique learnable parameters (i.e. the length of the theta vector)."""
        return len(self._learning_parameter_list)

    def get_learning_params_indexes(self) -> List[int]:
        """Return circuit-level indices of all learnable parameter slots.
        A single learnable parameter may occupy multiple slots when ``share_with`` is used.

        Returns:
            List of gate-position indices for every learnable slot in the circuit.
        """
        pos: List[int] = []
        for param in self._learning_parameter_list:
            for pos_in_circuit in param.positions_in_circuit:
                pos.append(pos_in_circuit.gate_pos)
        return pos

    def get_minimum_learning_param_indexes(self) -> List[int]:
        """Return the minimal set of circuit-level indices needed to represent all learnable parameters.
        Returns only the first slot for each learnable parameter, ignoring shared duplicates.

        Returns:
            List of gate-position indices, one per unique learnable parameter.
        """
        pos: List[int] = []
        for param in self._learning_parameter_list:
            pos.append(param.positions_in_circuit[0].gate_pos)
        return pos

    def get_input_params_indexes(self) -> List[int]:
        """Return circuit-level indices of all input-data-driven parameter slots.

        Returns:
            List of gate-position indices for every input parameter slot in the circuit.
        """
        pos: List[int] = []
        for param in self._input_parameter_list:
            pos.append(param.pos)
        return pos

    def bind_input_and_parameters(
        self, x: NDArray[np.float64], parameters: NDArray[np.float64]
    ) -> ImmutableBoundParametricQuantumCircuit:
        """Bind input data and learnable parameters to produce a concrete circuit.

        Args:
            x: Input data array.
            parameters: Learnable parameter vector of length ``learning_params_count``.

        Returns:
            A fully bound (non-parametric) quantum circuit.
        """
        bound_parameters = self.generate_bound_params(x, parameters)
        return self.circuit.bind_parameters(bound_parameters)

    def generate_bound_params(
        self, x: NDArray[np.float64], parameters: NDArray[np.float64]
    ) -> Sequence[float]:
        """Compute the full parameter list to bind to the circuit from input data and learnable parameters.

        Args:
            x: Input data array.
            parameters: Learnable parameter vector of length ``learning_params_count``.

        Returns:
            Sequence of float values with length ``parameter_count``, ready to pass to
            ``circuit.bind_parameters()``.
        """
        bound_parameters = [0.0 for _ in range(self.parameter_count)]
        # Learning parameters
        for param in self._learning_parameter_list:
            param_value = parameters[param.parameter_id]
            param.value = param_value
            for pos in param.positions_in_circuit:
                coef = pos.coef if pos.coef is not None else 1.0
                bound_parameters[pos.gate_pos] = param_value * coef
        # Input parameters
        for param in self._input_parameter_list:
            # Input parameter is resolved here (not in update_parameters),
            # because its value depends on the input data `x`.
            angle = 0.0
            # Exactly one branch is taken depending on whether func needs a learning parameter
            if need_learning_parameter_guard(param.func, param.companion_parameter_id):
                # func takes only x (no learning parameter)
                angle: float = param.func(x)
            elif not_needed_learning_parameter_guard(param.func, param.companion_parameter_id):
                # Present to help the type checker narrow companion_parameter_id to int
                if param.companion_parameter_id is None:
                    # * unreachable
                    continue
                theta = self._learning_parameter_list[param.companion_parameter_id]
                angle = param.func(theta.value, x)
                theta.value = angle
            bound_parameters[param.pos] = angle

        return bound_parameters

    def backprop_innner_product(
        self, x: NDArray[np.float64], theta: NDArray[np.float64], state: QulacsQuantumState
    ) -> NDArray[np.float64]:
        """Compute gradients of learnable parameters via qulacs backpropagation using inner product.
        Converts the circuit to qulacs format and calls ``backprop_inner_product``.

        Args:
            x: Input data array.
            theta: Learnable parameter vector of length ``learning_params_count``.
            state: Target qulacs quantum state used in the inner product.

        Returns:
            Gradient array of shape ``(learning_params_count,)``.
        """
        params = self.generate_bound_params(x, theta)
        (qulacs_circuit, param_mapper) = convert_parametric_circuit(self.circuit)
        for i, v in enumerate(param_mapper(params)):
            qulacs_circuit.set_parameter(i, v)
        ret = qulacs_circuit.backprop_inner_product(state)
        ans = np.zeros(self.learning_params_count)
        for param in self._learning_parameter_list:
            if not param.is_input:
                for pos in param.positions_in_circuit:
                    ans[param.parameter_id] += ret[pos.gate_pos] * (pos.coef or 1.0)

        return ans

    def _calc_gradient_observable(
        self,
        generator: _Axis,
        qubit_index: int,
        hamiltonian: Operator,
    ) -> Operator:
        """Calculate the gradient observable O_j = i[G_j, H].

        Args:
            generator: Axis of the generator (X, Y, or Z).
            qubit_index: Index of the qubit the generator acts on.
            hamiltonian: The Hamiltonian operator H.

        Returns:
            The gradient observable operator O_j.
        """
        simbol = {_Axis.X: "X", _Axis.Y: "Y", _Axis.Z: "Z"}[generator]
        generator_operator = Operator({pauli_label(f"{simbol}{qubit_index}"): 0.5})
        observable = 1j * commutator(generator_operator, hamiltonian)
        return observable

    def _get_gate_axis(self, gate: QuantumGate) -> _Axis:
        """Get gate axis by its name

        Args:
            gate (QuantumGate): Target gate

        Returns:
            _Axis: Axis of the gate
        """
        match gate.name:
            case "ParametricRX":
                return _Axis.X
            case "ParametricRY":
                return _Axis.Y
            case "ParametricRZ":
                return _Axis.Z
            case _:
                raise NotImplementedError("Unknown gate type found: ", gate.name)

    def _apply_gates_to_qc(
        self,
        qc: QuantumCircuit,
        gates: Sequence[QuantumGate],
        parameters: Sequence[float],
    ):
        """Apply Gates with Parameters to QuantumCircuit.

        Args:
            qc (QuantumCircuit): Target QuantumCircuit.
            gates (Sequence[QuantumGate]): Sequence of gates to apply.
            parameters (Sequence[float]): Sequence of parameters for the gates.
        """
        i = 0
        for gate in gates:
            if isinstance(gate, QuantumGate):
                qc.add_gate(gate)
            elif isinstance(gate, ParametricQuantumGate):
                param = parameters[i]
                g_axis = self._get_gate_axis(gate)
                g_qubit = gate.target_indices[0]
                match g_axis:
                    case _Axis.X:
                        qc.add_RX_gate(g_qubit, param)
                    case _Axis.Y:
                        qc.add_RY_gate(g_qubit, param)
                    case _Axis.Z:
                        qc.add_RZ_gate(g_qubit, param)
                i += 1
            else:
                raise NotImplementedError("Unknown gate type found: ", gate.name)

    def _get_inverse_gate(self, gate: QuantumGate) -> QuantumGate:
        """Get Inverse Gate

        Args:
            gate (QuantumGate): Target gate to invert.

        Returns:
            QuantumGate: Inverse of the target gate.
        """
        if isinstance(gate, QuantumGate):
            gate_inverse = QuantumGate(
                name=gate.name,
                target_indices=gate.target_indices,
                control_indices=gate.control_indices,
                classical_indices=gate.classical_indices,
                params=[-p for p in gate.params],
                pauli_ids=gate.pauli_ids,
                unitary_matrix=gate.unitary_matrix,
            )
        return gate_inverse

    def _create_hadamard_test_circuit(
        self,
        x,
        theta,
        gate_index: int,
    ) -> QuantumCircuit:
        """Create a circuit for Hadamard test.
        This circuit is used in the Hadamard test to estimate the gradient.

        When differentiating with respect to θj,
        U = U{>j} Uj(θj) U{<j}
        G is the generator of Uj(θj): RX->G=X/2, RY->G=Y/2, RZ->G=Z/2.

        The circuit is constructed as follows:
        U{>j} control{G} U†{>j} U |+ψ〉
        """
        _circuit = QuantumCircuit(self.n_qubits + 1)
        ancilla_index = self.n_qubits
        _circuit.add_H_gate(ancilla_index)
        bound_params = self.generate_bound_params(x, theta)
        gates_length = len(self.circuit.gates)

        # Create original gates (U |+ψ〉)
        self._apply_gates_to_qc(_circuit, self.circuit.gates, bound_params)

        # Apply backward gates (U†{>j})
        gates_backward = []
        params_backward = []
        j = len([_ for _ in self.circuit.gates if isinstance(_, ParametricQuantumGate)])
        for i in range(gates_length - 1, gate_index, -1):
            gate = self.circuit.gates[i]
            if isinstance(gate, QuantumGate):
                gate_inverse = self._get_inverse_gate(gate)
                gates_backward.append(gate_inverse)
            elif isinstance(gate, ParametricQuantumGate):
                gates_backward.append(gate)
                params_backward.append(-bound_params[j - 1])
                j -= 1
        self._apply_gates_to_qc(_circuit, gates_backward, params_backward)

        # Apply controlled gate (control{G})
        gate = self.circuit.gates[gate_index]
        if isinstance(gate, ParametricQuantumGate):
            axis = self._get_gate_axis(gate)
            target_qubit = gate.target_indices[0]
            match axis:
                case _Axis.X:
                    _circuit.add_CNOT_gate(ancilla_index, target_qubit)
                case _Axis.Y:
                    _circuit.add_Sdag_gate(target_qubit)
                    _circuit.add_CNOT_gate(ancilla_index, target_qubit)
                    _circuit.add_S_gate(target_qubit)
                case _Axis.Z:
                    _circuit.add_CZ_gate(ancilla_index, target_qubit)
                case _:
                    raise NotImplementedError

        # Apply forward gates (U{>j})
        gates_forward = []
        params_forward = []
        for i in range(gate_index + 1, gates_length):
            gate = self.circuit.gates[i]
            gates_forward.append(gate)
            if isinstance(gate, ParametricQuantumGate):
                params_forward.append(bound_params[j])
                j += 1
        self._apply_gates_to_qc(_circuit, gates_forward, params_forward)

        return _circuit

    def _calc_hadamard_gradient_observable(self, operator: Operator) -> Operator:
        # O ⊗ Y
        result_terms = {}
        for p1, c1 in operator.items():
            new_label = pauli_label(f"{str(p1)} Y{self.n_qubits}")
            result_terms[new_label] = result_terms.get(new_label, 0) + c1
        return Operator(result_terms)

    def hadamard_gradient(
        self,
        x: NDArray[np.float64],
        theta: NDArray[np.float64],
        operator: Operator,
        estimator: ConcurrentQuantumEstimator,
    ) -> NDArray[np.float64]:
        """Compute gradients of learnable parameters via the Hadamard test.

        For each learnable parameter θ_j, estimates
        ``∂⟨O⟩/∂θ_j = ⟨O ⊗ Y⟩`` on the Hadamard-test circuit,
        where the ancilla qubit is index ``n_qubits``.

        Args:
            x: Input data array.
            theta: Learnable parameter vector of length ``learning_params_count``.
            operator: Observable whose expectation value gradient is computed.
            estimator: Concurrent quantum estimator used to evaluate the Hadamard-test circuits.

        Returns:
            Gradient array of shape ``(learning_params_count,)``.
        """
        # Calculate operator for hadamard test
        operator = self._calc_hadamard_gradient_observable(operator)

        # Learning Param indexes
        learning_param_indexes = self.get_learning_params_indexes()

        # Calculate gradient for each learning parameter
        _generalCircuitQuantumStates = []
        param_gate_count = -1
        for i, gate in enumerate(self.circuit.gates):
            # Skip non-parametric gates
            if not isinstance(gate, ParametricQuantumGate):
                continue

            # Skip input parameters
            param_gate_count += 1
            if param_gate_count not in learning_param_indexes:
                continue

            _circuit = self._create_hadamard_test_circuit(x, theta, i)
            _generalCircuitQuantumStates.append(
                GeneralCircuitQuantumState(self.n_qubits + 1, _circuit)
            )

        operators = [operator] * len(_generalCircuitQuantumStates)
        results = estimator(operators, _generalCircuitQuantumStates)

        return np.array([res.value for res in results])

    def to_batched(
        self, data: NDArray[np.float64], parameters: NDArray[np.float64]
    ) -> Tuple[UnboundParametricQuantumCircuit, NDArray[np.float64]]:
        """Build a batched parameter array for use with scaluq (quri-parts-scaluq).

        Args:
            data: Input data array of shape ``(n_data, n_features)``.
            parameters: Learnable parameter vector of shape ``(n_params,)``.

        Returns:
            Tuple of ``(circuit, batched_params)`` where ``batched_params`` has shape
            ``(n_data, parameter_count)`` with each row ready to bind to the circuit.
        """
        batched_params = np.zeros((len(data), self.parameter_count))
        # Learning parameters
        for param in self._learning_parameter_list:
            param.value = parameters[param.parameter_id]
            for pos in param.positions_in_circuit:
                batched_params[:, pos.gate_pos] = param.value
        # Input parameters
        for i, x in enumerate(data):
            for param in self._input_parameter_list:
                angle = 0.0
                if need_learning_parameter_guard(param.func, param.companion_parameter_id):
                    angle = param.func(x)
                elif not_needed_learning_parameter_guard(param.func, param.companion_parameter_id):
                    if param.companion_parameter_id is None:
                        # * unreachable
                        continue
                    theta = self._learning_parameter_list[param.companion_parameter_id]
                    angle = param.func(theta.value, x)
                batched_params[i, param.pos] = angle
        return self.circuit, batched_params

    def to_batched_for_gradient(
        self,
        data: NDArray[np.float64],
        parameters: NDArray[np.float64],
        delta: float = 1e-5,
    ) -> Tuple[UnboundParametricQuantumCircuit, NDArray[np.float64]]:
        """Build shifted parameter arrays for numerical gradient estimation.

        For each sample and each learning parameter, creates two rows with the
        parameter shifted by +delta/2 and -delta/2. The resulting array has shape
        ``(n_samples * 2 * learning_params_count, parameter_count)``.

        Row layout: for sample i and learning param j,
          plus  row = i * 2 * learning_params_count + 2 * j
          minus row = i * 2 * learning_params_count + 2 * j + 1

        Args:
            data: Input data array of shape ``(n_data, n_features)``.
            parameters: Learnable parameter vector of shape ``(learning_params_count,)``.
            delta: Finite difference step size.

        Returns:
            Tuple of ``(circuit, shifted_params)`` where ``shifted_params`` has shape
            ``(n_data * 2 * learning_params_count, parameter_count)``.
        """
        n_samples = len(data)
        n_learning = self.learning_params_count
        total = n_samples * 2 * n_learning

        # Build base params once via to_batched: (n_samples, parameter_count)
        _, base_params = self.to_batched(data, parameters)

        # Repeat each sample's base params 2*n_learning times
        # base_params[s] -> shifted_params[s*2*n_learning .. (s+1)*2*n_learning - 1]
        shifted_params = np.repeat(base_params, 2 * n_learning, axis=0)

        # Build shift vectors: for each learning param j, get its circuit positions and coefs
        half_delta = delta / 2
        for j, lp in enumerate(self._learning_parameter_list):
            for pos in lp.positions_in_circuit:
                coef = pos.coef if pos.coef is not None else 1.0
                shift = half_delta * coef
                # Apply +shift to plus rows, -shift to minus rows for all samples
                for s in range(n_samples):
                    base = s * 2 * n_learning
                    shifted_params[base + 2 * j, pos.gate_pos] += shift
                    shifted_params[base + 2 * j + 1, pos.gate_pos] -= shift

        return self.circuit, shifted_params


def preprocess_x(x: NDArray[np.float64], i: int) -> float:
    a: float = x[i % len(x)]
    return a


if __name__ == "__main__":
    n_qubits = 3
    circuit = LearningCircuit(n_qubits)
    for i in range(n_qubits):
        circuit.add_input_RX_gate(i, lambda x, i=i: np.arcsin(preprocess_x(x, i)))
    circuit.add_parametric_RX_gate(0)
    circuit.add_parametric_RX_gate(1)
    bind_circuit = circuit.bind_input_and_parameters(
        np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5])
    )
    print(circuit.circuit.gates)
    print(circuit.circuit.param_mapping.in_params)
    for gate in bind_circuit.gates:
        print(gate)
