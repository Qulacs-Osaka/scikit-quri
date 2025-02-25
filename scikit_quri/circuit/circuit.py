from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional, Self, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from qulacs import QuantumState as QulacsQuantumState
from quri_parts.circuit import (
    ImmutableBoundParametricQuantumCircuit,
    QuantumGate,
    UnboundParametricQuantumCircuit,
    Parameter,
)
from quri_parts.qulacs.circuit import convert_parametric_circuit
from typing_extensions import deprecated


class _Axis(Enum):
    """Specifying axis. Used in inner private method in LearningCircuit."""

    X = auto()
    Y = auto()
    Z = auto()


class _GateTypes(Enum):
    """Specifying gate types. Used in inner private method in LearningCircuit."""

    Primitive = auto()
    Input = auto()
    Learning = auto()


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
        self._add_input_R_gate_inner(qubit, _Axis.X, input_function)

    def add_input_RY_gate(
        self, qubit: int, input_function: Callable[[NDArray[np.float64]], float]
    ) -> None:
        self._add_input_R_gate_inner(qubit, _Axis.Y, input_function)

    def add_input_RZ_gate(
        self, qubit: int, input_function: Callable[[NDArray[np.float64]], float]
    ) -> None:
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
        self, index: int, target: _Axis, share_with: Optional[int], share_with_coef: Optional[float]
    ) -> int:
        new_gate_pos = self._new_parameter_position()

        if share_with is None:
            parameter_id = len(self._learning_parameter_list)
            learning_parameter = _LearningParameter(
                parameter_id,
                # 仮置きで0.0にしてます
                0.0,
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
        return self._add_parametric_R_gate_inner(qubit, _Axis.X, share_with, share_with_coef)

    def add_parametric_RY_gate(
        self, qubit: int, share_with: Optional[int] = None, share_with_coef: Optional[float] = None
    ) -> int:
        return self._add_parametric_R_gate_inner(qubit, _Axis.Y, share_with, share_with_coef)

    def add_parametric_RZ_gate(
        self, qubit: int, share_with: Optional[int] = None, share_with_coef: Optional[float] = None
    ) -> int:
        return self._add_parametric_R_gate_inner(qubit, _Axis.Z, share_with, share_with_coef)

    def add_parametric_multi_Pauli_rotation_gate(
        self, targets: List[int], pauli_ids: List[int]
    ) -> Parameter:
        return self.circuit.add_ParametricPauliRotation_gate(targets, pauli_ids)

    @property
    def parameter_count(self) -> int:
        return self.circuit.parameter_count
        # return len(self._input_parameter_list) + len(self._learning_parameter_list)
        # return self.n_parameters - len(self.input_functions)

    @property
    def input_params_count(self) -> int:
        return len(self._input_parameter_list)

    @property
    def learning_params_count(self) -> int:
        return len(self._learning_parameter_list)
        # return self.n_learning_params

    def get_learning_param_indexes(self) -> List[int]:
        pos = []
        for param in self._learning_parameter_list:
            for pos_in_circuit in param.positions_in_circuit:
                pos.append(pos_in_circuit.gate_pos)
        return np.array(pos)

    def get_minimum_learning_param_indexes(self) -> List[int]:
        # Circuit内のパラメータのうち，Circuitを構成できる最小のパラメータのインデックスを返す
        pos = []
        for param in self._learning_parameter_list:
            pos.append(param.positions_in_circuit[0].gate_pos)
        return np.array(pos)

    def get_input_params_indexes(self) -> List[int]:
        pos = []
        for param in self._input_parameter_list:
            pos.append(param.pos)
        return np.array(pos)
        # input_param_mask = list(
        #     map(lambda gate: 1 if gate == _GateTypes.Input else None, self._gate_list)
        # )
        # return list(filter(lambda i: input_param_mask[i] != None, range(self.n_parameters)))

    # def get_input_params(self) -> List[float]:
    #     parametric_gates = list(filter(lambda x:isinstance(x[0],ParametricQuantumGate),self.circuit.gates_and_params))
    #     print(parametric_gates[0][1])
    #     return [self.input_functions[i] for i in self.get_input_params_indexes()]

    def update_parameters(
        self, parameters: NDArray[np.float64]
    ) -> ImmutableBoundParametricQuantumCircuit:
        return self.circuit.bind_parameters(parameters)

    def bind_input_and_parameters(
        self, x: NDArray[np.float64], parameters: NDArray[np.float64]
    ) -> ImmutableBoundParametricQuantumCircuit:
        bound_parameters = self.generate_bound_params(x, parameters)
        return self.circuit.bind_parameters(bound_parameters)

    # def generate_bound_params(
    #     self, x: NDArray[np.float64], learning_params: NDArray[np.float64]
    # ) -> NDArray[np.float64]:
    #     bound_parameters = []
    #     input_index = 0
    #     parameter_index = 0
    #     # TODO share_with未実装
    #     if len(learning_params) != self.n_learning_params:
    #         raise ValueError("Invalid number of learning parameters")
    #     for i in range(self.n_parameters):
    #         gate_type = self._gate_list[i]
    #         if gate_type == _GateTypes.Input:
    #             input_function = self.input_functions.get(input_index)
    #             bound_parameters.append(input_function(x))
    #             input_index += 1
    #         if gate_type == _GateTypes.Learning:
    #             bound_parameters.append(learning_params[parameter_index])
    #             parameter_index += 1
    #     return bound_parameters

    def generate_bound_params(self, x: NDArray[np.float64], theta: NDArray[np.float64]):
        bound_parameters = np.zeros(self.parameter_count)
        # Learning parameters
        for param in self._learning_parameter_list:
            param_value = theta[param.parameter_id]
            param.value = param_value
            for pos in param.positions_in_circuit:
                bound_parameters[pos.gate_pos] = param_value
        # Input parameters
        for param in self._input_parameter_list:
            # Input parameter is updated here, not update_parameters(),
            # because input parameter is determined with the input data `x`.
            if param.companion_parameter_id is None:
                # If `companion_parameter_id` is `None`, `func` does not need a learning parameter.
                angle = param.func(x)
            else:
                theta: _LearningParameter = self._learning_parameter_list[
                    param.companion_parameter_id
                ]
                angle = param.func(theta.value, x)
                theta.value = angle
            bound_parameters[param.pos] = angle

        return bound_parameters

    def backprop_innner_product(
        self, x: NDArray[np.float64], theta: NDArray[np.float64], state: QulacsQuantumState
    ) -> NDArray[np.float64]:
        """
        backprop(self, x: List[float],  state)->List[Float]
        qulacsに回路を変換しinner_productでbackpropします。
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

    def to_batched(
        self, data: NDArray[np.float64], theta: NDArray[np.float64]
    ) -> Tuple[UnboundParametricQuantumCircuit, NDArray[np.float64]]:
        """
        scaluq(quri-parts-scaluq)に流すためのMethod
        data: (size, n_features)
        theta: (n_params)

        Returns:
            (circuit, batched_params): (UnboundParametricQuantumCircuit, (size, parameter_count))
        """
        batched_params = np.zeros((len(data), self.parameter_count))
        # Learning parameters
        for param in self._learning_parameter_list:
            param.value = theta[param.parameter_id]
            for pos in param.positions_in_circuit:
                batched_params[:, pos.gate_pos] = param.value
        # Input parameters
        for i, x in enumerate(data):
            for param in self._input_parameter_list:
                if param.companion_parameter_id is None:
                    angle = param.func(x)
                else:
                    theta: _LearningParameter = self._learning_parameter_list[
                        param.companion_parameter_id
                    ]
                    angle = param.func(theta.value, x)
                    theta.value = angle
                batched_params[i, param.pos] = angle
        return self.circuit, batched_params


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
