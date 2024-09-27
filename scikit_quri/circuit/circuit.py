from dataclasses import dataclass, field
from typing import Callable, List

from enum import Enum, auto
import numpy as np
from numpy.typing import NDArray
from quri_parts.circuit import (
    UnboundParametricQuantumCircuit,
    ImmutableBoundParametricQuantumCircuit,
)


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


@dataclass
class LearningCircuit:
    n_qubits: int
    n_parameters: int = field(init=False, default=0)
    n_learning_params: int = field(init=False, default=0)
    circuit: UnboundParametricQuantumCircuit = field(init=False)
    input_functions: dict[int, Callable[[NDArray[np.float_]], float]] = field(
        init=False, default_factory=dict
    )
    _input_parameter_list: list[int] = field(init=False, default_factory=list)
    _gate_list: List[_GateTypes] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.circuit = UnboundParametricQuantumCircuit(self.n_qubits)

    def _new_parameter_position(self) -> int:
        """
        Return a position of a new parameter to be registered to `ParametricQuantumCircuit`.
        This function does not actually register a new parameter.
        """
        return self.circuit.parameter_count

    def add_input_RX_gate(
        self, qubit: int, input_function: Callable[[NDArray[np.float_]], float]
    ) -> None:
        self._add_input_R_gate_inner(qubit, _Axis.X, input_function)

    def add_input_RY_gate(
        self, qubit: int, input_function: Callable[[NDArray[np.float_]], float]
    ) -> None:
        self._add_input_R_gate_inner(qubit, _Axis.Y, input_function)

    def add_input_RZ_gate(
        self, qubit: int, input_function: Callable[[NDArray[np.float_]], float]
    ) -> None:
        self._add_input_R_gate_inner(qubit, _Axis.Z, input_function)

    def add_gate(self, gate) -> None:
        """Add arbitrary gate.

        Args:
            gate: Gate to add.
        """
        self.circuit.add_gate(gate)

    def _add_input_R_gate_inner(
        self,
        index: int,
        target: _Axis,
        input_function: Callable[[NDArray[np.float_]], float],
    ):
        self._gate_list.append(_GateTypes.Input)

        pos = self._new_parameter_position()
        self.input_functions[pos] = input_function
        self.n_parameters += 1
        if target == _Axis.X:
            self.circuit.add_ParametricRX_gate(index)
        elif target == _Axis.Y:
            self.circuit.add_ParametricRY_gate(index)
        elif target == _Axis.Z:
            self.circuit.add_ParametricRZ_gate(index)
        else:
            raise ValueError("Invalid target axis")

    def add_parametric_RX_gate(self, qubit: int) -> None:
        self.circuit.add_ParametricRX_gate(qubit)
        self.n_parameters += 1
        self.n_learning_params += 1
        self._gate_list.append(_GateTypes.Learning)

    def add_parametric_RY_gate(self, qubit: int) -> None:
        self.circuit.add_ParametricRY_gate(qubit)
        self.n_parameters += 1
        self.n_learning_params += 1
        self._gate_list.append(_GateTypes.Learning)

    def add_parametric_RZ_gate(self, qubit: int) -> None:
        self.circuit.add_ParametricRZ_gate(qubit)
        self.n_parameters += 1
        self.n_learning_params += 1
        self._gate_list.append(_GateTypes.Learning)

    @property
    def parameter_count(self) -> int:
        return self.n_parameters - len(self.input_functions)

    @property
    def learning_params_count(self) -> int:
        return self.n_learning_params

    def get_learning_param_indexes(self) -> List[int]:
        learning_param_mask = list(
            map(lambda gate: 1 if gate == _GateTypes.Learning else None, self._gate_list)
        )
        return list(filter(lambda i: learning_param_mask[i] != None, range(self.n_parameters)))
        # return [i for i, gate_type in enumerate(self._gate_list) if gate_type == _GateTypes.Learning]

    def get_input_params_indexes(self) -> List[int]:
        input_param_mask = list(
            map(lambda gate: 1 if gate == _GateTypes.Input else None, self._gate_list)
        )
        return list(filter(lambda i: input_param_mask[i] != None, range(self.n_parameters)))

    # def get_input_params(self) -> List[float]:
    #     parametric_gates = list(filter(lambda x:isinstance(x[0],ParametricQuantumGate),self.circuit.gates_and_params))
    #     print(parametric_gates[0][1])
    #     return [self.input_functions[i] for i in self.get_input_params_indexes()]

    def bind_input_and_parameters(
        self, x: NDArray[np.float_], parameters: NDArray[np.float_]
    ) -> ImmutableBoundParametricQuantumCircuit:
        bound_parameters = []
        parameter_index = 0
        for i in range(self.n_parameters):
            input_function = self.input_functions.get(i)
            if input_function is None:
                bound_parameters.append(parameters[parameter_index])
                parameter_index += 1
            else:
                bound_parameters.append(input_function(x))
        return self.circuit.bind_parameters(bound_parameters)

    def generate_bound_params(
        self, x: NDArray[np.float_], learning_params: NDArray[np.float_]
    ) -> NDArray[np.float_]:
        bound_parameters = []
        input_index = 0
        parameter_index = 0
        if len(learning_params) != self.n_learning_params:
            raise ValueError("Invalid number of learning parameters")
        for i in range(self.n_parameters):
            gate_type = self._gate_list[i]
            if gate_type == _GateTypes.Input:
                input_function = self.input_functions.get(input_index)
                bound_parameters.append(input_function(x))
                input_index += 1
            if gate_type == _GateTypes.Learning:
                bound_parameters.append(learning_params[parameter_index])
                parameter_index += 1
        return bound_parameters


def preprocess_x(x: NDArray[np.float_], i: int) -> float:
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
