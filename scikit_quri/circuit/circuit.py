from dataclasses import dataclass, field
from typing import Callable

from enum import Enum,auto
import numpy as np
from numpy.typing import NDArray
from quri_parts.circuit import UnboundParametricQuantumCircuit, ImmutableBoundParametricQuantumCircuit

class _Axis(Enum):
    """Specifying axis. Used in inner private method in LearningCircuit."""
    X = auto()
    Y = auto()
    Z = auto()

@dataclass
class LearningCircuit:
    n_qubits: int
    n_parameters: int = field(init=False, default=0)
    n_thetas: int = field(init=False, default=0)
    circuit: UnboundParametricQuantumCircuit = field(init=False)
    input_functions: dict[int, Callable[[NDArray[np.float_]], float]] = field(
        init=False, default_factory=dict
    )
    _input_parameter_list: list[int] = field(init=False, default_factory=list)

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

    def _add_input_R_gate_inner(
        self,
        index: int,
        target: _Axis,
        input_function: Callable[[NDArray[np.float_]], float]):

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
        self.n_thetas += 1


    def add_parametric_RY_gate(self, qubit: int) -> None:
        self.circuit.add_ParametricRY_gate(qubit)
        self.n_parameters += 1
        self.n_thetas += 1


    def add_parametric_RZ_gate(self, qubit: int) -> None:
        self.circuit.add_ParametricRZ_gate(qubit)
        self.n_parameters += 1
        self.n_thetas += 1
    
    @property
    def parameter_count(self) -> int:
        return self.n_parameters - len(self.input_functions)
    
    @property
    def theta_count(self) -> int:
        return self.n_thetas
    
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
    
    def generate_bound_params(self,x: NDArray[np.float_],theta: NDArray[np.float_]) -> NDArray[np.float_]:
        bound_parameters = []
        parameter_index = 0
        for i in range(self.n_parameters):
            input_function = self.input_functions.get(i)
            if input_function is None:
                bound_parameters.append(theta[parameter_index])
                parameter_index += 1
            else:
                bound_parameters.append(input_function(x))
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
    bind_circuit = circuit.bind_input_and_parameters(np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5]))
    print(circuit.circuit.gates)
    print(circuit.circuit.param_mapping.in_params)
    for gate in bind_circuit.gates:
        print(gate)
