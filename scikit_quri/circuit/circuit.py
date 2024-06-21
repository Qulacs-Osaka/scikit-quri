from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from quri_parts.circuit import UnboundParametricQuantumCircuit


@dataclass
class LearningCircuit:
    n_qubits: int
    n_parameters: int = field(init=False, default=0)
    circuit: UnboundParametricQuantumCircuit = field(init=False)
    input_functions: dict[int, Callable[[NDArray[np.float_]], float]] = field(
        init=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        self.circuit = UnboundParametricQuantumCircuit(self.n_qubits)

    def add_input_RX_gate(
        self, qubit: int, input_function: Callable[[NDArray[np.float_]], float]
    ) -> None:
        self.circuit.add_ParametricRX_gate(qubit)
        self.input_functions[qubit] = input_function
        self.n_parameters += 1

    def add_parametric_RX_gate(self, qubit: int) -> None:
        self.circuit.add_ParametricRX_gate(qubit)
        self.n_parameters += 1

    def add_input_RY_gate(
        self, qubit: int, input_function: Callable[[NDArray[np.float_]], float]
    ) -> None:
        self.circuit.add_ParametricRY_gate(qubit)
        self.input_functions[qubit] = input_function
        self.n_parameters += 1

    def add_parametric_RY_gate(self, qubit: int) -> None:
        self.circuit.add_ParametricRY_gate(qubit)
        self.n_parameters += 1

    def add_input_RZ_gate(
        self, qubit: int, input_function: Callable[[NDArray[np.float_]], float]
    ) -> None:
        self.circuit.add_ParametricRZ_gate(qubit)
        self.input_functions[qubit] = input_function
        self.n_parameters += 1

    def add_parametric_RZ_gate(self, qubit: int) -> None:
        self.circuit.add_ParametricRZ_gate(qubit)
        self.n_parameters += 1
    
    def bind_input_and_parameters(
        self, x: NDArray[np.float_], parameters: NDArray[np.float_]
    ) -> None:
        bound_parameters = []
        for i in range(self.n_parameters):
            input_function = self.input_functions.get(i)
            if input_function is None:
                bound_parameters.append(parameters[i])
            else:
                bound_parameters.append(input_function(x))
        self.circuit.bind_parameters(bound_parameters)


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
    circuit.bind_input_and_parameters(np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5]))
    print(circuit.circuit.gates)
    print(circuit.circuit.param_mapping.in_params)
