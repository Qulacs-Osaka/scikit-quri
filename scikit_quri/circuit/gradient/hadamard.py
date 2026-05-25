"""Hadamard-test gradient for LearningCircuit.

For each learnable parameter θ_j we estimate ``∂⟨O⟩/∂θ_j = ⟨O ⊗ Y_anc⟩`` on a
Hadamard-test circuit. The ancilla is the (``n_qubits``)-th qubit. Only RX/RY/RZ
parametric rotations are supported (the only parametric gate types LearningCircuit
itself emits via ``add_parametric_R*_gate``).
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, List, Sequence

import numpy as np
from numpy.typing import NDArray
from quri_parts.circuit import ParametricQuantumGate, QuantumCircuit, QuantumGate
from quri_parts.core.estimator import ConcurrentQuantumEstimator
from quri_parts.core.operator import Operator, pauli_label
from quri_parts.core.state import GeneralCircuitQuantumState

if TYPE_CHECKING:
    from ..circuit import LearningCircuit


class _Axis(Enum):
    X = auto()
    Y = auto()
    Z = auto()


def _gate_axis(gate: QuantumGate) -> _Axis:
    match gate.name:
        case "ParametricRX":
            return _Axis.X
        case "ParametricRY":
            return _Axis.Y
        case "ParametricRZ":
            return _Axis.Z
        case _:
            raise NotImplementedError(f"Unsupported parametric gate for Hadamard test: {gate.name}")


def _apply_gates(
    qc: QuantumCircuit,
    gates: Sequence[QuantumGate],
    parameters: Sequence[float],
) -> None:
    """Append ``gates`` to ``qc``, substituting concrete angles for any parametric gates.

    ``parameters[k]`` is the angle for the k-th parametric gate encountered in
    ``gates`` (in order).
    """
    i = 0
    for gate in gates:
        if isinstance(gate, QuantumGate):
            qc.add_gate(gate)
        elif isinstance(gate, ParametricQuantumGate):
            param = parameters[i]
            axis = _gate_axis(gate)
            qubit = gate.target_indices[0]
            match axis:
                case _Axis.X:
                    qc.add_RX_gate(qubit, param)
                case _Axis.Y:
                    qc.add_RY_gate(qubit, param)
                case _Axis.Z:
                    qc.add_RZ_gate(qubit, param)
            i += 1
        else:
            raise NotImplementedError(f"Unknown gate type: {gate.name}")


def _invert_gate(gate: QuantumGate) -> QuantumGate:
    return QuantumGate(
        name=gate.name,
        target_indices=gate.target_indices,
        control_indices=gate.control_indices,
        classical_indices=gate.classical_indices,
        params=[-p for p in gate.params],
        pauli_ids=gate.pauli_ids,
        unitary_matrix=gate.unitary_matrix,
    )


def _hadamard_observable(operator: Operator, ancilla_qubit: int) -> Operator:
    """O ⊗ Y on the ancilla qubit."""
    result_terms: dict = {}
    for p, c in operator.items():
        new_label = pauli_label(f"{str(p)} Y{ancilla_qubit}")
        result_terms[new_label] = result_terms.get(new_label, 0) + c
    return Operator(result_terms)


def _hadamard_test_circuit(
    circuit: "LearningCircuit",
    x: NDArray[np.float64],
    theta: NDArray[np.float64],
    gate_index: int,
) -> QuantumCircuit:
    """Build the Hadamard-test circuit for differentiating w.r.t. the parametric gate at ``gate_index``.

    The circuit applies U |+ψ〉, then U†{>j}, then a controlled generator,
    then U{>j}. The ancilla qubit is index ``n_qubits``.
    """
    n_qubits = circuit.n_qubits
    inner_gates = circuit.circuit.gates
    bound_params = circuit.generate_bound_params(x, theta)
    test = QuantumCircuit(n_qubits + 1)
    ancilla = n_qubits
    test.add_H_gate(ancilla)

    # U |+ψ〉
    _apply_gates(test, inner_gates, bound_params)

    # U†{>j}
    gates_backward: List[QuantumGate] = []
    params_backward: List[float] = []
    j = sum(1 for g in inner_gates if isinstance(g, ParametricQuantumGate))
    for i in range(len(inner_gates) - 1, gate_index, -1):
        gate = inner_gates[i]
        if isinstance(gate, QuantumGate):
            gates_backward.append(_invert_gate(gate))
        elif isinstance(gate, ParametricQuantumGate):
            gates_backward.append(gate)
            params_backward.append(-bound_params[j - 1])
            j -= 1
    _apply_gates(test, gates_backward, params_backward)

    # controlled{G}
    target_gate = inner_gates[gate_index]
    if isinstance(target_gate, ParametricQuantumGate):
        axis = _gate_axis(target_gate)
        target_qubit = target_gate.target_indices[0]
        match axis:
            case _Axis.X:
                test.add_CNOT_gate(ancilla, target_qubit)
            case _Axis.Y:
                test.add_Sdag_gate(target_qubit)
                test.add_CNOT_gate(ancilla, target_qubit)
                test.add_S_gate(target_qubit)
            case _Axis.Z:
                test.add_CZ_gate(ancilla, target_qubit)

    # U{>j}
    gates_forward: List[QuantumGate] = []
    params_forward: List[float] = []
    for i in range(gate_index + 1, len(inner_gates)):
        gate = inner_gates[i]
        gates_forward.append(gate)
        if isinstance(gate, ParametricQuantumGate):
            params_forward.append(bound_params[j])
            j += 1
    _apply_gates(test, gates_forward, params_forward)

    return test


def hadamard_gradient(
    circuit: "LearningCircuit",
    x: NDArray[np.float64],
    theta: NDArray[np.float64],
    operator: Operator,
    estimator: ConcurrentQuantumEstimator,
) -> NDArray[np.float64]:
    """Compute gradients of learnable parameters via the Hadamard test.

    For each learnable parameter θ_j, estimates ``∂⟨O⟩/∂θ_j = ⟨O ⊗ Y_anc⟩``
    on the Hadamard-test circuit (ancilla = qubit ``n_qubits``).

    Args:
        circuit: The learning circuit whose parameters are differentiated.
        x: Input data array.
        theta: Learnable parameter vector of length ``circuit.learning_params_count``.
        operator: Observable whose expectation value gradient is computed.
        estimator: Concurrent quantum estimator used to evaluate the Hadamard-test circuits.

    Returns:
        Gradient array of shape ``(learning_params_count,)``.
    """
    observable = _hadamard_observable(operator, ancilla_qubit=circuit.n_qubits)
    learning_param_indexes = circuit.get_learning_params_indexes()

    states = []
    param_gate_count = -1
    for i, gate in enumerate(circuit.circuit.gates):
        if not isinstance(gate, ParametricQuantumGate):
            continue
        param_gate_count += 1
        if param_gate_count not in learning_param_indexes:
            continue
        test_circuit = _hadamard_test_circuit(circuit, x, theta, i)
        states.append(GeneralCircuitQuantumState(circuit.n_qubits + 1, test_circuit))

    operators = [observable] * len(states)
    results = estimator(operators, states)
    return np.array([res.value for res in results])
