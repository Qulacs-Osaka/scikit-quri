# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Callable, Union, cast
import numpy as np
from numpy.typing import ArrayLike
from typing_extensions import assert_never

if TYPE_CHECKING:
    from scaluq.default.f64 import Circuit as ScaluqCircuit
from quri_parts.circuit import (
    ImmutableLinearMappedParametricQuantumCircuit,
    ImmutableParametricQuantumCircuit,
    ImmutableQuantumCircuit,
    ParametricQuantumCircuitProtocol,
    QuantumGate,
    ParametricQuantumGate,
    gate_names,
)

from quri_parts.circuit.gate_names import (
    MultiQubitGateNameType,
    SingleQubitGateNameType,
    ThreeQubitGateNameType,
    TwoQubitGateNameType,
    ParametricGateNameType,
    is_gate_name,
    is_multi_qubit_gate_name,
    is_parametric_gate_name,
    is_single_qubit_gate_name,
    is_three_qubit_gate_name,
    is_two_qubit_gate_name,
    is_unitary_matrix_gate_name,
)

from .. import cast_to_list, _backend


_single_qubit_gate_scaluq: Mapping[SingleQubitGateNameType, Callable[[int], _backend.Gate]] = {
    gate_names.Identity: _backend.gate.I,
    gate_names.X: _backend.gate.X,
    gate_names.Y: _backend.gate.Y,
    gate_names.Z: _backend.gate.Z,
    gate_names.H: _backend.gate.H,
    gate_names.S: _backend.gate.S,
    gate_names.Sdag: _backend.gate.Sdag,
    gate_names.SqrtX: _backend.gate.SqrtX,
    gate_names.SqrtXdag: _backend.gate.SqrtXdag,
    gate_names.SqrtY: _backend.gate.SqrtY,
    gate_names.SqrtYdag: _backend.gate.SqrtYdag,
    gate_names.T: _backend.gate.T,
    gate_names.Tdag: _backend.gate.Tdag,
}


def _u1_gate_scaluq(gate: QuantumGate) -> _backend.Gate:
    return cast(
        _backend.Gate,
        _backend.gate.U1(*gate.target_indices, *gate.params),
    )


def _u2_gate_scaluq(gate: QuantumGate) -> _backend.Gate:
    return cast(
        _backend.Gate,
        _backend.gate.U2(*gate.target_indices, *gate.params),
    )


def _u3_gate_scaluq(gate: QuantumGate) -> _backend.Gate:
    return cast(
        _backend.Gate,
        _backend.gate.U3(*gate.target_indices, *gate.params),
    )


_single_qubit_reverse_rotation_gate_scaluq: Mapping[
    SingleQubitGateNameType, Callable[[int, float], _backend.Gate]
] = {
    gate_names.RX: _backend.gate.RX,
    gate_names.RY: _backend.gate.RY,
    gate_names.RZ: _backend.gate.RZ,
}


_two_qubit_gate_scaluq: Mapping[TwoQubitGateNameType, Callable[[int, int], _backend.Gate]] = {
    gate_names.CNOT: _backend.gate.CX,
    gate_names.CZ: _backend.gate.CZ,
    gate_names.SWAP: _backend.gate.Swap,
}


_three_qubit_gate_scaluq: Mapping[
    ThreeQubitGateNameType, Callable[[int, int, int], _backend.Gate]
] = {
    gate_names.TOFFOLI: _backend.gate.Toffoli,
}


_multi_pauli_gate_scaluq: Mapping[
    MultiQubitGateNameType,
    Callable[[_backend.PauliOperator], _backend.Gate],
] = {
    gate_names.Pauli: _backend.gate.Pauli,
}


_multi_pauli_rotation_gate_scaluq: Mapping[
    MultiQubitGateNameType, Callable[[_backend.PauliOperator, float], _backend.Gate]
] = {
    gate_names.PauliRotation: _backend.gate.PauliRotation,
}


_single_param_gate_scaluq: Mapping[
    ParametricGateNameType, Callable[[int, float], _backend.Gate]
] = {
    gate_names.ParametricRX: _backend.gate.ParamRX,
    gate_names.ParametricRY: _backend.gate.ParamRY,
    gate_names.ParametricRZ: _backend.gate.ParamRZ,
}


def dense_matrix_gate_scaluq(
    targets: Union[int, Sequence[int]], unitary_matrix: ArrayLike
) -> _backend.Gate:
    if isinstance(targets, int):
        targets = [targets]
    unitary_matrix = np.array(unitary_matrix, dtype=np.complex64)
    return _backend.gate.DenseMatrix(targets, unitary_matrix)


def convert_gate(
    gate: QuantumGate,
) -> _backend.Gate:
    if not is_gate_name(gate.name):
        raise ValueError(f"Unknown gate name: {gate.name}")

    if is_single_qubit_gate_name(gate.name):
        if gate.name in _single_qubit_gate_scaluq:
            return _single_qubit_gate_scaluq[gate.name](*gate.target_indices, *gate.params)
        elif gate.name == gate_names.U1:
            return _u1_gate_scaluq(gate)
        elif gate.name == gate_names.U2:
            return _u2_gate_scaluq(gate)
        elif gate.name == gate_names.U3:
            return _u3_gate_scaluq(gate)
        elif gate.name in _single_qubit_reverse_rotation_gate_scaluq:
            return _single_qubit_reverse_rotation_gate_scaluq[gate.name](
                *gate.target_indices, *gate.params
            )
        else:
            assert False, "Unreachable"
    elif is_two_qubit_gate_name(gate.name):
        return _two_qubit_gate_scaluq[gate.name](*gate.control_indices, *gate.target_indices)
    elif is_three_qubit_gate_name(gate.name):
        return _three_qubit_gate_scaluq[gate.name](*gate.control_indices, *gate.target_indices)
    elif is_multi_qubit_gate_name(gate.name):
        target_indices = cast_to_list(gate.target_indices)
        pauli_ids = cast_to_list(gate.pauli_ids)
        if gate.name in _multi_pauli_gate_scaluq:
            pauli = _backend.PauliOperator(target_indices, pauli_ids)
            return _multi_pauli_gate_scaluq[gate.name](pauli)
        elif gate.name in _multi_pauli_rotation_gate_scaluq:
            pauli = _backend.PauliOperator(target_indices, pauli_ids)
            angle = gate.params[0]
            return _multi_pauli_rotation_gate_scaluq[gate.name](pauli, angle)
        else:
            assert False, "Unreachable"
    elif is_unitary_matrix_gate_name(gate.name):
        return dense_matrix_gate_scaluq(gate.target_indices, gate.unitary_matrix)
    elif is_parametric_gate_name(gate.name):
        raise ValueError("Parametric gates are not supported")
    else:
        assert False, "Unreachable"


def convert_parametric_gate(
    gate: ParametricQuantumGate,
) -> _backend.Gate:
    if gate.name not in _single_param_gate_scaluq:
        raise ValueError(f"Unknown parametric gate name: {gate.name}")

    if gate.name != gate_names.ParametricPauliRotation:
        return _single_param_gate_scaluq[gate.name](*gate.target_indices, 1.0)

    elif gate.name == gate_names.ParametricPauliRotation:
        target_indices = cast_to_list(gate.target_indices)
        pauli_ids = cast_to_list(gate.pauli_ids)
        pauli = _backend.PauliOperator(target_indices, pauli_ids)
        return _backend.gate.ParamPauliRotation(pauli)
    assert False, "Unreachable"


def convert_circuit(circuit: ImmutableQuantumCircuit) -> _backend.Circuit:
    scaluq_circuit = _backend.Circuit(circuit.qubit_count)

    for gate in circuit.gates:
        scaluq_circuit.add_gate(convert_gate(gate))

    return scaluq_circuit


def convert_parametric_circuit(
    circuit: ParametricQuantumCircuitProtocol,
) -> tuple[
    ScaluqCircuit,
    Callable[[Sequence[float]], dict[str, list[float]]],
]:
    param_circuit: ImmutableParametricQuantumCircuit
    param_mapper: Callable[[Sequence[float]], dict[str, list[float]]]
    if isinstance(circuit, ImmutableLinearMappedParametricQuantumCircuit):
        param_mapping = circuit.param_mapping
        param_circuit = circuit.primitive_circuit()
        orig_param_mapper = param_mapping.seq_mapper

        def _linear_mapped_param_mapper(
            s: Sequence[float],
        ) -> dict[str, list[float]]:
            seq = list(orig_param_mapper(s))
            return {str(i): [v] for i, v in enumerate(seq)}

        param_mapper = _linear_mapped_param_mapper

    elif isinstance(circuit, ImmutableParametricQuantumCircuit):
        param_circuit = circuit

        def _immutable_param_mapper(s: Sequence[float]) -> dict[str, list[float]]:
            return {str(i): [v] for i, v in enumerate(s)}

        param_mapper = _immutable_param_mapper

    else:
        raise ValueError(f"Unsupported parametric circuit type: {type(circuit)}")

    scaluq_circuit = _backend.Circuit(circuit.qubit_count)
    param_count = 0
    for gate, _ in param_circuit._gates:
        if is_parametric_gate_name(gate.name):
            if gate.name == gate_names.ParametricRX:
                scaluq_circuit.add_param_gate(
                    _backend.gate.ParamRX(*gate.target_indices), str(param_count)
                )
            elif gate.name == gate_names.ParametricRY:
                scaluq_circuit.add_param_gate(
                    _backend.gate.ParamRY(*gate.target_indices), str(param_count)
                )
            elif gate.name == gate_names.ParametricRZ:
                scaluq_circuit.add_param_gate(
                    _backend.gate.ParamRZ(*gate.target_indices), str(param_count)
                )
            # TODO: Confirm ParametricPauliRotation spec and add tests
            elif gate.name == gate_names.ParametricPauliRotation:
                target_indices = cast_to_list(gate.target_indices)
                pauli_ids = cast_to_list(gate.pauli_ids)
                scaluq_circuit.add_param_gate(
                    _backend.gate.ParamPauliRotation(
                        _backend.PauliOperator(target_indices, pauli_ids)
                    ),
                    str(param_count),
                )
            else:
                assert_never(gate.name)

            param_count += 1

        else:
            scaluq_circuit.add_gate(convert_gate(gate))

    return scaluq_circuit, param_mapper
