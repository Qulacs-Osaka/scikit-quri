import numpy as np
from numpy.typing import NDArray
from quri_parts.core.operator import Operator, pauli_label

from scikit_quri.circuit import LearningCircuit

from .model import GateInfo, GateMode, GateType, TestData


def apply_gate(lc: LearningCircuit, gates: list[GateInfo], theta: list[float], x: list[float]):
    input_index = 0
    lp_index = 0
    for gate in gates:
        if gate.gate_mode == None:
            match gate.gate_type:
                case GateType.X:
                    lc.add_X_gate(gate.t_bit)
                case GateType.Y:
                    lc.add_Y_gate(gate.t_bit)
                case GateType.Z:
                    lc.add_Z_gate(gate.t_bit)
                case GateType.RX:
                    lc.add_RX_gate(gate.t_bit, gate.param)
                case GateType.RY:
                    lc.add_RY_gate(gate.t_bit, gate.param)
                case GateType.RZ:
                    lc.add_RZ_gate(gate.t_bit, gate.param)
                case GateType.H:
                    lc.add_H_gate(gate.t_bit)
                case GateType.CX:
                    lc.add_CNOT_gate(gate.c_bit, gate.t_bit)
        elif gate.gate_mode == GateMode.LEARNING:
            match gate.gate_type:
                case GateType.RX:
                    lc.add_parametric_RX_gate(gate.t_bit)
                case GateType.RY:
                    lc.add_parametric_RY_gate(gate.t_bit)
                case GateType.RZ:
                    lc.add_parametric_RZ_gate(gate.t_bit)
        elif gate.gate_mode == GateMode.INPUT:
            match gate.gate_type:
                case GateType.RX:
                    lc.add_input_RX_gate(gate.t_bit, lambda x: x[0])
                case GateType.RY:
                    lc.add_input_RY_gate(gate.t_bit, lambda x: x[0])
                case GateType.RZ:
                    lc.add_input_RZ_gate(gate.t_bit, lambda x: x[0])
            lp_index += 1


def execute(test: TestData):
    lc = LearningCircuit(test.n_qubits)

    apply_gate(lc, test.gates, test.theta, test.x)
    labels = {}
    for pauli_str, coef in test.observable:
        labels[pauli_label(pauli_str)] = coef
    op = Operator(labels)

    expectation = lc.hadamard_gradient(
        x=np.array(test.x),
        theta=np.array(test.theta),
        operator=op,
    )
    print(f"Hadamard grad: {np.round(expectation, 4)}")
