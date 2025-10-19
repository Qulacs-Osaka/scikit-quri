import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient

from .model import GateInfo, GateMode, GateType, TestData


def apply_gate(qc: QuantumCircuit, gates: list[GateInfo], theta: list[float]):
    theta_list = [Parameter(f"θ{i}") for i in range(len(theta))]
    i = 0
    for gate in gates:
        if gate.gate_mode == GateMode.LEARNING:
            match gate.gate_type:
                case GateType.RX:
                    qc.rx(theta_list[i], gate.t_bit)
                case GateType.RY:
                    qc.ry(theta_list[i], gate.t_bit)
                case GateType.RZ:
                    qc.rz(theta_list[i], gate.t_bit)
            i += 1
        else:
            match gate.gate_type:
                case GateType.X:
                    qc.x(gate.t_bit)
                case GateType.Y:
                    qc.y(gate.t_bit)
                case GateType.Z:
                    qc.z(gate.t_bit)
                case GateType.RX:
                    qc.rx(gate.param, gate.t_bit)
                case GateType.RY:
                    qc.ry(gate.param, gate.t_bit)
                case GateType.RZ:
                    qc.rz(gate.param, gate.t_bit)
                case GateType.H:
                    qc.h(gate.t_bit)
                case GateType.CX:
                    qc.cx(gate.c_bit, gate.t_bit)
                case GateType.CY:
                    qc.cy(gate.c_bit, gate.t_bit)
                case GateType.CZ:
                    qc.cz(gate.c_bit, gate.t_bit)
                case GateType.SWAP:
                    qc.swap(gate.c_bit, gate.t_bit)


def execute(test: TestData):
    qc = QuantumCircuit(test.n_qubits)
    apply_gate(qc, test.gates, test.theta)

    observable = []
    for pauli_str, coef in test.observable:
        # "X0 I1 I2" -> "IIX", "X0 I3" -> "I I I X"のように直したい
        pauli_ops = ["I"] * test.n_qubits
        for i in range(test.n_qubits):
            if f"{pauli_str[0]}{i}" in pauli_str:
                pauli_ops[test.n_qubits - 1 - i] = pauli_str[0]
        new_pauli_str = "".join(pauli_ops)
        observable.append((new_pauli_str, coef))
    print("Observable:", observable)

    H = SparsePauliOp.from_list(observable)

    estimator = StatevectorEstimator()
    gradient = ParamShiftEstimatorGradient(estimator)
    job = gradient.run(circuits=[qc], observables=[H], parameter_values=[test.theta])
    result = job.result()
    print("Qiskit Simulator:", np.round(result.gradients[0], 4))
