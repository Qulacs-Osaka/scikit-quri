import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient

pi = np.pi

# //＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
# n_qubits = 2

# params = np.array(
#     [pi / 3, pi / 3],
#     dtype=float,
# )

# H = Pauli("XY")

test_cases = [
    {
        "n_qubits": 3,
        "params": np.array(
            [pi / 3, pi / 3, pi / 3, pi / 3, pi / 3, pi / 3, pi / 3, pi / 3, pi / 3], dtype=float
        ),
        "hamiltonian": [("IIZY", 1.0), ("IXIY", 1.0), ("YIIY", 1.0)],
        "gates": [
            ("RX", 0),
            ("RZ", 0),
            ("RY", 0),
            ("RY", 1),
            ("RX", 1),
            ("RZ", 1),
            ("RX", 2),
            ("RY", 2),
            ("RZ", 2),
        ],
    },
    {
        "n_qubits": 1,
        "params": np.array([pi / 3, pi / 3], dtype=float),
        "hamiltonian": [("XY", 1.0)],
        "gates": [("RY", 0), ("RZ", 0)],
    },
    {
        "n_qubits": 1,
        "params": np.array(
            [
                pi / 4,
                pi / 4,
                pi / 2,
            ],
            dtype=float,
        ),
        "hamiltonian": [("XY", 1.0)],
        "gates": [("RY", 0), ("RZ", 0), ("RY", 0)],
    },
    {
        "n_qubits": 1,
        "params": np.array([pi / 4], dtype=float),
        "hamiltonian": [("ZY", 1.0)],
        "gates": [("RX", 0)],
    },
    {
        "n_qubits": 1,
        "params": np.array([pi / 3, pi / 6], dtype=float),
        "hamiltonian": [("YY", 1.0)],
        "gates": [("RX", 0), ("RZ", 0)],
    },
    {
        "n_qubits": 2,
        "params": np.array(
            [
                pi / 4,
                pi / 4,
                pi / 2,
                pi / 2,
            ],
            dtype=float,
        ),
        "hamiltonian": [("XIY", 1.0), ("IXY", 1.0)],
        "gates": [("RY", 0), ("RZ", 0), ("RY", 1), ("RZ", 1)],
    },
    {
        "n_qubits": 3,
        "params": np.array(
            [
                pi / 3,
                pi / 5,
                pi / 7,
                pi / 11,
                pi / 13,
                1,
                2,
            ],
            dtype=float,
        ),
        "hamiltonian": [
            ("XIIY", 1.0),
            ("IXIY", 1.0),
            ("IIXY", 1.0),
        ],
        "gates": [("RY", 0), ("RZ", 0), ("RY", 1), ("RZ", 1), ("RZ", 2), ("RY", 2), ("RZ", 2)],
    },
]

# //ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
# theta_list = [Parameter(f"θ{i}") for i in range(len(params))]

# qc = QuantumCircuit(n_qubits, n_qubits)
# qc.h(0)

# qc.ry(params[0], 1)
# qc.rz(params[1], 1)
# qc.rz(-params[1], 1)
# qc.cy(0, 1)
# qc.rz(params[1], 1)
# qc.cz(0, 1)

# H = SparsePauliOp.from_list(hamiltonian)


def apply_gate_by_label(qc, gate, param, target):
    match gate:
        case "RX":
            qc.rx(param, target)
        case "RY":
            qc.ry(param, target)
        case "RZ":
            qc.rz(param, target)
        case _:
            raise ValueError("Unknown gate")


def calc_grad(case):
    n_qubits = case["n_qubits"] + 1
    params = case["params"]
    hamiltonian = case["hamiltonian"]
    gates = case["gates"]
    grad = []

    for i in range(len(params)):
        qc = QuantumCircuit(n_qubits)
        qc.h(0)

        for j, (gate, target) in enumerate(gates):
            apply_gate_by_label(qc, gate, params[j], target + 1)

        qc.barrier()

        for j in range(len(params) - 1, i, -1):
            gate, target = gates[j]
            apply_gate_by_label(qc, gate, -params[j], target + 1)

        gate, target = gates[i]
        match gate:
            case "RX":
                qc.cx(0, target + 1)
            case "RY":
                qc.cy(0, target + 1)
            case "RZ":
                qc.cz(0, target + 1)
            case _:
                raise ValueError("Unknown gate")

        for j in range(i + 1, len(params)):
            gate, target = gates[j]
            apply_gate_by_label(qc, gate, params[j], target + 1)

        # print(qc)
        H = SparsePauliOp.from_list(hamiltonian)

        psi = Statevector.from_instruction(qc)
        expectation = psi.expectation_value(H)
        grad.append(expectation.real)

    return grad


# //ーーーーーーーーーーーーーーーーーーーーー
# psi = Statevector.from_instruction(qc)
# expectation = psi.expectation_value(H)
# print(expectation)


if __name__ == "__main__":
    for case in test_cases:
        print(np.round(calc_grad(case), 4))
        print(
            "//ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー"
        )
