import numpy as np
from numpy import pi
from qulacs import Observable, ParametricQuantumCircuit
from quri_parts.core.operator import Operator, pauli_label

from scikit_quri.circuit import LearningCircuit

# //＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
test_case_enable = [
    False,
    True,
    True,
]


# //＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
def array_f4(array):
    for i in range(len(array)):
        array[i] = float(f"{array[i]:.4f}")
    return array


# //＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
# 2 qubit
# p_RX(0, -pi/2) p_RY(1, pi/2)
# Observable: Z0 + Z1
# expected = [1, -1]

print("Case 1")
if test_case_enable[0]:
    n_qubits = 2
    params = np.array([-pi / 2, pi / 2])

    # //〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜
    # Simulator
    c = ParametricQuantumCircuit(n_qubits)

    c.add_parametric_RX_gate(0, params[0])
    c.add_parametric_RY_gate(1, params[1])

    obs = Observable(n_qubits)
    obs.add_operator(1.0, "Z 0")
    obs.add_operator(1.0, "Z 1")

    ans = c.backprop(obs)

    print("Simulator:", array_f4(ans))

    # //〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜
    # OQTOPUS device
    circuit = LearningCircuit(n_qubits)
    circuit.add_parametric_RX_gate(0)
    circuit.add_parametric_RY_gate(1)

    x = np.array([])
    operator = Operator(
        {
            pauli_label("Z0"): 1.0,
            pauli_label("Z1"): 1.0,
        }
    )

    ans = circuit.backprop(x, params, operator)
    print("OQTOPUS device:", array_f4(ans))

    print()
else:
    print("Skip")
    print()

# //ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
# 2 qubit
# p_RY(0, -pi/4) p_RX(1, pi/4)
# Observable: X0 + Y1
# expected = [-0.7071, 0.7071]

print("Case 2")
if test_case_enable[1]:
    n_qubits = 2
    params = np.array([-pi / 4, pi / 4])

    # //〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜
    # Simulator
    c = ParametricQuantumCircuit(n_qubits)

    c.add_parametric_RY_gate(0, params[0])
    c.add_parametric_RX_gate(1, params[1])

    obs = Observable(n_qubits)
    obs.add_operator(1.0, "X 0")
    obs.add_operator(1.0, "Y 1")

    ans = c.backprop(obs)

    print("Simulator:", array_f4(ans))

    # //〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜
    # OQTOPUS device
    circuit = LearningCircuit(n_qubits)
    circuit.add_parametric_RY_gate(0)
    circuit.add_parametric_RX_gate(1)

    x = np.array([])
    operator = Operator(
        {
            pauli_label("X0"): 1.0,
            pauli_label("Y1"): 1.0,
        }
    )

    ans = circuit.backprop(x, params, operator)
    print("OQTOPUS device:", array_f4(ans))

    print()
else:
    print("Skip")
    print()

# //ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
# 2 qubit
# p_RY(0, pi/2) p_RZ(0, pi/4)
# p_RX(1, pi/2) p_RZ(1, pi/4)
# p_RX(2, pi/2) p_RY(2, pi/4)
# Observable: X0 + Y1 + Z2
# expected = [0, 0.7071, 0, -0.7071, -0.7071, 0]

print("Case 3")
if test_case_enable[2]:
    n_qubits = 3
    params = np.array(
        [
            pi / 2,
            pi / 4,
            pi / 2,
            pi / 4,
            pi / 2,
            pi / 4,
        ]
    )

    # //〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜
    # Simulator
    c = ParametricQuantumCircuit(n_qubits)

    c.add_parametric_RY_gate(0, params[0])
    c.add_parametric_RZ_gate(0, params[1])
    c.add_parametric_RX_gate(1, params[2])
    c.add_parametric_RZ_gate(1, params[3])
    c.add_parametric_RX_gate(2, params[4])
    c.add_parametric_RY_gate(2, params[5])

    obs = Observable(n_qubits)
    obs.add_operator(1.0, "X 0")
    obs.add_operator(1.0, "Y 1")
    obs.add_operator(1.0, "Z 2")

    ans = c.backprop(obs)

    print("Simulator:", array_f4(ans))

    # //〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜
    # OQTOPUS device
    circuit = LearningCircuit(n_qubits)
    circuit.add_parametric_RY_gate(0)
    circuit.add_parametric_RZ_gate(0)
    circuit.add_parametric_RX_gate(1)
    circuit.add_parametric_RZ_gate(1)
    circuit.add_parametric_RX_gate(2)
    circuit.add_parametric_RY_gate(2)

    x = np.array([])
    operator = Operator(
        {
            pauli_label("X0"): 1.0,
            pauli_label("Y1"): 1.0,
            pauli_label("Z2"): 1.0,
        }
    )

    ans = circuit.backprop(x, params, operator)
    print("OQTOPUS device:", array_f4(ans))

    print()
else:
    print("Skip")
    print()
