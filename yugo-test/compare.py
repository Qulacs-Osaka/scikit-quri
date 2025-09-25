import numpy as np
from numpy import pi
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
from qulacs import Observable, ParametricQuantumCircuit
from quri_parts.core.operator import Operator, pauli_label
from utils import array_f4

from scikit_quri.circuit import LearningCircuit

# //＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
test_case_enable = [False, False, True, False, False, False]

enable_oqtopus = True

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
    # c = ParametricQuantumCircuit(n_qubits)

    # c.add_parametric_RX_gate(0, params[0])
    # c.add_parametric_RY_gate(1, params[1])

    # obs = Observable(n_qubits)
    # obs.add_operator(1.0, "Z 0")
    # obs.add_operator(1.0, "Z 1")

    # ans = c.backprop(obs)

    # print("Simulator:", array_f4(ans))

    # //〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜
    # Qiskit Simulator
    theta_list = [Parameter("θ0"), Parameter("θ1")]
    qc = QuantumCircuit(n_qubits)
    qc.rx(theta_list[0], 0)
    qc.ry(theta_list[1], 1)

    pauli_list = [
        ("ZI", 1.0),
        ("IZ", 1.0),
    ]
    H = SparsePauliOp.from_list(pauli_list)

    estimator = StatevectorEstimator()
    gradient = ParamShiftEstimatorGradient(estimator)
    job = gradient.run(circuits=[qc], observables=[H], parameter_values=[params.tolist()])
    result = job.result()
    print("Qiskit Simulator:", result.gradients[0])

    # //〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜
    # OQTOPUS device
    if enable_oqtopus:
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
    # c = ParametricQuantumCircuit(n_qubits)

    # c.add_parametric_RY_gate(0, params[0])
    # c.add_parametric_RX_gate(1, params[1])

    # obs = Observable(n_qubits)
    # obs.add_operator(1.0, "X 0")
    # obs.add_operator(1.0, "Y 1")

    # ans = c.backprop(obs)

    # print("Simulator:", array_f4(ans))

    # //〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜
    # Qiskit Simulator
    theta_list = [Parameter("θ0"), Parameter("θ1")]
    qc = QuantumCircuit(n_qubits)
    qc.ry(theta_list[0], 0)
    qc.rx(theta_list[1], 1)

    pauli_list = [
        ("IX", 1.0),
        ("YI", 1.0),
    ]
    H = SparsePauliOp.from_list(pauli_list)

    estimator = StatevectorEstimator()
    gradient = ParamShiftEstimatorGradient(estimator)
    job = gradient.run(circuits=[qc], observables=[H], parameter_values=[params.tolist()])
    result = job.result()
    print("Qiskit Simulator:", array_f4(result.gradients[0]))

    # //〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜
    # OQTOPUS device
    if enable_oqtopus:
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
else:
    print("Skip")

print()

# //ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
# 3 qubit
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
            pi / 4,
            pi / 4,
            pi / 4,
            pi / 4,
            pi / 4,
            pi / 4,
        ]
    )

    # //〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜
    # Simulator
    # c = ParametricQuantumCircuit(n_qubits)

    # c.add_parametric_RY_gate(0, params[0])
    # c.add_parametric_RZ_gate(0, params[1])
    # c.add_parametric_RX_gate(1, params[2])
    # c.add_parametric_RZ_gate(1, params[3])
    # c.add_parametric_RX_gate(2, params[4])
    # c.add_parametric_RY_gate(2, params[5])

    # obs = Observable(n_qubits)
    # obs.add_operator(1.0, "X 0")
    # obs.add_operator(1.0, "Y 1")
    # obs.add_operator(1.0, "Z 2")

    # ans = c.backprop(obs)

    # print("Simulator:", array_f4(ans))

    # //〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜
    # Qiskit Simulator
    theta_list = [
        Parameter("θ0"),
        Parameter("θ1"),
        Parameter("θ2"),
        Parameter("θ3"),
        Parameter("θ4"),
        Parameter("θ5"),
    ]
    qc = QuantumCircuit(n_qubits)
    qc.ry(theta_list[0], 0)
    qc.rz(theta_list[1], 0)
    qc.rx(theta_list[2], 1)
    qc.rz(theta_list[3], 1)
    qc.rx(theta_list[4], 2)
    qc.ry(theta_list[5], 2)

    pauli_list = [
        ("IIX", 1.0),
        ("IYI", 1.0),
        ("ZII", 1.0),
    ]
    H = SparsePauliOp.from_list(pauli_list)

    estimator = StatevectorEstimator()
    gradient = ParamShiftEstimatorGradient(estimator)
    job = gradient.run(circuits=[qc], observables=[H], parameter_values=[params.tolist()])
    result = job.result()
    print("Qiskit Simulator:", array_f4(result.gradients[0]))

    # //〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜
    # OQTOPUS device
    if enable_oqtopus:
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
else:
    print("Skip")

print()

# //ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
# 3 qubit
# p_RX(0, pi/2) p_RY(0, pi/2) p_RZ(0, pi/2)
# p_RX(1, pi/2) p_RY(1, pi/2) p_RZ(1, pi/2)
# p_RX(2, pi/2) p_RY(2, pi/2) p_RZ(2, pi/2)
# Observable: X0 + Y1 + Z2
# expected = [0.0, 0.0, 0.0, -1.0, -0.0, -1.0, 0.0, 0.0, 0.0]

print("Case 4")
if test_case_enable[3]:
    n_qubits = 3
    params = np.array(
        [
            pi / 2,
            pi / 2,
            pi / 2,
            pi / 2,
            pi / 2,
            pi / 2,
            pi / 2,
            pi / 2,
            pi / 2,
        ]
    )

    # //〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜
    # Simulator
    c = ParametricQuantumCircuit(n_qubits)

    c.add_parametric_RX_gate(0, params[0])
    c.add_parametric_RY_gate(0, params[1])
    c.add_parametric_RZ_gate(0, params[2])
    c.add_parametric_RX_gate(1, params[3])
    c.add_parametric_RY_gate(1, params[4])
    c.add_parametric_RZ_gate(1, params[5])
    c.add_parametric_RX_gate(2, params[6])
    c.add_parametric_RY_gate(2, params[7])
    c.add_parametric_RZ_gate(2, params[8])

    obs = Observable(n_qubits)
    obs.add_operator(1.0, "X 0")
    obs.add_operator(1.0, "Y 1")
    obs.add_operator(1.0, "Z 2")

    ans = c.backprop(obs)

    print("Simulator:", array_f4(ans))

    # //〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜
    # OQTOPUS device
    if enable_oqtopus:
        pass
        # circuit = LearningCircuit(n_qubits)
        # circuit.add_parametric_RX_gate(0)
        # circuit.add_parametric_RY_gate(0)
        # circuit.add_parametric_RZ_gate(0)
        # circuit.add_parametric_RX_gate(1)
        # circuit.add_parametric_RY_gate(1)
        # circuit.add_parametric_RZ_gate(1)
        # circuit.add_parametric_RX_gate(2)
        # circuit.add_parametric_RY_gate(2)
        # circuit.add_parametric_RZ_gate(2)

        # x = np.array([])
        # operator = Operator(
        #     {
        #         pauli_label("X0"): 1.0,
        #         pauli_label("Y1"): 1.0,
        #         pauli_label("Z2"): 1.0,
        #     }
        # )

        # ans = circuit.backprop(x, params, operator, shots=2024)
        # print("OQTOPUS device:", array_f4(ans))
else:
    print("Skip")

print()

# //ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
# 18 qubit
# p_RX(0, pi/2) p_RY(0, pi/2) p_RZ(0, pi/2)
# p_RX(1, pi/2) p_RZ(1, pi/2) p_RY(1, pi/2)
# p_RY(2, pi/2) p_RZ(2, pi/2) p_RX(2, pi/2)
# p_RY(3, pi/2) p_RX(3, pi/2) p_RZ(3, pi/2)
# p_RZ(4, pi/2) p_RX(4, pi/2) p_RY(4, pi/2)
# p_RZ(5, pi/2) p_RY(5, pi/2) p_RX(5, pi/2)
#
# p_RX(6, pi/2) p_RY(6, pi/2) p_RZ(6, pi/2)
# p_RX(7, pi/2) p_RZ(7, pi/2) p_RY(7, pi/2)
# p_RY(8, pi/2) p_RZ(8, pi/2) p_RX(8, pi/2)
# p_RY(9, pi/2) p_RX(9, pi/2) p_RZ(9, pi/2)
# p_RZ(10, pi/2) p_RX(10, pi/2) p_RY(10, pi/2)
# p_RZ(11, pi/2) p_RY(11, pi/2) p_RX(11, pi/2)
#
# p_RX(12, pi/2) p_RY(12, pi/2) p_RZ(12, pi/2)
# p_RX(13, pi/2) p_RZ(13, pi/2 ) p_RY(13, pi/2)
# p_RY(14, pi/2) p_RZ(14, pi/2) p_RX(14, pi/2)
# p_RY(15, pi/2) p_RX(15, pi/2) p_RZ(15, pi/2)
# p_RZ(16, pi/2) p_RX(16, pi/2) p_RY(16, pi/2)
# p_RZ(17, pi/2) p_RY(17, pi/2) p_RX(17, pi/2)
#
# Observable: X0 + X1 + X2 + X3 + X4 + X5 +
#             Y6 + Y7 + Y8 + Y9 + Y10 + Y11 +
#             Z12 + Z13 + Z14 + Z15 + Z16 + Z17

print("Case 5")
if test_case_enable[4]:
    n_qubits = 18
    params = np.array([pi / 2] * 54)

    # //〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜
    # Simulator
    c = ParametricQuantumCircuit(n_qubits)

    for i in range(3):
        c.add_parametric_RX_gate(i, params[i * 3 + 0])
        c.add_parametric_RY_gate(i, params[i * 3 + 1])
        c.add_parametric_RZ_gate(i, params[i * 3 + 2])
        #
        c.add_parametric_RX_gate(i, params[i * 3 + 3])
        c.add_parametric_RZ_gate(i, params[i * 3 + 4])
        c.add_parametric_RY_gate(i, params[i * 3 + 5])
        #
        c.add_parametric_RY_gate(i, params[i * 3 + 6])
        c.add_parametric_RZ_gate(i, params[i * 3 + 7])
        c.add_parametric_RX_gate(i, params[i * 3 + 8])
        #
        c.add_parametric_RY_gate(i, params[i * 3 + 9])
        c.add_parametric_RX_gate(i, params[i * 3 + 10])
        c.add_parametric_RZ_gate(i, params[i * 3 + 11])
        #
        c.add_parametric_RZ_gate(i, params[i * 3 + 12])
        c.add_parametric_RX_gate(i, params[i * 3 + 13])
        c.add_parametric_RY_gate(i, params[i * 3 + 14])
        #
        c.add_parametric_RZ_gate(i, params[i * 3 + 15])
        c.add_parametric_RY_gate(i, params[i * 3 + 16])
        c.add_parametric_RX_gate(i, params[i * 3 + 17])

    obs = Observable(n_qubits)
    for i in range(6):
        obs.add_operator(1.0, f"X {i}")
    for i in range(6, 12):
        obs.add_operator(1.0, f"Y {i}")
    for i in range(12, 18):
        obs.add_operator(1.0, f"Z {i}")

    print(obs)

    ans = c.backprop(obs)

    # print("Simulator:", array_f4(ans))
    ans_f4 = array_f4(ans)
    print(ans_f4[0:17])
    print(ans_f4[18:35])
    print(ans_f4[36:53])
    # ans_f4_reshaped = np.array(ans_f4).reshape(-1, 3)
    # for i in range(len(ans_f4_reshaped)):
    #     print(f"{ans_f4_reshaped[i]}")
    #     if i % 6 == 5:
    #         print()

    # //〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜
    # OQTOPUS device
    if enable_oqtopus:
        circuit = LearningCircuit(n_qubits)
        circuit.add_parametric_RX_gate(0)
        circuit.add_parametric_RY_gate(0)
        circuit.add_parametric_RZ_gate(0)
        circuit.add_parametric_RX_gate(1)
        circuit.add_parametric_RY_gate(1)
        circuit.add_parametric_RZ_gate(1)
        circuit.add_parametric_RX_gate(2)
        circuit.add_parametric_RY_gate(2)
        circuit.add_parametric_RZ_gate(2)

        x = np.array([])
        operator = Operator(
            {
                pauli_label("X0"): 1.0,
                pauli_label("Y1"): 1.0,
                pauli_label("Z2"): 1.0,
            }
        )

        ans = circuit.backprop(x, params, operator, shots=2024)
        print("OQTOPUS device:", array_f4(ans))
else:
    print("Skip")

print()

# //ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
print("Case 6")
if test_case_enable[5]:
    n_qubits = 9
    params = np.array([pi / 4] * 9)

    # //〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜
    # Simulator
    # c = ParametricQuantumCircuit(n_qubits)

    # c.add_parametric_RX_gate(0, params[0])
    # c.add_parametric_RY_gate(1, params[1])
    # c.add_parametric_RZ_gate(2, params[2])
    # c.add_parametric_RX_gate(3, params[3])
    # c.add_parametric_RZ_gate(4, params[4])
    # c.add_parametric_RY_gate(5, params[5])
    # c.add_parametric_RY_gate(6, params[6])
    # c.add_parametric_RZ_gate(7, params[7])
    # c.add_parametric_RX_gate(8, params[8])

    # obs = Observable(n_qubits)
    # obs.add_operator(1.0, "X 0")
    # obs.add_operator(1.0, "X 1")
    # obs.add_operator(1.0, "X 2")
    # obs.add_operator(1.0, "Y 3")
    # obs.add_operator(1.0, "Y 4")
    # obs.add_operator(1.0, "Y 5")
    # obs.add_operator(1.0, "Z 6")
    # obs.add_operator(1.0, "Z 7")
    # obs.add_operator(1.0, "Z 8")

    # print(obs)

    # ans = c.backprop(obs)

    # ans_f4 = array_f4(ans)
    # for element in ans_f4:
    #     print(f"{element}")

    # //〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜
    # Qiskit Simulator
    theta_list = [Parameter(f"θ{i}") for i in range(9)]

    qc = QuantumCircuit(n_qubits)
    qc.rx(theta_list[0], 0)
    qc.ry(theta_list[1], 1)
    qc.rz(theta_list[2], 2)
    qc.rx(theta_list[3], 3)
    qc.rz(theta_list[4], 4)
    qc.ry(theta_list[5], 5)
    qc.ry(theta_list[6], 6)
    qc.rz(theta_list[7], 7)
    qc.rx(theta_list[8], 8)

    pauli_list = [
        ("IIIIIIIIX", 1.0),
        ("IIIIIIIXI", 1.0),
        ("IIIIIIXII", 1.0),
        ("IIIIIYIII", 1.0),
        ("IIIIYIIII", 1.0),
        ("IIIYIIIII", 1.0),
        ("IIZIIIIII", 1.0),
        ("IZIIIIIII", 1.0),
        ("ZIIIIIIII", 1.0),
    ]
    H = SparsePauliOp.from_list(pauli_list)

    estimator = StatevectorEstimator()
    gradient = ParamShiftEstimatorGradient(estimator)
    job = gradient.run(circuits=[qc], observables=[H], parameter_values=[params.tolist()])
    result = job.result()
    print("Qiskit Simulator:", array_f4(result.gradients[0]))

    # //〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜
    # OQTOPUS device
    if enable_oqtopus:
        circuit = LearningCircuit(n_qubits)
        circuit.add_parametric_RX_gate(0)
        circuit.add_parametric_RY_gate(1)
        circuit.add_parametric_RZ_gate(2)
        circuit.add_parametric_RX_gate(3)
        circuit.add_parametric_RZ_gate(4)
        circuit.add_parametric_RY_gate(5)
        circuit.add_parametric_RY_gate(6)
        circuit.add_parametric_RZ_gate(7)
        circuit.add_parametric_RX_gate(8)

        x = np.array([])
        operator = Operator(
            {
                pauli_label("X0"): 1.0,
                pauli_label("X1"): 1.0,
                pauli_label("X2"): 1.0,
                pauli_label("Y3"): 1.0,
                pauli_label("Y4"): 1.0,
                pauli_label("Y5"): 1.0,
                pauli_label("Z6"): 1.0,
                pauli_label("Z7"): 1.0,
                pauli_label("Z8"): 1.0,
            }
        )

        ans = circuit.backprop(x, params, operator, shots=2024)
        print("OQTOPUS device:", array_f4(ans))
else:
    print("Skip")

print()
