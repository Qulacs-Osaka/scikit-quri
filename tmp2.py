import numpy as np
from numpy import pi
from qulacs import Observable, ParametricQuantumCircuit


def array_f4(array):
    for i in range(len(array)):
        array[i] = float(f"{array[i]:.4f}")
    return array


# //〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜
def calc_grad(axis, label):
    c = ParametricQuantumCircuit(1)

    match axis:
        case "X":
            c.add_parametric_RX_gate(0, pi / 4)
        case "Y":
            c.add_parametric_RY_gate(0, pi / 4)
        case "Z":
            c.add_parametric_RZ_gate(0, pi / 4)

    obs = Observable(1)
    obs.add_operator(1.0, label)

    ans = c.backprop(obs)

    print("Simulator:", array_f4(ans))


# //〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜
n_qubits = 1
params = [pi / 2]

calc_grad("X", "X 0")
calc_grad("Y", "X 0")
calc_grad("Z", "X 0")
calc_grad("X", "Y 0")
calc_grad("Y", "Y 0")
calc_grad("Z", "Y 0")
calc_grad("X", "Z 0")
calc_grad("Y", "Z 0")
calc_grad("Z", "Z 0")
