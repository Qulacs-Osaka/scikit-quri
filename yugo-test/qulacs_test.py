import numpy as np
from qulacs import Observable, ParametricQuantumCircuit
from utils import array_f4

pi = np.pi

n_qubits = 1
params = np.array([-pi / 4, pi / 4])

# //〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜
# Simulator
c = ParametricQuantumCircuit(n_qubits)

c.add_parametric_RX_gate(0, params[0])
c.add_parametric_RY_gate(0, params[1])

obs = Observable(n_qubits)
obs.add_operator(1.0, "Z 0")

ans = c.backprop(obs)

print("Simulator:", array_f4(ans))
