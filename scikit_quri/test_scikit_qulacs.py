from skqulacs.qnn import QNNRegressor
from skqulacs.circuit import create_farhi_neven_ansatz
from skqulacs.qnn.solver import Adam,Bfgs
from numpy.random import default_rng
import numpy as np
from qulacsvis import circuit_drawer

n_qubits = 2
depth = 2


def generate_noisy_sine(x_min, x_max, num_x):
    rng = default_rng(0)
    x_train = [[rng.uniform(x_min, x_max)] for _ in range(num_x)]
    y_train = [np.sin(np.pi * x[0]) for x in x_train]
    mag_noise = 0.01
    y_train += mag_noise * rng.random(num_x)
    # return np.array(x_train), np.array(y_train)
    return np.array(x_train).flatten(), np.array(y_train)

circuit = create_farhi_neven_ansatz(n_qubits, depth)
print(f"{circuit.get_parameters()}")
circuit_drawer(circuit._circuit,"text")

solver = Adam()
model = QNNRegressor(circuit,solver)

x_train,y_train = generate_noisy_sine(-1.0, 1.0, 10)
print(f"{x_train=}")
model.fit(x_train, y_train,maxiter_or_lr=1) 