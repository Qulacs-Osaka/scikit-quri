import random
from typing import List

import numpy as np
from numpy.random import RandomState
from numpy.typing import NDArray
from sklearn.metrics import mean_squared_error

from scikit_quri.backend import QulacsSampler
from scikit_quri.circuit import create_ibm_embedding_circuit
from scikit_quri.qsvm import QSVR


def func_to_learn(x) -> float:
    return np.sin(x[0] * x[1] * 2)


def generate_noisy_sine(
    x_min: float,
    x_max: float,
    num_x: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    seed = 0
    random.seed(seed)
    random_state = RandomState(seed)

    x_train: List[List[float]] = []
    y_train: List[float] = []
    mag_noise = 0.05
    for _ in range(num_x):
        xa = x_min + (x_max - x_min) * random.random()
        xb = x_min + (x_max - x_min) * random.random()
        x_train.append([xa, xb])
        y_train.append(func_to_learn([xa, xb]))
    y_train += mag_noise * random_state.randn(num_x)
    return np.array(x_train), np.array(y_train)


def test_noisy_sine():
    x_min = -0.5
    x_max = 0.5
    num_x = 200
    x_train, y_train = generate_noisy_sine(x_min, x_max, num_x)
    x_test, y_test = generate_noisy_sine(x_min, x_max, num_x)
    n_qubit = 4
    circuit = create_ibm_embedding_circuit(n_qubit)
    qsvr = QSVR(circuit)
    sampler = QulacsSampler()
    qsvr.fit(x_train, y_train, sampler, n_shots=2**12)
    y_pred = qsvr.predict(x_test)
    loss = mean_squared_error(y_pred, y_test)
    # QulacsSampler cannot be seeded (quri-parts does not thread a generator through),
    # so this is a shot-noise distribution, not a number. Measured over 25 runs with
    # the corrected ZZ feature map: p50 0.0075, p95 0.0083, max 0.0090, sd 0.00063.
    # The old threshold of 0.008 sat at the p90 of that distribution and flaked on CI.
    #
    # 0.015 is ~12 sd above the median and still less than half of the 0.0306 that
    # predicting the training mean scores, so a model that stopped learning fails.
    # The label noise puts a floor of 0.05^2 = 0.0025 on any predictor.
    assert loss < 0.015
