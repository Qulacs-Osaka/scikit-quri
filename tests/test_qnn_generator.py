from math import exp

import numpy as np
import pytest
from numpy.random import default_rng
from quri_parts.algo.optimizer import Adam, LBFGS

from scikit_quri.backend import QulacsSampler
from scikit_quri.circuit import create_farhi_neven_ansatz
from scikit_quri.qnn import QNNGenerator


@pytest.fixture
def qulacs_sampler() -> QulacsSampler:
    return QulacsSampler()


def test_predict_shape_and_normalization(qulacs_sampler: QulacsSampler) -> None:
    """Sanity check: predict returns a valid probability vector of correct shape."""
    np.random.seed(0)
    n_qubit = 3
    circuit = create_farhi_neven_ansatz(n_qubit, c_depth=2)
    qnn = QNNGenerator(circuit, LBFGS(), qulacs_sampler, n_shots=1024, fitting_qubit=2)

    target = np.array([0.4, 0.3, 0.2, 0.1])
    qnn.fit_direct_distribution(target, maxiter=2, n_target_samples=500)
    p = qnn.predict(n_shots=4000)
    assert p.shape == (4,)
    assert abs(p.sum() - 1.0) < 1e-9
    assert np.all(p >= 0.0)


def test_cost_decreases(qulacs_sampler: QulacsSampler) -> None:
    """Training reduces MMD^2 cost on a simple target distribution.

    A 2-qubit Farhi-Neven ansatz with depth 2 has 8 learning parameters,
    so each gradient step does ~17 sampler calls (1 cost + 2*8 shifts).
    Uses Adam (more robust than LBFGS under shot noise) with modest shots
    and iterations to stay well under a minute.
    """
    np.random.seed(0)
    n_qubit = 2
    circuit = create_farhi_neven_ansatz(n_qubit, c_depth=2)
    qnn = QNNGenerator(circuit, Adam(), qulacs_sampler, n_shots=1024, fitting_qubit=2)

    target = np.array([0.1, 0.2, 0.3, 0.4])
    # Build target samples once so we can measure cost before/after fit.
    rng = np.random.default_rng(0)
    target_samples = rng.choice(4, size=2000, p=target).astype(np.int64)

    initial_theta = 2 * np.pi * np.random.random(circuit.learning_params_count)
    initial_cost = qnn.cost_func(initial_theta, target_samples)

    qnn.fit_direct_distribution(target, maxiter=15, n_target_samples=2000)
    final_cost = qnn.cost_func(qnn.trained_param, target_samples)

    # Training should reduce cost. Use a noticeable margin (cost is on the order
    # of 0.01-0.1 for these problems; shot noise on a single estimate is ~0.001).
    assert final_cost < initial_cost - 0.01, (
        f"cost did not decrease: initial={initial_cost:.4f} final={final_cost:.4f}"
    )


@pytest.mark.skip(
    "Parameter-shift gradient on many params with shot-based sampling is slow; covered by mini."
)
def test_mix_gauss(qulacs_sampler: QulacsSampler) -> None:
    n_qubit = 6
    depth = 10
    circuit = create_farhi_neven_ansatz(n_qubit, depth)
    qnn = QNNGenerator(circuit, LBFGS(), qulacs_sampler, n_shots=4096, fitting_qubit=6)
    prob_list = np.zeros(64)
    ua = 64 * 2 / 7
    ub = 64 * 5 / 7
    v = 64 * 1 / 8
    prob_sum = 0.0
    for i in range(64):
        prob_list[i] = exp(-((ua - i) ** 2) / (2 * v * v)) + exp(-((ub - i) ** 2) / (2 * v * v))
        prob_sum += prob_list[i]
    prob_list /= prob_sum

    rng = default_rng(1)
    datas = rng.choice(a=range(64), size=10000, p=prob_list)

    qnn.fit(datas, maxiter=120)
    data_param = qnn.predict(n_shots=20000)

    gosa = 0.0
    for i in range(3, 61):
        hei = sum((data_param[i + j] - prob_list[i + j]) / 7 for j in range(-3, 4))
        gosa += abs(hei)
    assert gosa / 2 < 0.08


@pytest.mark.skip("This test takes too long time to finish")
def test_bar_stripe(qulacs_sampler: QulacsSampler) -> None:
    n_qubit = 9
    depth = 22
    circuit = create_farhi_neven_ansatz(n_qubit, depth)
    qnn = QNNGenerator(circuit, LBFGS(), qulacs_sampler, n_shots=4096, fitting_qubit=9)

    prob_list = np.zeros(512)
    for i in range(8):
        prob_list[i * 73] += 0.0625
        uuu = 0
        if (i & 4) > 0:
            uuu += 64
        if (i & 2) > 0:
            uuu += 8
        if (i & 1) > 0:
            uuu += 1
        prob_list[uuu * 7] += 0.0625

    rng = default_rng(1)
    datas = rng.choice(a=range(512), size=100000, p=prob_list)

    qnn.fit(datas, maxiter=600)
    data_param = qnn.predict(n_shots=20000)

    gosa = float(np.sum(np.abs(data_param - prob_list)))
    assert gosa < 0.4
