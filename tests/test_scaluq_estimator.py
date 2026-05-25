"""Tests for ScaluqEstimator: capability inheritance, the batched API,
and behavioral parity with QulacsEstimator on the QNN classifier path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from quri_parts.algo.optimizer import LBFGS
from quri_parts.core.estimator.gradient import create_numerical_gradient_estimator
from quri_parts.qulacs.estimator import create_qulacs_vector_concurrent_parametric_estimator
from sklearn import datasets
from sklearn.model_selection import train_test_split

from scikit_quri.backend import (
    BaseEstimator,
    BatchedSimEstimator,
    QulacsEstimator,
    ScaluqEstimator,
)
from scikit_quri.circuit import create_qcl_ansatz
from scikit_quri.qnn import QNNClassifier


def test_capability_hierarchy() -> None:
    """ScaluqEstimator advertises the BatchedSimEstimator capability; QulacsEstimator does not."""
    scaluq = ScaluqEstimator()
    qulacs = QulacsEstimator()

    assert isinstance(scaluq, BaseEstimator)
    assert isinstance(scaluq, BatchedSimEstimator)

    assert isinstance(qulacs, BaseEstimator)
    assert not isinstance(qulacs, BatchedSimEstimator)


def test_estimate_batched_shape() -> None:
    """estimate_batched returns (n_operators, n_samples) for an ansatz bound across a batch."""
    from quri_parts.core.operator import Operator, pauli_label

    n_qubits = 3
    circuit = create_qcl_ansatz(n_qubits, 2, time_step=0.5, seed=0)
    n_samples = 4
    rng = np.random.default_rng(0)
    x_batch = rng.uniform(-1.0, 1.0, (n_samples, n_qubits))
    theta = rng.uniform(0, 2 * np.pi, circuit.learning_params_count)

    parametric_circuit, batched_params = circuit.to_batched(x_batch, theta)

    operators = [Operator({pauli_label(f"Z {q}"): 1.0}) for q in range(n_qubits)]
    estimator = ScaluqEstimator()
    result = estimator.estimate_batched(operators, parametric_circuit, batched_params)
    arr = np.asarray(result, dtype=np.float64)

    assert arr.shape == (len(operators), n_samples)
    # Z expectation values must lie in [-1, 1].
    assert np.all(arr >= -1.0 - 1e-9) and np.all(arr <= 1.0 + 1e-9)


def test_classify_iris_matches_qulacs() -> None:
    """ScaluqEstimator and QulacsEstimator should produce close predictions on the same QNN.

    Trains two classifiers from the same initial parameters and verifies that their
    softmax outputs agree within a small tolerance — exercising the batched path
    end-to-end and confirming behavioral parity with the per-sample path.
    """
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    x = df.loc[:, ["petal length (cm)", "petal width (cm)"]]

    x_train, x_test, y_train, _ = train_test_split(x, iris.target, test_size=0.25, random_state=0)
    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()

    nqubit = 5
    circuit = create_qcl_ansatz(nqubit, 2, 0.5, 0)
    gradient_estimator = create_numerical_gradient_estimator(
        create_qulacs_vector_concurrent_parametric_estimator(), delta=1e-10
    )

    init = 2 * np.pi * np.random.default_rng(0).random(circuit.learning_params_count)

    def train(estimator):
        qcl = QNNClassifier(circuit, 3, estimator, gradient_estimator, LBFGS())
        qcl.trained_param = init.copy()
        qcl.fit(x_train, y_train, maxiter=3)
        return qcl.predict(x_test)

    qulacs_pred = train(QulacsEstimator())
    scaluq_pred = train(ScaluqEstimator())

    # Numerical agreement: both paths evaluate the same circuit, so probabilities
    # must agree closely. Tolerance is loose because scaluq and qulacs differ in
    # floating-point ordering on aggregated sums.
    np.testing.assert_allclose(qulacs_pred, scaluq_pred, atol=1e-4)


def test_sim_estimator_factory_returns_scaluq_with_use_scaluq() -> None:
    """The deprecated SimEstimator factory must still route use_scaluq=True to ScaluqEstimator."""
    from scikit_quri.backend import SimEstimator

    with pytest.warns(DeprecationWarning):
        e = SimEstimator(use_scaluq=True)
    assert isinstance(e, ScaluqEstimator)

    with pytest.warns(DeprecationWarning):
        e = SimEstimator()
    assert isinstance(e, QulacsEstimator)
