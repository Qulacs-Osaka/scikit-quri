"""Contract of the public API: what a caller can rely on.

Each case here is something that used to fail silently or block ordinary
scikit-learn usage.
"""

import numpy as np
import pytest
from quri_parts.algo.optimizer import Adam
from quri_parts.core.estimator.gradient import create_numerical_gradient_estimator
from quri_parts.core.operator import Operator, pauli_label
from quri_parts.qulacs.estimator import create_qulacs_vector_concurrent_parametric_estimator

from scikit_quri.backend import QulacsEstimator, QulacsSampler
from scikit_quri.circuit import create_ibm_embedding_circuit, create_qcl_ansatz
from scikit_quri.qkrr import QKRR
from scikit_quri.qnn import QNNClassifier
from scikit_quri.qsvm import QSVC

GRADIENT_ESTIMATOR = create_numerical_gradient_estimator(
    create_qulacs_vector_concurrent_parametric_estimator(), delta=1e-6
)


def _toy(n=6, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, 1))
    y = (x[:, 0] > 0).astype(int)
    return x, y


def test_qsvc_takes_the_sampler_in_the_constructor():
    """fit(X, y) must work: sklearn helpers never pass a backend."""
    x, y = _toy()
    model = QSVC(create_ibm_embedding_circuit(2), sampler=QulacsSampler(), n_shots=200)
    model.fit(x, y)
    assert model.predict(x).shape == (len(x),)


def test_qkrr_takes_the_sampler_in_the_constructor():
    x, y = _toy()
    model = QKRR(create_ibm_embedding_circuit(2), n_iteration=2, sampler=QulacsSampler())
    model.fit(x, y.astype(float))
    assert model.predict(x).shape == (len(x),)


def test_passing_the_sampler_to_fit_still_works_but_warns():
    x, y = _toy()
    model = QSVC(create_ibm_embedding_circuit(2))
    with pytest.warns(DeprecationWarning, match="pass it to the constructor"):
        model.fit(x, y, sampler=QulacsSampler(), n_shots=200)
    assert model.predict(x).shape == (len(x),)


def test_fit_without_a_sampler_says_what_is_missing():
    x, y = _toy()
    model = QSVC(create_ibm_embedding_circuit(2))
    with pytest.raises(ValueError, match="No sampler configured"):
        model.fit(x, y)


def test_qkrr_can_be_fitted_twice():
    """The second fit used to raise "cannot reshape array of size ...": data_circuits
    was appended to rather than reset, so the Gram matrix outgrew len(x)."""
    x, y = _toy()
    model = QKRR(create_ibm_embedding_circuit(2), n_iteration=2, sampler=QulacsSampler())
    model.fit(x, y.astype(float))
    model.fit(x, y.astype(float))
    assert model.predict(x).shape == (len(x),)


def test_a_supplied_operator_survives_fit():
    """The observable passed to the constructor used to be discarded by fit()."""
    custom = [Operator({pauli_label(f"X {i}"): 1.0}) for i in range(2)]
    model = QNNClassifier(
        create_qcl_ansatz(3, 1, 0.5, 0),
        2,
        QulacsEstimator(),
        GRADIENT_ESTIMATOR,
        Adam(),
        operator=list(custom),
        seed=0,
    )
    x, y = _toy()
    model.fit(x, y, maxiter=1)
    assert model.operator == custom


def test_default_operator_is_still_built_when_none_is_given():
    model = QNNClassifier(
        create_qcl_ansatz(3, 1, 0.5, 0),
        2,
        QulacsEstimator(),
        GRADIENT_ESTIMATOR,
        Adam(),
        seed=0,
    )
    x, y = _toy()
    model.fit(x, y, maxiter=1)
    assert len(model.operator) == 2


def test_kernel_methods_reject_a_trainable_ansatz():
    """QSVC/QKRR evaluate the feature map only. Passing a trainable ansatz used to
    fail deep inside the parameter registry with
    "IndexError: index 0 is out of bounds for axis 0 with size 0"."""
    for factory in (QSVC, QKRR):
        with pytest.raises(ValueError, match="no learnable parameters"):
            factory(create_qcl_ansatz(2, 1, 0.5, 0))
