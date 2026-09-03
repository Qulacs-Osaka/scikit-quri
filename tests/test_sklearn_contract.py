"""The scikit-learn estimator contract.

`scikit-quri` names itself after scikit-learn and already depends on it (SVC, SVR,
KernelRidge and log_loss are all sklearn), but the models did not implement its
estimator API, so `clone` — and therefore every cross-validation helper — failed.
"""

import contextlib
import io

import numpy as np
import pytest
from quri_parts.algo.optimizer import Adam
from quri_parts.core.estimator.gradient import create_numerical_gradient_estimator
from quri_parts.qulacs.estimator import create_qulacs_vector_concurrent_parametric_estimator
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scikit_quri.backend import QulacsEstimator, QulacsSampler
from scikit_quri.circuit import create_ibm_embedding_circuit, create_qcl_ansatz
from scikit_quri.qkrr import QKRR
from scikit_quri.qnn import QNNClassifier, QNNRegressor
from scikit_quri.qsvm import QSVC

GRADIENT_ESTIMATOR = create_numerical_gradient_estimator(
    create_qulacs_vector_concurrent_parametric_estimator(), delta=1e-6
)


def _quiet(fn, *a, **kw):
    """QNN.fit prints progress unconditionally."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def _data(n=12, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, 1))
    return x, (x[:, 0] > 0).astype(int)


def _qnn_classifier():
    return QNNClassifier(
        create_qcl_ansatz(3, 1, 0.5, 0), 2, QulacsEstimator(), GRADIENT_ESTIMATOR, Adam(), seed=0
    )


def _qnn_regressor():
    return QNNRegressor(
        create_qcl_ansatz(3, 1, 0.5, 0), QulacsEstimator(), GRADIENT_ESTIMATOR, Adam(), seed=0
    )


def _qsvc():
    return QSVC(create_ibm_embedding_circuit(2), sampler=QulacsSampler(), n_shots=200)


def _qkrr():
    return QKRR(create_ibm_embedding_circuit(2), n_iteration=2, sampler=QulacsSampler())


ALL = [_qnn_classifier, _qnn_regressor, _qsvc, _qkrr]
IDS = ["QNNClassifier", "QNNRegressor", "QSVC", "QKRR"]


@pytest.mark.parametrize("factory", ALL, ids=IDS)
def test_models_implement_the_estimator_api(factory):
    model = factory()
    assert isinstance(model, BaseEstimator)
    assert model.get_params()  # non-empty: params are discoverable
    assert hasattr(model, "score")
    clone(model)  # raised TypeError before


@pytest.mark.parametrize("factory", [_qnn_classifier, _qsvc], ids=["QNNClassifier", "QSVC"])
def test_cross_val_score_runs(factory):
    x, y = _data()
    scores = _quiet(cross_val_score, factory(), x, y, cv=3)
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))


def test_grid_search_runs():
    x, y = _data()
    search = GridSearchCV(_qsvc(), {"n_shots": [100, 200]}, cv=2)
    _quiet(search.fit, x, y)
    assert search.best_params_["n_shots"] in (100, 200)


def test_pipeline_runs():
    x, y = _data()
    pipeline = Pipeline([("scale", StandardScaler()), ("model", _qsvc())])
    _quiet(pipeline.fit, x, y)
    assert pipeline.predict(x).shape == (len(x),)


def test_classifier_predict_returns_labels():
    """predict used to return the (n, num_class) score matrix while its docstring
    called the values probabilities, so accuracy_score and log_loss were both wrong."""
    x, y = _data()
    model = _qnn_classifier()
    _quiet(model.fit, x, y, maxiter=1)

    labels = model.predict(x)
    assert labels.shape == (len(x),)
    assert set(np.unique(labels)) <= {0, 1}

    scores = model.decision_function(x)
    assert scores.shape == (len(x), 2)

    proba = model.predict_proba(x)
    assert proba.shape == (len(x), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(proba >= 0.0)

    np.testing.assert_array_equal(labels, scores.argmax(axis=1))


def test_generator_predict_is_deprecated_in_favour_of_sample():
    from scikit_quri.circuit import create_farhi_neven_ansatz
    from scikit_quri.qnn.generation import QNNGenerator
    from quri_parts.algo.optimizer import LBFGS

    rng = np.random.default_rng(0)
    model = QNNGenerator(
        create_farhi_neven_ansatz(3, 1), LBFGS(), QulacsSampler(), n_shots=256, fitting_qubit=3
    )
    _quiet(model.fit, rng.integers(0, 8, 200), maxiter=1)

    distribution = model.sample(n_shots=256)
    assert distribution.shape == (8,)
    np.testing.assert_allclose(distribution.sum(), 1.0, atol=1e-9)

    with pytest.warns(DeprecationWarning, match="use sample"):
        model.predict(n_shots=256)
