import numpy as np
import pytest
from numpy.random import default_rng
from sklearn.metrics import f1_score

from scikit_quri.circuit.pre_defined import create_qcnn_ansatz
from scikit_quri.qnn import QNNClassifier
from quri_parts.algo.optimizer import Adam, Optimizer

from quri_parts.qulacs.estimator import (
    create_qulacs_vector_concurrent_estimator,
    create_qulacs_vector_concurrent_parametric_estimator,
)
from quri_parts.core.estimator.gradient import (
    create_numerical_gradient_estimator,
)


def generate_data(bits: int, random_seed: int = 0):
    """Generate training and testing data."""
    rng = default_rng(random_seed)
    n_rounds = 20  # Produces n_rounds * bits datapoints.
    excitations = []
    labels = []
    for _ in range(n_rounds):
        for _ in range(bits):
            r = rng.uniform(-np.pi, np.pi)
            excitations.append(r)
            labels.append(1 if (-np.pi / 2) <= r <= (np.pi / 2) else 0)

    train_ratio = 0.7
    split_ind = int(len(excitations) * train_ratio)
    train_excitations = excitations[:split_ind]
    test_excitations = excitations[split_ind:]

    train_labels = labels[:split_ind]
    test_labels = labels[split_ind:]

    return (
        np.array(train_excitations),
        np.array(train_labels),
        np.array(test_excitations),
        np.array(test_labels),
    )


"""
tests/test_qcnn.py: 55040 warnings
  circuit.py:421: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
    bound_parameters[param.pos] = angle

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
"""


@pytest.mark.parametrize(("solver", "maxiter"), [(Adam(), 20)])
# @pytest.mark.skip("This test takes too long time to finish")
def test_qcnn(solver: Optimizer, maxiter: int):
    n_qubit = 8
    random_seed = 0
    circuit = create_qcnn_ansatz(n_qubit, random_seed)

    estimator = create_qulacs_vector_concurrent_estimator()
    gradient_estimator = create_numerical_gradient_estimator(
        create_qulacs_vector_concurrent_parametric_estimator(), delta=1e-10
    )
    num_class = 2
    qcl = QNNClassifier(circuit, num_class, estimator, gradient_estimator, solver)

    x_train, y_train, x_test, y_test = generate_data(n_qubit)
    qcl.fit(x_train, y_train, maxiter)
    y_pred = qcl.predict(x_test).argmax(axis=1)
    score = f1_score(y_test, y_pred, average="weighted")
    assert score > 0.9
