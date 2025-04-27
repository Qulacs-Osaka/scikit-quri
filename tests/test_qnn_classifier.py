import pandas as pd
import pytest
from sklearn import datasets
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from scikit_quri.circuit import create_qcl_ansatz
from scikit_quri.qnn import QNNClassifier
from quri_parts.algo.optimizer import Adam, LBFGS, Optimizer
from quri_parts.qulacs.estimator import (
    create_qulacs_vector_concurrent_estimator,
    create_qulacs_vector_concurrent_parametric_estimator,
)
from quri_parts.core.estimator.gradient import (
    create_numerical_gradient_estimator,
)
from quri_parts.core.operator import Operator, pauli_label


@pytest.mark.parametrize(("solver", "maxiter"), [(Adam(ftol=1e-2), 777), (LBFGS(), 8)])
def test_classify_iris(solver: Optimizer, maxiter: int) -> None:
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    x = df.loc[:, ["petal length (cm)", "petal width (cm)"]]

    x_train, x_test, y_train, y_test = train_test_split(
        x, iris.target, test_size=0.25, random_state=0
    )
    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()

    nqubit = 5
    c_depth = 3
    time_step = 0.5
    num_class = 3
    circuit = create_qcl_ansatz(nqubit, c_depth, time_step, 0)
    estimator = create_qulacs_vector_concurrent_estimator()
    gradient_estimator = create_numerical_gradient_estimator(
        create_qulacs_vector_concurrent_parametric_estimator(), delta=1e-10
    )
    qcl = QNNClassifier(circuit, num_class, estimator, gradient_estimator, solver)

    qcl.fit(x_train, y_train, maxiter)
    y_pred = qcl.predict(x_test).argmax(axis=1)

    assert f1_score(y_test, y_pred, average="weighted") > 0.94


# @pytest.mark.parametrize(
#     ("solver", "maxiter"),
#     [(Adam(ftol=1e-2), 777), (LBFGS(), 8)],
# )
# def test_classify_iris_many(solver: Optimizer, maxiter: int) -> None:
#     iris = datasets.load_iris()
#     df = pd.DataFrame(iris.data, columns=iris.feature_names)
#     x = df.loc[:, ["petal length (cm)", "petal width (cm)"]]

#     x_train, x_test, y_train, y_test = train_test_split(
#         x, iris.target, test_size=0.25, random_state=0
#     )
#     x_train = x_train.to_numpy()
#     x_test = x_test.to_numpy()

#     nqubit = 5
#     c_depth = 3
#     time_step = 0.5
#     num_class = 3
#     circuit = create_qcl_ansatz(nqubit, c_depth, time_step, 0)
#     estimator = create_qulacs_vector_concurrent_estimator()
#     gradient_estimator = create_numerical_gradient_estimator(
#         create_qulacs_vector_concurrent_parametric_estimator(), delta=1e-10
#     )
#     qcl = QNNClassifier(circuit, num_class,  estimator, gradient_estimator, solver)

#     qcl.fit(x_train, y_train, maxiter)
#     y_pred = qcl.predict(x_test).argmax(axis=1)

#     assert f1_score(y_test, y_pred, average="weighted") > 0.94
