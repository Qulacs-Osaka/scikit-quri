import csv
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from scikit_quri.circuit.pre_defined import create_dqn_cl, create_dqn_cl_no_cz
from scikit_quri.qnn.classifier import QNNClassifier
from quri_parts.algo.optimizer import Adam
from quri_parts.qulacs.estimator import (
    create_qulacs_vector_concurrent_parametric_estimator,
)
from quri_parts.core.estimator import Estimatable
from quri_parts.core.estimator.gradient import (
    create_numerical_gradient_estimator,
)
from quri_parts.core.operator import Operator, pauli_label

from scikit_quri.circuit import LearningCircuit
from scikit_quri.backend import SimEstimator

# This script aims to reproduce Ⅳ.B Binary classification in https://arxiv.org/pdf/2112.15002.pdf.
from typing import List, Tuple

locality = 2


def load_dataset(
    file_path: str, ignore_kind: int, test_ratio: float
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Load dataset from specified path.

    Args:
        file_path: File path from which data is loaded.
        ignore_kind: The dataset expected to have 3 classes and we need 2 classes to test. So specify here which class to ignore in loading.
    """
    x = []
    y = []
    with open(file_path) as f:
        reader = csv.reader(f)
        for row in reader:
            kind = int(row[0])
            if kind == ignore_kind:
                continue
            y.append(kind)
            x.append([float(feature) for feature in row[1:]])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, shuffle=True)

    return x_train, x_test, y_train, y_test


def create_classifier(n_features: int, circuit: LearningCircuit, locality: int):
    # Observables are hard-coded in QNNClassifier, so overwrite here.
    estimator = SimEstimator()
    gradient_estimator = create_numerical_gradient_estimator(
        create_qulacs_vector_concurrent_parametric_estimator(), delta=1e-10
    )
    operators: List[Estimatable] = []
    for i in range(n_features):
        if i < locality:
            operators.append(Operator({pauli_label(f"Z {i}"): 1.0}))
        else:
            pass
            # operators.append(Operator({pauli_label(f"I {i}"): 1.0}))
    classifier = QNNClassifier(
        circuit, 2, estimator, gradient_estimator, Adam(), operator=operators
    )
    return classifier


def test_dqn_cl():
    # Use wine dataset retrieved from: https://archive-beta.ics.uci.edu/ml/datasets/wine
    x_train, x_test, y_train, y_test = load_dataset("datasets/wine.data", 3, 0.5)
    for i in range(len(y_train)):
        y_train[i] -= 1
    for i in range(len(y_test)):
        y_test[i] -= 1

    n_features = 6
    maxiter = 30
    circuit = create_dqn_cl(n_features, 5, locality)
    classifier = create_classifier(n_features, circuit, locality)
    classifier.fit(np.array(x_train), np.array(y_train), maxiter)

    y_pred = classifier.predict(np.array(x_test)).argmax(axis=1)
    score = f1_score(y_test, y_pred, average="weighted")
    assert score > 0.8


def test_dqn_cl_no_cz():
    # Use wine dataset retrieved from: https://archive-beta.ics.uci.edu/ml/datasets/wine
    x_train, x_test, y_train, y_test = load_dataset("datasets/wine.data", 3, 0.5)
    for i in range(len(y_train)):
        y_train[i] -= 1
    for i in range(len(y_test)):
        y_test[i] -= 1

    n_features = 6
    maxiter = 30
    circuit = create_dqn_cl_no_cz(n_features, 5)
    classifier = create_classifier(n_features, circuit, locality)
    classifier.fit(np.array(x_train), np.array(y_train), maxiter)

    y_pred = classifier.predict(np.array(x_test)).argmax(axis=1)
    score = f1_score(y_test, y_pred, average="weighted")
    assert score > 0.8
