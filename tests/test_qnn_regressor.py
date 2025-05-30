from typing import Callable, Optional, Tuple

import numpy as np
import pytest

from numpy.random import default_rng
from numpy.typing import NDArray
from sklearn.metrics import mean_squared_error

from scikit_quri.circuit import create_qcl_ansatz

# from scikit_quri.circuit.pre_defined import (
#     create_multi_qubit_param_rotational_ansatz,
# )
from quri_parts.qulacs.estimator import (
    create_qulacs_vector_concurrent_estimator,
    create_qulacs_vector_concurrent_parametric_estimator,
)
from quri_parts.core.estimator.gradient import (
    create_numerical_gradient_estimator,
    create_parameter_shift_gradient_estimator,
)
from quri_parts.core.operator import Operator, pauli_label
from scikit_quri.qnn import QNNRegressor
from quri_parts.algo.optimizer import Adam, LBFGS, Optimizer


def generate_noisy_data(
    x_min: float,
    x_max: float,
    x_shape: Tuple[int, int],
    function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    seed: Optional[int] = 0,
    mag_noise: float = 0.001,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate random input data and its output

    Args:
        x_min: Minimum limit of random x value.
        x_max: Maximum limit of random x value.
        x_shape: Shape of x, (batch, features).
        function: Function which generates output from x. It takes 1D feature vector and returns 1D output vector.
        seed: Seed for random value.
        mag_noise: Noise amplitude to be added to output.
    """
    rng = default_rng(seed)
    x_train = rng.uniform(x_min, x_max, x_shape)
    y_train = np.array([function(x) for x in x_train])
    y_train += mag_noise * rng.random(y_train.shape)
    return x_train, y_train


def two_vars_two_outputs(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([2.0 * x[0] + x[1], 1.5 * x[0] - 3.0 * x[1]])


@pytest.mark.parametrize(("solver", "maxiter"), [(LBFGS(), 30), (Adam(), 30)])
def test_noisy_two_vars_two_outputs(solver: Optimizer, maxiter: int) -> None:
    x_min = -0.5
    x_max = 0.5
    num_x = 50
    x_train, y_train = generate_noisy_data(x_min, x_max, (num_x, 2), two_vars_two_outputs)

    n_qubit = 4
    depth = 3
    time_step = 0.5
    estimator = create_qulacs_vector_concurrent_estimator()
    gradient_estimator = create_numerical_gradient_estimator(
        create_qulacs_vector_concurrent_parametric_estimator(), delta=1e-10
    )
    circuit = create_qcl_ansatz(n_qubit, depth, time_step, 0)
    qnn = QNNRegressor(circuit, estimator, gradient_estimator, solver)
    qnn.fit(x_train, y_train, maxiter)

    x_test, y_test = generate_noisy_data(x_min, x_max, (num_x, 2), two_vars_two_outputs)
    y_pred = qnn.predict(x_test)
    loss = mean_squared_error(y_pred, y_test)
    assert loss < 0.11


def sine_two_vars(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sin(np.pi * x[0] * x[1])


@pytest.mark.parametrize(("solver", "maxiter"), [(LBFGS(), 20), (Adam(), 20)])
def test_noisy_sine_two_vars(solver: Optimizer, maxiter: int) -> None:
    x_min = -0.5
    x_max = 0.5
    num_x = 50
    x_train, y_train = generate_noisy_data(x_min, x_max, (num_x, 2), sine_two_vars)
    n_qubit = 4
    depth = 3
    time_step = 0.5
    circuit = create_qcl_ansatz(n_qubit, depth, time_step, 0)
    estimator = create_qulacs_vector_concurrent_estimator()
    gradient_estimator = create_numerical_gradient_estimator(
        create_qulacs_vector_concurrent_parametric_estimator(), delta=1e-10
    )
    qnn = QNNRegressor(circuit, estimator, gradient_estimator, solver)
    qnn.fit(x_train, y_train, maxiter)

    x_test, y_test = generate_noisy_data(x_min, x_max, (num_x, 2), sine_two_vars)
    y_pred = qnn.predict(x_test)
    loss = mean_squared_error(y_pred, y_test)
    assert loss < 0.1


def sine(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sin(np.pi * x[0])


@pytest.mark.parametrize(
    ("solver", "maxiter"),
    [
        (LBFGS(), 20),
        (
            Adam(
                ftol=2e-4,
            ),
            20,
        ),
    ],
)
def test_noisy_sine(solver: Optimizer, maxiter: int) -> None:
    x_min = -1.0
    x_max = 1.0
    num_x = 50
    x_train, y_train = generate_noisy_data(x_min, x_max, (num_x, 1), sine)

    n_qubit = 3
    depth = 3
    time_step = 0.5
    n_outputs = 1
    estimator = create_qulacs_vector_concurrent_estimator()
    gradient_estimator = create_numerical_gradient_estimator(
        create_qulacs_vector_concurrent_parametric_estimator(), delta=1e-10
    )
    circuit = create_qcl_ansatz(n_qubit, depth, time_step, 0)
    qnn = QNNRegressor(circuit, estimator, gradient_estimator, solver)
    qnn.fit(x_train, y_train, maxiter)

    x_test, y_test = generate_noisy_data(x_min, x_max, (num_x, 1), sine)
    y_pred = qnn.predict(x_test)
    loss = mean_squared_error(y_pred, y_test)
    assert loss < 0.03


"""
def test_noisy_sine_gradient_descent() -> None:
    x_min = -1.0
    x_max = 1.0
    num_x = 200
    x_train, y_train = generate_noisy_data(x_min, x_max, (num_x, 1), sine)
    n_qubit = 3
    depth = 15
    batch_size = 50
    epochs = 100
    lr = 0.1
    circuit = create_multi_qubit_param_rotational_ansatz(n_qubit, c_depth=depth)
    solver = GradientDescent()
    qnn = QNNRegressor(circuit, solver, observables_str=["Z 2"])
    loader = DataLoader(x_train, y_train, batch_size=batch_size, shuffle=True, seed=0)
    for _ in range(epochs):
        for x_batch, y_batch in loader:
            qnn.fit(x_batch, y_batch, lr)
    x_test, y_test = generate_noisy_data(x_min, x_max, (num_x, 1), sine)
    y_pred = qnn.predict(x_test)
    error = mean_squared_error(y_pred, y_test)
    assert error < 0.05
"""
