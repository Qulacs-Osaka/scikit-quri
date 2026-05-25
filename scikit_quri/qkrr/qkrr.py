# mypy: ignore-errors
from typing import List

import numpy as np
from numpy.typing import NDArray
from quri_parts.circuit import QuantumCircuit
from quri_parts.core.sampling import ConcurrentSampler
from scipy.stats import loguniform
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import RandomizedSearchCV


from scikit_quri.circuit import LearningCircuit
from scikit_quri.state.overlap_estimator import OverlapEstimator


class QKRR:
    """class to solve regression problems with kernel ridge regressor with a quantum kernel"""

    def __init__(self, circuit: LearningCircuit, n_iteration=10) -> None:
        """
        :param circuit: circuit to generate quantum feature
        """
        self.krr = KernelRidge(kernel="precomputed")
        self.kernel_ridge_tuned = None
        self.circuit = circuit
        self.data_circuits: List[QuantumCircuit] = []
        self.n_qubit: int = circuit.n_qubits
        self.n_iteration = n_iteration
        self.estimator = None

    def fit(self, x: NDArray[np.float64], y: NDArray[np.int_], sampler: ConcurrentSampler) -> None:
        """
        train the machine.
        :param x: training inputs
        :param y: training teacher values
        """
        kar = np.zeros((len(x), len(x)))
        # Compute UΦx to get kernel of `x` and `y`.
        for i in range(len(x)):
            self.data_circuits.append(self._run_circuit(x[i]))

        self.estimator = OverlapEstimator(sampler)
        kar = self.estimator.estimate_concurrent(self.data_circuits, self.data_circuits).reshape(
            len(x), len(x)
        )
        self.krr.fit(kar, y)

        # hyperparameter tuning
        alpha_low = 1e-3
        alpha_high = 1e2
        n_iteration = self.n_iteration
        random_state = 0
        param_distributions = {
            "alpha": loguniform(
                alpha_low, alpha_high
            ),  # Hyperparameter in the cost function for the regularizaton
            # "kernel__length_scale": loguniform(1e-3, 1e3), # Hyperparameter of the Kernel (If we apply the Quantum Kernel, this must be ignored)
            # "kernel__periodicity": loguniform(1e0, 1e1), # For periodic Kernel
        }
        kernel_ridge_tuned = RandomizedSearchCV(
            self.krr,
            param_distributions=param_distributions,
            n_iter=n_iteration,
            random_state=random_state,
        )

        kernel_ridge_tuned.fit(kar, y)
        print(kernel_ridge_tuned.best_params_)
        self.kernel_ridge_tuned = kernel_ridge_tuned

    def predict(self, xs: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        predict y values for each of xs
        :param xs: inputs to make predictions
        :return: List[int], predicted values of y
        """
        if self.kernel_ridge_tuned is None or self.estimator is None:
            raise ValueError("run fit() before predict")

        test_circuits = [self._run_circuit(_xs) for _xs in xs]
        kar = self.estimator.estimate_concurrent(test_circuits, self.data_circuits).reshape(
            len(xs), len(self.data_circuits)
        )
        pred: NDArray[np.float64] = self.kernel_ridge_tuned.predict(kar)
        return pred

    def _run_circuit(self, x: NDArray[np.float64]) -> QuantumCircuit:
        return self.circuit.bind_input_and_parameters(x, np.array([])).get_mutable_copy()
