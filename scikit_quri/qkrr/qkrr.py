from typing import List

import numpy as np
from numpy.typing import NDArray
from quri_parts.core.state import QuantumState, quantum_state
from scipy.stats import loguniform
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import RandomizedSearchCV


from scikit_quri.circuit import LearningCircuit
from scikit_quri.state.overlap_estimator import overlap_estimator


class QKRR:
    """class to solve regression problems with kernel ridge regressor with a quantum kernel"""

    def __init__(self, circuit: LearningCircuit, n_iteration=10) -> None:
        """
        :param circuit: circuit to generate quantum feature
        """
        self.krr = KernelRidge(kernel="precomputed")
        self.kernel_ridge_tuned = None
        self.circuit = circuit
        self.data_states: List[QuantumState] = []
        self.n_qubit: int = circuit.n_qubits
        self.n_iteration = n_iteration

    def run_circuit(self, x: NDArray[np.float64]):
        # ここにはparametrizeされたcircuitは入ってこないはず...
        # circuit = self.circuit.bind_input_and_parameters(x,[])
        circuit = self.circuit.bind_input_and_parameters(x, [])
        state = quantum_state(n_qubits=self.n_qubit, circuit=circuit)
        return state

    def fit(self, x: NDArray[np.float64], y: NDArray[np.int_]) -> None:
        """
        train the machine.
        :param x: training inputs
        :param y: training teacher values
        """
        kar = np.zeros((len(x), len(x)))
        # Compute UΦx to get kernel of `x` and `y`.
        for i in range(len(x)):
            self.data_states.append(self.run_circuit(x[i]))
        self.estimator = overlap_estimator(self.data_states.copy())
        for i in range(len(x)):
            for j in range(len(x)):
                kar[i][j] = self.estimator.estimate(i, j)

        self.krr.fit(kar, y)

        # hyperparameter tuning
        alpha_low = 1e-3
        alpha_high = 1e2
        n_iteration = 5
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
        kar = np.zeros((len(xs), len(self.data_states)))
        new_status = []
        for i in range(len(xs)):
            x_qc = self.run_circuit(xs[i])
            new_status.append(x_qc)
        self.estimator.add_state(new_status)
        offset = len(self.data_states)
        for i in range(len(xs)):
            for j in range(len(self.data_states)):
                kar[i][j] = self.estimator.estimate(offset + i, j)
        prid: NDArray[np.float64] = self.kernel_ridge_tuned.predict(kar)
        return prid
