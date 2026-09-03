import warnings
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray
from quri_parts.circuit import QuantumCircuit
from scipy.stats import loguniform
from sklearn.base import BaseEstimator as SklearnBaseEstimator, RegressorMixin
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import RandomizedSearchCV


from scikit_quri.backend import BaseSampler
from scikit_quri.circuit import LearningCircuit
from scikit_quri.state.overlap_estimator import OverlapEstimator


class QKRR(RegressorMixin, SklearnBaseEstimator):
    """class to solve regression problems with kernel ridge regressor with a quantum kernel"""

    def __init__(
        self,
        circuit: LearningCircuit,
        n_iteration=10,
        sampler: Optional[BaseSampler] = None,
    ) -> None:
        """
        :param circuit: circuit to generate quantum feature
        :param sampler: sampling backend; a property of the model, not of the data
        """
        if circuit.learning_params_count:
            raise ValueError(
                f"{type(self).__name__} is a kernel method: it evaluates the feature map "
                f"only and never optimizes parameters, so the circuit must have no "
                f"learnable parameters (got {circuit.learning_params_count}). Use a "
                f"data-encoding circuit such as create_ibm_embedding_circuit. Passing a "
                f"trainable ansatz used to fail later with an unrelated IndexError."
            )
        self.sampler = sampler
        self.krr = KernelRidge(kernel="precomputed")
        self.kernel_ridge_tuned: Optional[KernelRidge] = None
        self.circuit = circuit
        self.data_circuits: List[QuantumCircuit] = []
        self.n_qubit: int = circuit.n_qubits
        self.n_iteration = n_iteration
        self.estimator: Optional[OverlapEstimator] = None

    def fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.int_],
        sampler: Optional[BaseSampler] = None,
    ) -> None:
        """
        train the machine.
        :param x: training inputs
        :param y: training teacher values
        :param sampler: deprecated; pass it to the constructor instead
        """
        if sampler is not None:
            warnings.warn(
                "Passing sampler to fit() is deprecated; pass it to the constructor "
                "instead. Configuration in fit() prevents use with scikit-learn's "
                "Pipeline and cross-validation helpers, which call fit(X, y).",
                DeprecationWarning,
                stacklevel=2,
            )
            self.sampler = sampler
        if self.sampler is None:
            raise ValueError(
                "No sampler configured. Pass one to the constructor, e.g. "
                "QKRR(circuit, sampler=QulacsSampler())."
            )

        kar = np.zeros((len(x), len(x)))
        # Reset rather than append: fitting a second time used to keep the circuits
        # from the first fit, so the Gram matrix no longer matched len(x) and the
        # reshape below raised "cannot reshape array of size ... into shape ...".
        self.data_circuits = []
        for i in range(len(x)):
            self.data_circuits.append(self._run_circuit(x[i]))

        self.estimator = OverlapEstimator(self.sampler)
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
