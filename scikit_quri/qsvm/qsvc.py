from typing import List, Optional

import numpy as np
from numpy.typing import NDArray
from ..circuit import LearningCircuit
from sklearn import svm
from quri_parts.core.state import (
    QuantumState,
    quantum_state,
    GeneralCircuitQuantumState,
)
from quri_parts.circuit import QuantumCircuit
from quri_parts.backend import SamplingBackend
from ..state.overlap_estimator import overlap_estimator
from ..state.overlap_estimator_real_device import overlap_estimator_real_device
from .utils import is_real_device


class QSVC:
    """Class to solve classification problems with support vector machine with quantum kernel"""
    def __init__(self, circuit: LearningCircuit, is_sim: bool = True):
        """
        Parameters:
            circuit: Circuit to generate quantum features.
            is_sim: Whether to run in simulation.
        """
        self.svc = svm.SVC(kernel="precomputed")
        self.circuit = circuit
        # for sim
        self.data_states: List[QuantumState] = []
        # for real devices
        self.data_circuits: List[QuantumCircuit] = []
        self.n_qubit: int = circuit.n_qubits
        self.is_sim = is_sim
        self.estimator = None

    def _run_circuit(self, x: NDArray[np.float64]) -> GeneralCircuitQuantumState:
        """
        Run circuit to generate quantum state.

        Parameters:
            x: Training inputs.

        Returns:
            state: Quantum state.
        """
        # ここにはparametrizeされたcircuitは入ってこないはず...
        circuit = self.circuit.bind_input_and_parameters(x, np.array([]))
        state = quantum_state(n_qubits=self.n_qubit, circuit=circuit)
        return state

    def fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.int_],
        n_shots: int = 1000,
        sampling_backend: Optional[SamplingBackend] = None,
    ):
        """
        Train the machine.
        
        Parameters:
            x: Training inputs.
            y: Training labels.
            n_shots: Numbers of times the quantum circuit will be sampled.
            sampling_backend: Optional, SamplingBackend if run in simulation.
        """
        if not self.is_sim and sampling_backend is None:
            raise ValueError("sampling_backend is required for real devices")

        kar = np.zeros((len(x), len(x)))
        # Compute UΦx to get kernel of `x` and `y`.
        for i in range(len(x)):
            state = self._run_circuit(x[i])
            self.data_states.append(state)
            self.data_circuits.append(state.circuit.get_mutable_copy())

        if is_real_device(sampling_backend, self.is_sim):
            self.estimator = overlap_estimator_real_device(
                self.data_circuits.copy(), sampling_backend, n_shots
            )
        else:
            self.estimator = overlap_estimator(self.data_states.copy())

        for i in range(len(x)):
            for j in range(len(x)):
                kar[i][j] = self.estimator.estimate(i, j)
            print("\r", f"{i}/{len(x)}", end="")
        print("")
        self.svc.fit(kar, y)

    def predict(self, xs: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict labels of given data.

        Parameters:
            xs: Inputs to predict labels.
        
        Returns:
            Predicted labels.
        """
        if self.estimator is None:
            raise ValueError("run fit() before predict")
        kar = np.zeros((len(xs), len(self.data_states)))
        new_states = []
        new_circuits = []
        for i in range(len(xs)):
            x_qc = self._run_circuit(xs[i])
            new_states.append(x_qc)
            new_circuits.append(x_qc.circuit.get_mutable_copy())
        if self.is_sim:
            self.estimator.add_data(new_states)
        else:
            self.estimator.add_data(new_circuits)
        offset = len(self.data_states)
        for i in range(len(xs)):
            for j in range(len(self.data_states)):
                kar[i][j] = self.estimator.estimate(offset + i, j)
            print("\r", f"{i}/{len(xs)}", end="")
        print("")
        pred: NDArray[np.float64] = self.svc.predict(kar)
        return pred
