from functools import partial
from typing import List, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from qulacs import QuantumState as QulacsQuantumState
from quri_parts.algo.optimizer import Optimizer, OptimizerStatus
from quri_parts.core.operator import pauli_label
from quri_parts.core.state import QuantumState, quantum_state
from quri_parts.qulacs.simulator import evaluate_state_to_vector

from scikit_quri.circuit import LearningCircuit


class QNNGenerator:
    def __init__(
        self,
        circuit: LearningCircuit,
        solver: Optimizer,
        kernel_type: Literal["gauss", "exp_hamming", "same"],
        gauss_sigma: float,
        fitting_qubit: int,
    ):
        self.n_qubit = circuit.n_qubits
        self.circuit = circuit
        self.solver = solver
        self.kernel_type = kernel_type
        self.gauss_sigma = gauss_sigma
        self.fitting_qubit = fitting_qubit
        self.theta: None | List[float] = None

        self.observables = [pauli_label(f"Z{i}") for i in range(self.n_qubit)]

    def fit(self, train_data: NDArray[np.float64], maxiter: Optional[int] = None):
        train_scaled = np.zeros(2**self.fitting_qubit)
        for i in train_data:
            train_scaled[i] += 1 / len(train_data)
        return self.fit_direct_distribution(train_scaled, maxiter)

    def fit_direct_distribution(
        self, train_scaled: NDArray[np.float64], maxiter: Optional[int] = None
    ) -> Tuple[float, List[float]]:
        theta_init = 2 * np.pi * np.random.random(self.circuit.learning_params_count)
        optimizer_state = self.solver.get_init_state(theta_init)
        cost_func = partial(self.cost_func, train_scaled=train_scaled)
        grad_func = partial(self._cost_func_grad, train_scaled=train_scaled)
        c = 0
        while maxiter > c:
            optimizer_state = self.solver.step(optimizer_state, cost_func, grad_func)
            print("\r", f"iter:{c}/{maxiter} cost:{optimizer_state.cost=}", end="")

            if optimizer_state.status == OptimizerStatus.CONVERGED:
                break
            if optimizer_state.status == OptimizerStatus.FAILED:
                break

            c += 1
        print("")
        self.trained_param = optimizer_state.params

    def predict(self) -> NDArray[np.float64]:
        y_pred_in = evaluate_state_to_vector(self._predict_inner()).vector
        y_pred_conj = y_pred_in.conjugate()
        data_per: NDArray[np.float64] = np.abs(y_pred_in * y_pred_conj)

        if self.n_qubit != self.fitting_qubit:
            data_per = data_per.reshape(
                (2 ** (self.n_qubit - self.fitting_qubit), 2**self.fitting_qubit)
            )
            data_per = data_per.sum(axis=0)

        return data_per

    def _predict_and_inner(self) -> Tuple[NDArray[np.float64], QuantumState]:
        state = self._predict_inner()
        y_pred_in = evaluate_state_to_vector(state).vector
        y_pred_conj = y_pred_in.conjugate()

        data_per = y_pred_in * y_pred_conj

        if self.n_qubit != self.fitting_qubit:
            data_per = data_per.reshape(
                (2 ** (self.n_qubit - self.fitting_qubit), 2**self.fitting_qubit)
            )
            data_per = data_per.sum(axis=0)
        return (data_per, state)

    def _predict_inner(self) -> QuantumState:
        circuit = self.circuit.bind_input_and_parameters([0], self.theta)
        state = quantum_state(n_qubits=self.n_qubit, circuit=circuit)
        return state

    def conving(self, data_diff: NDArray[np.float64]) -> NDArray[np.float64]:
        # data_diffは、現在の分布ー正しい分布
        # (data_diff) (カーネル行列) (data_diffの行ベクトル)を計算すると、cost_funcになる。
        # ここでは、(data_diff) (カーネル行列)  のベクトルを求める。
        # つまり、確率差ベクトルにカーネル行列を掛ける。
        if self.kernel_type == "gauss":
            beta = -0.5 / self.gauss_sigma
            width = int(4 * np.sqrt(self.gauss_sigma))
            conv_len = width * 2 + 1
            conv_target = np.zeros(conv_len)
            for i in range(conv_len):
                conv_target[i] = np.exp((i - width) ** 2 * beta)
            if conv_len <= 2**self.fitting_qubit:
                conv_diff = np.convolve(data_diff, conv_target, mode="same")
            else:
                conv_diff = np.convolve(data_diff, conv_target)[width:-width]
            return conv_diff
        # elif self.kernel_type == "exp_hamming":
        #     beta = -0.5 / self.gauss_sigma
        #     swap_pena = np.exp(beta)
        #     # ハミング距離の畳み込み演算をします。
        #     # これはバタフライ演算でできます。
        #     # バタフライ演算を行うのにqulacsを使います。
        #     # 注意！ユニタリ的な量子演算ではありません！
        #     # この演算をすることで高速にできる。
        #     # TODO 実機運用
        #     diff_state = QuantumState(self.fitting_qubit)
        #     diff_state
        elif self.kernel_type == "same":
            return data_diff
        else:
            raise NotImplementedError(f"Kernel type {self.kernel_type} is not implemented yet.")

    def cost_func(
        self,
        theta: List[float],
        train_scaled: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        self.theta = theta
        data_diff = self.predict() - train_scaled
        conv_diff = self.conving(data_diff)
        cost = np.dot(data_diff, conv_diff)
        return cost

    def _cost_func_grad(
        self,
        theta: List[float],
        train_scaled: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        self.theta = theta
        (pre, prein) = self._predict_and_inner()
        data_diff = pre - train_scaled
        conv_diff = self.conving(data_diff)

        convconv_diff = np.tile(conv_diff, 2 ** (self.n_qubit - self.fitting_qubit))
        state_vec = evaluate_state_to_vector(prein).vector
        ret = QulacsQuantumState(self.n_qubit)
        ret.load(convconv_diff * state_vec * 4)
        grad = np.array(self.circuit.backprop_innner_product([0], self.theta, ret))
        return -grad
