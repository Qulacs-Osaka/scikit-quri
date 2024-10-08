from typing import Callable
from ..circuit.circuit import LearningCircuit
from numpy.typing import NDArray
import numpy as np
from quri_parts.algo.optimizer import Adam
from functools import partial
from quri_parts.qulacs.overlap_estimator import create_qulacs_vector_overlap_estimator
from quri_parts.core.state import quantum_state, GeneralCircuitQuantumState
import time
import matplotlib.pyplot as plt

EPS_abs = 1e-12


# pqc_fを入力によってcacheするclass
class pqc_f_helper:
    def __init__(self, pqs_f: Callable[[NDArray[np.float_]], GeneralCircuitQuantumState]):
        self.pqs_f = pqs_f
        self.cache = {}

    def get(self, input: NDArray[np.float_]):
        hashed = hash(input.tobytes())
        state = self.cache.get(hashed, None)
        if state is None:
            state = self.pqs_f(input)
            self.cache[hashed] = state
        return state


# p_ijを計算するTSNE Class
class TSNE:
    def __init__(self, perplexity=30):
        self.perplexity = perplexity

    def calc_probabilities_p(self, X_train: NDArray[np.float_]):
        sq_distance = self.cdist(X_train)
        p_probs = self.joint_probabilities(sq_distance, self.perplexity)
        return p_probs

    def calc_probabilities_q(self, c_data: NDArray[np.float_]) -> NDArray[np.float_]:
        # Student's t-distribution
        q_tmp = 1 / (1 + self.cdist(c_data))
        n_data = len(c_data)
        for i in range(n_data):
            q_tmp[i][i] = 0.0
        q_sum = np.sum(q_tmp)
        q_probs = q_tmp / q_sum
        return q_probs

    def joint_probabilities(self, sq_distance: NDArray[np.float_], perplexity: int):
        conditional_P = self.binary_search_perplexity(sq_distance, perplexity)
        P = conditional_P + conditional_P.T
        P /= np.sum(P)
        return P

    def binary_search_perplexity(self, sq_distance: NDArray[np.float_], perplexity: int):
        PERPLEXITY_TOLERANCE = 1e-5
        n = sq_distance.shape[0]
        # Maximum number of binary search steps
        max_iter = 100
        eps = 1.0e-10
        full_eps = np.full(n, eps)
        beta = np.full(n, 1.0)
        beta_max = np.full(n, np.inf)
        beta_min = np.full(n, -np.inf)
        logPerp = np.log(perplexity)
        for _ in range(max_iter):
            conditional_P = np.exp(-sq_distance * beta.reshape((n, 1)))
            conditional_P[range(n), range(n)] = 0.0
            P_sum = np.sum(conditional_P, axis=1)
            P_sum = np.maximum(P_sum, full_eps)
            conditional_P /= P_sum.reshape((n, 1))
            H = np.log(P_sum) + beta * np.sum(sq_distance * conditional_P, axis=1)
            H_diff = H - logPerp
            if np.abs(H_diff).max() < PERPLEXITY_TOLERANCE:
                break

            # 二分探索
            # beta_min
            pos_flag = np.logical_and((H_diff > 0.0), (np.abs(H_diff) > eps))
            beta_min[pos_flag] = beta[pos_flag]
            inf_flag = np.logical_and(pos_flag, (beta_max == np.inf))
            beta[inf_flag] *= 2.0
            not_inf_flag = np.logical_and((H_diff > 0.0), (beta_max != np.inf))
            not_inf_flag = np.logical_and(np.logical_not(inf_flag), not_inf_flag)
            beta[not_inf_flag] = (beta[not_inf_flag] + beta_max[not_inf_flag]) / 2.0
            # beta_max
            neg_flag = np.logical_and((H_diff <= 0.0), np.abs(H_diff) > eps)
            beta_max[neg_flag] = beta[neg_flag]
            neg_inf_flag = np.logical_and(neg_flag, (beta_min == -np.inf))
            beta[neg_inf_flag] /= 2.0
            neg_not_inf_flag = np.logical_and((H_diff <= 0.0), (beta_min != -np.inf))
            neg_not_inf_flag = np.logical_and(np.logical_not(neg_inf_flag), neg_not_inf_flag)
            beta[neg_not_inf_flag] = (beta[neg_not_inf_flag] + beta_min[neg_not_inf_flag]) / 2.0
        return conditional_P

    def kldiv(self, p_probs, q_probs):
        C = p_probs * np.log(p_probs / q_probs)
        c = np.sum(C)
        return c

    def cdist(self, X: NDArray[np.float_]):
        """
        Calculate the distances by Euclidean distance between the data
        """
        n = len(X)
        Xsq = np.sum(np.square(X), axis=1)
        # sq_distance[i,j]はX[i]とX[j]のユークリッド距離の二乗
        sq_distance = (Xsq.reshape(n, 1) + Xsq) - 2 * np.dot(X, X.T)
        sq_distance = np.sqrt(sq_distance)
        return sq_distance


class quantum_kernel_tsne:
    def __init__(self, perplexity=30, max_iter=400):
        self.perplexity = perplexity
        self.max_iter = max_iter
        self.tsne = TSNE(perplexity)
        self.optimizer = Adam(ftol=1e-12)
        self.estimator = create_qulacs_vector_overlap_estimator()

    # pqc_fの設定
    def init(self, pqc_f: Callable[[], LearningCircuit], theta: NDArray[np.float_]):
        self.pqc_f = pqc_f
        # parametric circuit state
        self.pqs_f = partial(self.input_quantum_state, pqc_f=self.pqc_f, theta=theta)
        self.pqs_f_helper = pqc_f_helper(self.pqs_f)

    def calc_loss(self, p_prob: NDArray[np.float_], q_prob: NDArray[np.float_]):
        # ここでyを計算
        p_prob = np.maximum(p_prob, EPS_abs)
        q_prob = np.maximum(q_prob, EPS_abs)
        loss = self.tsne.kldiv(p_prob, q_prob)
        return loss

    def cost_f(
        self,
        alpha: NDArray[np.float_],
        p_prob: NDArray[np.float_],
        fidelity: NDArray[np.float_],
    ):
        # optimizerに1次元配列で渡されるので、2次元に戻す
        y = self.calc_y(fidelity, alpha.reshape(len(alpha) // 2, 2))
        q_prob = self.tsne.calc_probabilities_q(y)
        loss = self.calc_loss(p_prob, q_prob)
        # print(f"{loss=}")
        return loss

    def train(self, X_train: NDArray[np.float_], y_label, method="adam"):
        if self.pqc_f is None:
            raise ValueError("please call 'init' before training")
        n_data = X_train.shape[0]
        print("calculating p_ij")
        # p_ijを求める
        p_probs = self.tsne.calc_probabilities_p(X_train)
        print("calculating fidelity")
        # fidelity計算
        start = time.perf_counter()
        fidelity = self.calc_fidelity(X_train, X_train, self.pqs_f_helper)
        print(f"elapsed time:{time.perf_counter()-start}")
        cost_f = partial(self.cost_f, p_prob=p_probs, fidelity=fidelity)
        # d=2次元に落とすので2倍
        alpha = np.random.rand(n_data * 2)
        self.plot(self.calc_y(fidelity, alpha.reshape(n_data, 2)), y_label, "before")
        if method == "adam":
            self.optimizer_state = self.optimizer.get_init_state(alpha)
            for n_epoch in range(self.max_iter):
                if n_epoch % 10 == 0:
                    print(f"epoch:{n_epoch}")
                self.optimizer_state = self.optimizer.step(self.optimizer_state, cost_f, None)
        elif method == "COBYLA":
            from scipy.optimize import minimize

            result = minimize(cost_f, alpha, method="COBYLA", options={"maxiter": self.max_iter})
            print(result)
            self.trained_alpha = result.x

        y = self.calc_y(fidelity, self.trained_alpha.reshape(n_data, 2))
        self.plot(y, y_label, "after")

    def transform(self, X_train: NDArray[np.float_]):
        pass

    """
    @param fidelity: |<φi|φj>|^2 (n_data, n_data)
    @param alpha: α (n_data, 2)
    """

    def calc_y(self, fidelity: NDArray[np.float_], alpha: NDArray[np.float_]) -> NDArray[np.float_]:
        fidelity = (fidelity + fidelity.T) / 2.0
        return fidelity @ alpha

    # |φi,Θ>を計算
    def input_quantum_state(
        self,
        input: NDArray[np.float_],
        pqc_f: Callable[[], LearningCircuit],
        theta: NDArray[np.float_],
    ) -> GeneralCircuitQuantumState:
        qc = pqc_f()
        bind_params = qc.generate_bound_params(input, theta)
        circuit_state = quantum_state(n_qubits=qc.n_qubits, circuit=qc.circuit).bind_parameters(
            bind_params
        )
        return circuit_state

    # Parallelに|<φi|φj>|^2計算するための関数
    # 対称性を持つので、j<=iの場合は計算しない
    # TODO parallelize
    # ? Cacheできそう
    def _calc_fidelity(self, j, data, data_tr, pqs_f_helper: pqc_f_helper):
        n_data = len(data)
        fidelities = np.zeros(n_data)
        # TODO こいつをcacheに
        state_bra = pqs_f_helper.get(data[j])
        for k in range(j + 1):
            state_ket = pqs_f_helper.get(data_tr[k])
            inner_prod = self.estimator(state_bra, state_ket)
            fidelity = inner_prod
            fidelities[k] = fidelity[0].real
        return fidelities

    def calc_fidelity(self, data, data_tr, pqs_f_helper: pqc_f_helper):
        n_data = len(data)
        fidelities = np.zeros((n_data, n_data))
        for j in range(n_data):
            fidelities[j] = self._calc_fidelity(j, data, data_tr, pqs_f_helper)
        return fidelities

    def plot(self, y: NDArray[np.float_], y_label: NDArray[np.int_], title: str):
        for i in np.unique(y_label):
            plt.scatter(y[:, 0][y_label == i], y[:, 1][y_label == i])
        plt.title(title)
        plt.show()


if __name__ == "__main__":
    from quri_parts.circuit import H, CZ
    from sklearn.datasets import load_digits
    from sklearn.preprocessing import MinMaxScaler

    def create_quantum_circuit():
        qc = LearningCircuit(n_qubits)

        def preprocess_x(x: NDArray[np.float_], index: int) -> float:
            xa = x[index % len(x)]
            return min(1, max(-1, xa))

        for i in range(n_qubits):
            qc.add_gate(H(i))
        for d in range(depth):
            for i in range(n_qubits):
                qc.add_input_RY_gate(i, lambda x, i=i: preprocess_x(x, i))
            for i in range(n_qubits):
                qc.add_input_RX_gate(i, lambda x, i=i: preprocess_x(x, i))
            if d < depth - 1:
                for i in range(n_qubits):
                    qc.add_gate(CZ(i, (i + 1) % n_qubits))
        return qc

    X_train, y_train = load_digits(return_X_y=True)
    X_train = X_train / 16.0
    X_train = X_train[:500]
    y_train = y_train[:500]
    scaler = MinMaxScaler((0, np.pi / 2))
    n_qubits = 12
    depth = 1

    X_train = scaler.fit_transform(X_train)
    qk_tsne = quantum_kernel_tsne(max_iter=1000)
    qk_tsne.init(create_quantum_circuit, [])
    qk_tsne.train(X_train, y_train, method="COBYLA")
