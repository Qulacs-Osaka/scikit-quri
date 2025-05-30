# mypy: ignore-errors
from typing import Callable, overload, List
from ..circuit.circuit import LearningCircuit
from numpy.typing import NDArray
import numpy as np
from quri_parts.algo.optimizer import Adam
from quri_parts.qulacs.circuit import convert_circuit
from functools import partial
from quri_parts.qulacs.overlap_estimator import (
    create_qulacs_vector_overlap_estimator,
    _create_qulacs_initial_state,
)
from qulacs.state import inner_product
from qulacs import QuantumState
from quri_parts.core.state import quantum_state, GeneralCircuitQuantumState
import time
import matplotlib.pyplot as plt
from scipy.spatial import distance
from quri_parts.algo.optimizer import OptimizerStatus

EPS_abs = 1e-12


class pqc_f_helper:
    """
    入力データXに対して，量子回路を計算してcacheしておくClass
    """

    def __init__(self, pqs_f: Callable[[NDArray[np.float64]], GeneralCircuitQuantumState]) -> None:
        """
        Args:
            pqs_f: 入力データXを受け取って，量子状態を返す関数
        """
        self.pqs_f = pqs_f
        self.cache = {}

    def get(self, input: NDArray[np.float64]) -> GeneralCircuitQuantumState:
        """
        入力データXに対して，cacheされた量子状態を返す．もしcacheされていない場合は計算してcacheする

        Args:
            input: 入力データX
        Returns:
            GeneralCircuitQuantumState: 量子状態
        """
        hashed = hash(input.tobytes())
        state = self.cache.get(hashed, None)
        if state is None:
            state = self.pqs_f(input)
            self.cache[hashed] = state
        return state


class overlap_estimator:
    """
    quri-partsのoverlap_estimatorの代替Class
    (n_data:500のとき,x60 faster)
    """

    def __init__(self, states: List[GeneralCircuitQuantumState]):
        """
        Args:
            states (List[GeneralCircuitQuantumState]): 量子状態のリスト
        """
        self.states = states
        self.qula_states = np.full(len(states), fill_value=None, dtype=object)

    def _state_to_qula_state(self, state: GeneralCircuitQuantumState) -> QuantumState:
        """量子状態をqulacsのstateに変換

        Args:
            state (GeneralCircuitQuantumState): quri-partsの量子状態
        Returns:
            qulacs_state (QuantumState): qulacsの量子状態
        """
        circuit = convert_circuit(state.circuit)
        qulacs_state = _create_qulacs_initial_state(state)
        circuit.update_quantum_state(qulacs_state)
        return qulacs_state

    def calc_all_qula_states(self):
        """
        cache用に予め全ての量子状態をqulacsのstateに変換
        """
        for i in range(len(self.states)):
            self.qula_states[i] = self._state_to_qula_state(self.states[i])

    def estimate(self, i: int, j: int):
        # ? これi,jじゃなくて数値でhash取った方が使いやすそう
        """与えられた量子状態のi番目とj番目の内積の絶対値の二乗を計算

        Args:
            i (int): 量子状態のindex(ket)
            j (int): 量子状態のindex(bra)
        Returns:
            float: ``|<φi|φj>|^2``
        """
        ket = self.qula_states[i]
        # qulacsのstateを使いまわす
        if ket is None:
            ket = self._state_to_qula_state(self.states[i])
            self.qula_states[i] = ket
        bra = self.qula_states[j]
        if bra is None:
            bra = self._state_to_qula_state(self.states[j])
            self.qula_states[j] = bra
        overlap = inner_product(bra, ket)
        overlap_mag_sqrd = abs(overlap) ** 2
        return overlap_mag_sqrd


# p_ijを計算するTSNE Class
class TSNE:
    """
    基本的なTSNEの実装
    """

    def __init__(self, perplexity=30):
        self.perplexity = perplexity

    def calc_probabilities_p(self, X_train: NDArray[np.float64]) -> NDArray[np.float64]:
        """p行列を計算する

        Args:
            X_train (NDArray[np.float64]): 入力データ
        Returns:
            p_probs (NDArray[np.float64]): t-sneのp行列 (X_trainのサイズ, X_trainのサイズ)
        """
        sq_distance = self.cdist(X_train, X_train)
        p_probs = self.joint_probabilities(sq_distance, self.perplexity)
        return p_probs

    def calc_probabilities_p_state(
        self, X_train_state: List[GeneralCircuitQuantumState]
    ) -> NDArray[np.float64]:
        """p行列を計算する

        Args:
            X_train_state (List[GeneralCircuitQuantumState]): 量子状態に変換した入力データ
        Returns:
            p_probs (NDArray[np.float64]): t-sneのp行列 (X_trainのサイズ, X_trainのサイズ)
        """
        n_data = len(X_train_state)
        sq_distance = np.zeros((n_data, n_data))
        estimator = overlap_estimator(X_train_state)
        # xが量子状態の場合
        for i in range(n_data):
            for j in range(i + 1, n_data):
                inner_prod = estimator.estimate(i, j)
                sq_distance[i][j] = 1 - inner_prod
                sq_distance[j][i] = sq_distance[i][j]
            print("\r", f"{i}/{n_data}", end="")
        print()
        p_probs = self.joint_probabilities(sq_distance, self.perplexity)
        return p_probs

    def calc_probabilities_q(self, c_data: NDArray[np.float64]) -> NDArray[np.float64]:
        """q行列を計算する

        Args:
            c_data (NDArray[np.float64]): 入力データ(論文ではy)
        Returns:
            q_probs (NDArray[np.float64]): t-sneのq行列 (c_dataのサイズ, c_dataのサイズ)
        """
        # Student's t-distribution
        q_tmp = 1 / (1 + self.cdist(c_data, c_data))
        n_data = len(c_data)
        for i in range(n_data):
            q_tmp[i][i] = 0.0
        q_sum = np.sum(q_tmp)
        q_probs = q_tmp / q_sum
        return q_probs

    def joint_probabilities(self, sq_distance: NDArray[np.float64], perplexity: int):
        conditional_P = self.binary_search_perplexity(sq_distance, perplexity)
        P = conditional_P + conditional_P.T
        P /= np.sum(P)
        return P

    def binary_search_perplexity(self, sq_distance: NDArray[np.float64], perplexity: int):
        """
        二分探索で分散をperplexityにする
        """
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

    def cdist(self, X: NDArray[np.float64], X_tr: NDArray[np.float64]):
        """
        Calculate the distances by Euclidean distance between the data
        """
        if X_tr is None:
            raise ValueError("X_tr is None")
        # n = len(X)
        # Xsq = np.sum(np.square(X), axis=1)
        # # sq_distance[i,j]はX[i]とX[j]のユークリッド距離の二乗
        # sq_distance = (Xsq.reshape(n, 1) + Xsq) - 2 * np.dot(X, X.T)
        # sq_distance = np.sqrt(sq_distance)
        sq_distance = distance.cdist(X, X_tr)
        return sq_distance


class quantum_kernel_tsne:
    """
    quantum kernel t-sneで学習するためのClass
    """

    def __init__(self, perplexity=30, max_iter=400):
        self.perplexity = perplexity
        self.max_iter = max_iter
        self.tsne = TSNE(perplexity)
        self.optimizer = Adam(ftol=1e-12)
        self.estimator = create_qulacs_vector_overlap_estimator()
        self.X_train = None

    # pqc_fの設定
    def init(self, pqc_f: Callable[[], LearningCircuit], theta: NDArray[np.float64]) -> None:
        """
        Args:
            pqc_f: 量子回路を返す関数
            theta: 量子回路のパラメータ
        """
        self.pqc_f = pqc_f
        # parametric circuit state
        self.pqs_f = partial(self.input_quantum_state, pqc_f=self.pqc_f, theta=theta)
        self.pqs_f_helper = pqc_f_helper(self.pqs_f)

    def calc_loss(self, p_prob: NDArray[np.float64], q_prob: NDArray[np.float64]):
        """最適化するためのlossを計算

        Args:
            p_prob (NDArray[np.float64]): p_ij
            q_prob (NDArray[np.float64]): q_ij
        """
        # ここでyを計算
        p_prob = np.maximum(p_prob, EPS_abs)
        q_prob = np.maximum(q_prob, EPS_abs)
        loss = self.tsne.kldiv(p_prob, q_prob)
        return loss

    def _calc_grad(
        self, alpha: NDArray[np.float64], p_prob: NDArray[np.float64], fidelity: NDArray[np.float64]
    ):
        y = self.calc_y(fidelity, alpha.reshape(len(alpha) // 2, 2))
        q_prob = self.tsne.calc_probabilities_q(y)
        loss = self.calc_loss(p_prob, q_prob)
        return loss

    def calc_grad(
        self, alpha: NDArray[np.float64], p_prob: NDArray[np.float64], fidelity: NDArray[np.float64]
    ):
        dx = 1e-6
        grads = np.zeros(len(alpha))
        alpha = alpha.copy()
        for i in range(len(alpha)):
            print("\r", f"{i}/{len(alpha)}", end="")
            alpha[i] += dx
            loss_plus = self._calc_grad(alpha, p_prob, fidelity)
            alpha[i] -= 2 * dx
            loss_minus = self._calc_grad(alpha, p_prob, fidelity)
            alpha[i] += dx
            grad = (loss_plus - loss_minus) / (2 * dx)
            grads[i] = grad
        print(f"{grads=}")
        return grads

    def cost_f(
        self,
        alpha: NDArray[np.float64],
        p_prob: NDArray[np.float64],
        fidelity: NDArray[np.float64],
    ):
        # optimizerに1次元配列で渡されるので、2次元に戻す
        y = self.calc_y(fidelity, alpha.reshape(len(alpha) // 2, 2))
        # print(f"{y=}")
        q_prob = self.tsne.calc_probabilities_q(y)
        loss = self.calc_loss(p_prob, q_prob)
        self.cost_f_iter += 1
        if self.cost_f_iter % 100 == 0:
            print("\r", f"iter={self.cost_f_iter} {loss=}", end="")
        return loss

    def generate_X_train_state(self, X_train: NDArray[np.float64]):
        """X_train(NDAarray[np.float64])から量子状態のリストを生成

        Args:
            X_train (NDArray[np.float64]):64入力データ
        Returns:
            X_train_state (List[GeneralCircuitQuantumState]): 量子状態のリスト
        """
        X_train_state = np.zeros(len(X_train), dtype=object)
        for i in range(len(X_train)):
            X_train_state[i] = self.pqs_f_helper.get(X_train[i])
        return X_train_state

    def train(self, X_train: NDArray[np.float64], y_label: NDArray[np.int8], method="Powell"):
        if self.pqc_f is None:
            raise ValueError("please call 'init' before training")
        self.X_train = X_train
        n_data = X_train.shape[0]
        # transformで使う
        self.X_train = X_train
        print("calculating p_ij")
        # p_ijを求める
        # p_probs = self.tsne.calc_probabilities_p(X_train)
        p_probs = self.tsne.calc_probabilities_p_state(self.generate_X_train_state(X_train))
        print("calculating fidelity")
        # fidelity計算
        start = time.perf_counter()
        fidelity = self.calc_fidelity(X_train, X_train, self.pqs_f_helper)
        print(f"elapsed time:{time.perf_counter() - start}")
        cost_f = partial(self.cost_f, p_prob=p_probs, fidelity=fidelity)
        # d=2次元に落とすので2倍
        alpha = np.random.rand(n_data * 2)
        # cost_fの呼び出し回数
        self.cost_f_iter = 0
        self.plot(self.calc_y(fidelity, alpha.reshape(n_data, 2)), y_label, "before")
        if method == "adam":
            self.optimizer_state = self.optimizer.get_init_state(alpha)
            for n_epoch in range(self.max_iter):
                if n_epoch % 10 == 0:
                    print(f"epoch:{n_epoch} loss:{self.optimizer_state.cost}")
                grad_f = lambda alpha: self.calc_grad(alpha, p_probs, fidelity)
                self.optimizer_state = self.optimizer.step(self.optimizer_state, cost_f, grad_f)
                if self.optimizer_state.status == OptimizerStatus.CONVERGED:
                    break
                if self.optimizer_state.status == OptimizerStatus.FAILED:
                    print("failed")
                    break
            self.trained_alpha = self.optimizer_state.params
        elif method == "COBYLA":
            from scipy.optimize import minimize

            result = minimize(cost_f, alpha, method="COBYLA", options={"maxiter": self.max_iter})
            print(result)
            self.trained_alpha = result.x
        elif method == "Powell":
            from scipy.optimize import minimize

            result = minimize(cost_f, alpha, method="Powell", options={"maxfev": self.max_iter})
            print(result)
            self.trained_alpha = result.x

        y = self.calc_y(fidelity, self.trained_alpha.reshape(n_data, 2))
        self.plot(y, y_label, "after")

    def transform(self, X_test: NDArray[np.float64]) -> NDArray[np.float64]:
        """学習したαを使ってyを計算

        Args:
            X_test (NDArray[np.float64]): テストデータ
        Returns:
            y (NDArray[np.float64]): 低次元表現
        """
        fidelity = self.calc_fidelity_all(X_test, self.X_train, self.pqs_f_helper)
        y = self.calc_y(fidelity, self.trained_alpha.reshape(len(self.trained_alpha) // 2, 2))
        return y

    def calc_y(
        self, fidelity: NDArray[np.float64], alpha: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """fidelityとαからyを計算

        Args:
            fidelity: ``|<φi|φj>|^2`` (n_data, n_data)
            alpha: α (n_data, 2)
        """
        return fidelity @ alpha

    # ``|φi,Θ>`` を計算
    def input_quantum_state(
        self,
        input: NDArray[np.float64],
        pqc_f: Callable[[], LearningCircuit],
        theta: NDArray[np.float64],
    ) -> GeneralCircuitQuantumState:
        """
        入力データXとpqc_f,thetaを受け取って、量子状態を計算する
        """
        qc = pqc_f()
        bind_params = qc.generate_bound_params(input, theta)
        circuit_state = quantum_state(n_qubits=qc.n_qubits, circuit=qc.circuit).bind_parameters(
            bind_params
        )
        return circuit_state

    # Parallelに``|<φi|φj>|^2`` 計算するための関数
    # 対称性を持つので、``j<=i`` の場合は計算しない
    # TODO parallelize
    # ? Cacheできそう
    def _calc_fidelity(self, j, data, data_tr, estimator: overlap_estimator):
        n_data = len(data)
        if np.array_equal(data, data_tr):
            fidelities = np.zeros(n_data)
            for k in range(j + 1):
                inner_prod = estimator.estimate(j, k)
                fidelities[k] = inner_prod
        else:
            n_data_offset = n_data
            n_data_tr = len(data_tr)
            fidelities = np.zeros(n_data_tr)
            for k in range(n_data_tr):
                # estimatorは[data,data_tr]なので，offsetを使ってdata_trのindexを計算
                inner_prod = estimator.estimate(j, k + n_data_offset)
                fidelities[k] = inner_prod
        return fidelities

    def calc_fidelity(self, data, data_tr, pqs_f_helper: pqc_f_helper):
        """
        data==data_trの場合，fidelity( ``|<φi|φj>|^2`` ) を計算
        """
        if not np.array_equal(data, data_tr):
            raise ValueError("data and data_tr must be the same")
        n_data = len(data)
        n_data_tr = len(data_tr)
        fidelities = np.zeros((n_data, n_data_tr))
        estimator = overlap_estimator([pqs_f_helper.get(data[i]) for i in range(n_data)])
        # どうせ全部使うので，先に全部計算する
        estimator.calc_all_qula_states()
        for j in range(n_data):
            fidelities[j] = self._calc_fidelity(j, data, data_tr, estimator)
            print("\r", f"{j}/{n_data}", end="")
        fidelities = fidelities + fidelities.T - np.eye(n_data)
        return fidelities

    def calc_fidelity_all(self, data, data_tr, pqs_f_helper: pqc_f_helper):
        """
        data != data_trの場合，fidelity( ``|<φi|φj>|^2`` )を計算
        """
        n_data = len(data)
        n_data_tr = len(data_tr)
        fidelities = np.zeros((n_data, n_data_tr))
        # dataとdata_trの両方の量子状態をestimatorに入れる
        estimator = overlap_estimator(
            [pqs_f_helper.get(x) for x in np.concatenate([data, data_tr])]
        )
        estimator.calc_all_qula_states()
        for j in range(n_data):
            fidelities[j] = self._calc_fidelity(j, data, data_tr, estimator)
            print("\r", f"{j}/{n_data}", end="")
        print()
        return fidelities

    def plot(self, y: NDArray[np.float64], y_label: NDArray[np.int64], title: str):
        """yをplotする

        Args:
            y (NDArray[np.float64]): 低次元表現
            y_label (NDArray[np.int64]): ラベル
            title (str): タイトル
        """
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

        def preprocess_x(x: NDArray[np.float64], index: int) -> float:
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
