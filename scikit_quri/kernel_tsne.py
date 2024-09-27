import numpy as np
from quri_parts.algo.optimizer import Adam
from scipy.optimize import minimize

EPS_abs = 1e-12
INF = np.inf


class qk_liner_layer:
    def __init__(self, x_train, beta) -> None:
        self.x_train = x_train
        self.beta = beta
        self.sq_dist = self.cdist(x_train, x_train)

    def forward(self, input, alpha):
        """ """
        if np.equal(self.x_train, input):
            sq_dist = self.sq_dist
        else:
            sq_dist = self.cdist(input, self.x_train)
        return np.exp(-self.beta * sq_dist) @ alpha

    def cdist(self, X, Y):
        dist = np.sqrt(np.sum((X[:, np.newaxis] - Y[np.newaxis, :]) ** 2, axis=2))
        return dist


class kernel_tsne:

    def __init__(self, perplexity=30, cost_f="kldiv"):
        self.perplexity = perplexity
        self.Y = None
        self.p = None
        self.q = None

    def binary_search_perplexity(self, sq_distance, perplexity):
        n = len(X)
        print(f"{n=}")
        print(f"{X=}")
        # Maximum number of binary search steps
        max_iter = 100
        eps = 1.0e-10
        full_eps = np.full(n, eps)
        beta = np.full(n, 1.0)
        beta_max = np.full(n, np.inf)
        beta_min = np.full(n, -np.inf)
        logPerp = np.log(perplexity)
        for k in range(max_iter):

            conditional_P = np.exp(-sq_distance * beta.reshape((n, 1)))
            conditional_P[range(n), range(n)] = 0.0
            P_sum = np.sum(conditional_P, axis=1)
            P_sum = np.maximum(P_sum, full_eps)
            conditional_P /= P_sum.reshape((n, 1))
            H = np.log(P_sum) + beta * np.sum(sq_distance * conditional_P, axis=1)
            print(f"{beta=}")
            print(f"{H=}")
            H_diff = H - logPerp
            print(f"{H_diff=}")
            if np.abs(H_diff).max() < eps:
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
            neg_not_inf_flag = np.logical_and(
                np.logical_not(neg_inf_flag), neg_not_inf_flag
            )
            beta[neg_not_inf_flag] = (
                beta[neg_not_inf_flag] + beta_min[neg_not_inf_flag]
            ) / 2.0
            print(f"iter:{k}")
            print(f"distribution:{np.mean(np.sqrt(1/beta))}")
            # break
        return conditional_P

    def joint_probabilities(self, sq_distance, perplexity):
        conditional_P = self.binary_search_perplexity(sq_distance, perplexity)
        # Symmetric SNE ?
        P = conditional_P + conditional_P.T
        P /= np.sum(P)
        return P

    def calc_probabilities_p(self, X):
        print("calculate the distances by Euclidean distance between the data")
        sq_distance = self.cdist(X)
        p_probs = self.joint_probabilities(sq_distance, self.perplexity)
        print(f"{p_probs=}")
        print(f"{p_probs.shape=}")
        return p_probs

    def calc_probabilities_q(self, c_data):
        # Student's t-distribution
        q_tmp = 1 / (1 + self.cdist(c_data))
        n_data = len(c_data)
        q_tmp[range(n_data), range(n_data)] = 0.0
        q_sum = np.sum(q_tmp)
        q_probs = q_tmp / q_sum
        print(f"{q_probs=}")
        print(f"{q_probs.shape=}")
        return q_probs

    def transform(self, X):
        pass

    def init(self, alpha, x_train, beta):
        # * alpha: Y (n_samples, n_components)
        """
        set optimizer and scheduler
        quantum_circuit: (function)
        """
        self.optimizer = Adam()
        self.opt_state = self.optimizer.get_init_state(alpha)
        self.model = qk_liner_layer(x_train, beta)
        return

    def _train_init(self, d_features, d_valid_features=None):
        self.min_loss = INF
        self.min_index = -1
        self.losses = {"KLdiv": []}
        self.loss_custom = []
        self.d_p_prob = self.calc_probabilities_p(d_features)
        # !未実装
        # if d_valid_features is not None:
        #     valid_features = torch.tensor(d_valid_features, requires_grad=False)
        #     self.valid_features = valid_features.to(dev)
        #     self.valid_p_prob = self.calc_probabilities_p(self.valid_features)
        return

    def _train_first_step(self, d_features, d_indexes):
        p_prob = self.d_p_prob[d_indexes][:, d_indexes]  # consider indexes are shuffled

        # tmp_ret = self.model.forward(self.opt_state.params) # y
        # q_prob = self.calc_probabilities_q(tmp_ret)
        # q_prob = np.max(q_prob,EPS_abs)
        # p_prob = np.max(p_prob,EPS_abs)
        def calc_cost(params, p_prob):
            tmp_ret = self.model.forward(p_prob, params)
            q_prob = self.calc_probabilities_q(tmp_ret)
            q_prob = np.max(q_prob, EPS_abs)
            p_prob = np.max(p_prob, EPS_abs)
            return self.kldiv(p_prob, q_prob)

        _calc_cost = lambda x: calc_cost(x, p_prob)

        result = minimize(
            _calc_cost,
        )
        # self.opt_state = self.optimizer.step(self.opt_state,_calc_cost)

        loss = self.kldiv(p_prob, q_prob)
        if self.cost_f == "kldiv":
            self.losses["KLdiv"].append(loss)

    def kldiv(self, p_probs, q_probs):
        C = p_probs * np.log(p_probs / q_probs)
        c = np.sum(C)
        return c

    def calc_cost(self, p_probs, q_probs):

        #
        C = p_probs * np.log(p_probs / q_probs)
        c = np.sum(C)
        print(f"{c=}")
        return c

    def calc_grad(self, p_probs, q_probs, params):
        # params is self.Y
        # y_t_tmp[i][j] = (1+||y[i]-y[j]||^2)^-1
        n = len(p_probs)
        y_t_tmp = 1 / (1 + self.cdist(params))
        y_tile = np.array([np.tile(row, (n, 1)) for row in params])
        y_tile2 = np.tile(params, (n, 1, 1))
        print(f"{params=}")
        print(f"{y_tile=}")
        print(f"{y_tile2=}")
        # y_diff[i][j] = y[i] - y[j]
        y_diff = y_tile - y_tile2
        p_q_diff = p_probs - q_probs
        print(f"{y_diff=}")
        C_expanded = (p_q_diff * y_t_tmp)[:, :, np.newaxis]
        print(f"{y_diff.shape=}")
        grad = 4.0 * np.sum(C_expanded * y_diff, axis=1)
        print(f"{grad=}")
        return grad

    def fit(self, X, n_components=2):
        # ! n_components=2 only
        n = len(X)
        self.Y = np.random.randn(n, n_components)
        self.p = self.calc_probabilities_p(X)
        self.q = self.calc_probabilities_q(self.Y)
        self.calc_cost(self.p, self.q)
        self.calc_grad(self.p, self.q, self.Y)

    def cdist(self, X):
        """
        Calculate the distances by Euclidean distance between the data
        """
        n = len(X)
        Xsq = np.sum(np.square(X), axis=1)
        # sq_distance[i,j]はX[i]とX[j]のユークリッド距離の二乗
        sq_distance = (Xsq.reshape(n, 1) + Xsq) - 2 * np.dot(X, X.T)
        return sq_distance


perplexity = 30
import numpy as np
from sklearn.datasets import load_digits, load_iris
from quri_parts.circuit import UnboundParametricQuantumCircuit, QuantumCircuit
from quri_parts.core.state import (
    quantum_state,
    QuantumState,
    ParametricCircuitQuantumState,
)
from quri_parts.qulacs.overlap_estimator import create_qulacs_vector_overlap_estimator
from quri_parts.circuit.utils.circuit_drawer import draw_circuit
from functools import partial
from typing import Callable
from numpy.typing import NDArray


# data: data for bra
# data_tr: data for ket
# pqc_f: function for create pqc
def _calc_fidelity(j, data, data_tr, pqc_f):
    n_data = len(data)
    fidelities = np.zeros(n_data)
    pqc_bra = pqc_f(data[j])
    state_bra = quantum_state(n_qubits=n_qubits, circuit=pqc_bra)
    estimator = create_qulacs_vector_overlap_estimator()
    for k in range(j + 1):
        pqc_ket = pqc_f(data_tr[k])
        state_ket = quantum_state(n_qubits=n_qubits, circuit=pqc_ket)
        inner_prod = estimator(state_bra, state_ket)
        fidelity = np.abs(inner_prod) ** 2
        fidelities[k] = fidelity[0].real
    return fidelities


# TODO parallelize
# ? Cacheできそう
def calc_fidelity(data, data_tr, pqc_f):
    n_data = len(data)
    fidelities = np.zeros((n_data, n_data))
    for j in range(n_data):
        fidelities[j] = _calc_fidelity(j, data, data_tr, pqc_f)
    return fidelities


if __name__ == "__main__":

    def load_visualize_data(n_sample):
        (X_test, y_test) = load_iris(return_X_y=True)
        return X_test[:n_sample], y_test[:n_sample]

    X, y = load_visualize_data(3)
    tsne = kernel_tsne(perplexity)
    n_qubits = 4
    depth = 1

    def input_quantum_circuit(input, theta, scaling_params):
        # qc = UnboundParametricQuantumCircuit(n_qubits)
        def preprocess_x(x: NDArray[np.float_], index: int) -> float:
            xa = x[index % len(x)]
            return min(1, max(-1, xa))

        qc = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            qc.add_H_gate(i)
        for d in range(depth):
            for i in range(n_qubits):
                qc.add_RY_gate(i, lambda x, i=i: preprocess_x(x, i))
            for i in range(n_qubits):
                qc.add_RX_gate(i, lambda x, i=i: preprocess_x(x, i))
            if d < depth - 1:
                for i in range(n_qubits):
                    qc.add_CZ_gate(i, (i + 1) % n_qubits)
        return qc

    # draw_circuit(input_quantum_circuit([0.1,0.2,0.3],None,None))
    print(X)
    print(
        calc_fidelity(
            X, X, partial(input_quantum_circuit, theta=None, scaling_params=None)
        )
    )
    # print(f"{X=}")
    # print(f"{X.shape=}")
    # tsne.fit(X)
    # alpha = np.random.randn(len(X),2)
    # tsne.init(alpha,X,1.)
    # tsne._train_init(X)
    # tsne._train_first_step(X,[0,1,2])
