# mypy: ignore-errors
from typing import Callable, List
from ..circuit.circuit import LearningCircuit
from numpy.typing import NDArray
import numpy as np
from quri_parts.algo.optimizer import Adam
from quri_parts.qulacs.circuit import convert_circuit
from functools import partial, wraps
from quri_parts.qulacs.overlap_estimator import _create_qulacs_initial_state
from qulacs import QuantumState
from quri_parts.core.state import quantum_state, GeneralCircuitQuantumState
import time
from scipy.spatial import distance
from quri_parts.algo.optimizer import OptimizerStatus

EPS_abs = 1e-12


def _quiet_accelerate_fp(func):
    """Suppress spurious FP-flag RuntimeWarnings from numpy matmul on Apple Accelerate.

    On macOS the Accelerate BLAS sets divide/overflow/invalid floating-point flags
    inside ``@`` / matmul even when the result is finite, producing noisy (harmless)
    RuntimeWarnings. This decorator scopes the suppression to the decorated function,
    so genuine FP issues in other code paths remain visible.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return func(*args, **kwargs)

    return wrapper


class pqc_f_helper:
    """Helper class that evaluates and caches quantum states for input data."""

    def __init__(self, pqs_f: Callable[[NDArray[np.float64]], GeneralCircuitQuantumState]) -> None:
        """
        Args:
            pqs_f: A function that takes an input array and returns a quantum state.
        """
        self.pqs_f = pqs_f
        self.cache = {}

    def get(self, input: NDArray[np.float64]) -> GeneralCircuitQuantumState:
        """Return the cached quantum state for the given input, computing it if not yet cached.

        Args:
            input: Input data array.

        Returns:
            Quantum state corresponding to the input.
        """
        hashed = hash(input.tobytes())
        state = self.cache.get(hashed, None)
        if state is None:
            state = self.pqs_f(input)
            self.cache[hashed] = state
        return state


class overlap_estimator:
    """Materializes quri-parts quantum states into qulacs state vectors.

    Holds a list of states and converts them to qulacs ``QuantumState`` objects
    (cached in ``qula_states``). Used by :func:`fidelity_gram` / :func:`fidelity_cross`
    to obtain the raw state vectors for a single vectorized overlap computation.
    """

    def __init__(self, states: List[GeneralCircuitQuantumState]):
        """
        Args:
            states: List of quantum states to materialize.
        """
        self.states = states
        self.qula_states = np.full(len(states), fill_value=None, dtype=object)

    def _state_to_qula_state(self, state: GeneralCircuitQuantumState) -> QuantumState:
        """Convert a quri-parts quantum state to a qulacs QuantumState.

        Args:
            state: quri-parts quantum state.

        Returns:
            Equivalent qulacs QuantumState.
        """
        circuit = convert_circuit(state.circuit)
        qulacs_state = _create_qulacs_initial_state(state)
        circuit.update_quantum_state(qulacs_state)
        return qulacs_state

    def calc_all_qula_states(self):
        """Convert and cache the qulacs state vector for every input state."""
        for i in range(len(self.states)):
            self.qula_states[i] = self._state_to_qula_state(self.states[i])


def _state_vectors(states: List[GeneralCircuitQuantumState]) -> NDArray[np.complex128]:
    """Materialize a list of quantum states into a stacked state-vector matrix.

    Args:
        states: Quantum states to evaluate.

    Returns:
        Array of shape (len(states), 2**n_qubits) whose i-th row is the i-th state vector.
    """
    est = overlap_estimator(states)
    est.calc_all_qula_states()
    return np.stack([qs.get_vector() for qs in est.qula_states])


@_quiet_accelerate_fp
def fidelity_gram(states: List[GeneralCircuitQuantumState]) -> NDArray[np.float64]:
    """Compute the symmetric fidelity matrix |⟨φi|φj⟩|² for all pairs in one BLAS call.

    The diagonal is exactly 1 for normalized states.

    Args:
        states: Quantum states.

    Returns:
        Fidelity matrix of shape (n, n).
    """
    vectors = _state_vectors(states)
    return np.abs(vectors.conj() @ vectors.T) ** 2


@_quiet_accelerate_fp
def fidelity_cross(
    states: List[GeneralCircuitQuantumState],
    states_tr: List[GeneralCircuitQuantumState],
) -> NDArray[np.float64]:
    """Compute the rectangular fidelity matrix |⟨φi|ψj⟩|² between two sets of states.

    Args:
        states: Query states (rows).
        states_tr: Reference states (columns).

    Returns:
        Fidelity matrix of shape (len(states), len(states_tr)).
    """
    vectors = _state_vectors(states)
    vectors_tr = _state_vectors(states_tr)
    return np.abs(vectors.conj() @ vectors_tr.T) ** 2


class TSNE:
    """Basic t-SNE implementation for computing p and q probability matrices."""

    def __init__(self, perplexity=30):
        self.perplexity = perplexity

    def calc_probabilities_p(self, X_train: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the t-SNE joint probability matrix P from Euclidean distances.

        Args:
            X_train: Input data of shape (n_samples, n_features).

        Returns:
            Symmetric joint probability matrix P of shape (n_samples, n_samples).
        """
        sq_distance = self.cdist(X_train, X_train)
        p_probs = self.joint_probabilities(sq_distance, self.perplexity)
        return p_probs

    def calc_probabilities_p_state(
        self, X_train_state: List[GeneralCircuitQuantumState]
    ) -> NDArray[np.float64]:
        """Compute the t-SNE joint probability matrix P from quantum state overlaps.
        Uses 1 - |⟨φi|φj⟩|² as the distance metric between quantum states.

        Args:
            X_train_state: List of quantum states corresponding to the training inputs.

        Returns:
            Symmetric joint probability matrix P of shape (n_samples, n_samples).
        """
        return self.calc_probabilities_p_from_fidelity(fidelity_gram(X_train_state))

    def calc_probabilities_p_from_fidelity(
        self, fidelity: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute the joint probability matrix P from a precomputed fidelity matrix.

        Uses 1 - |⟨φi|φj⟩|² as the (squared) distance between quantum states. Kept
        separate from the fidelity computation so callers that already hold the
        fidelity matrix (e.g. the embedding kernel) need not recompute it.

        Args:
            fidelity: Symmetric fidelity matrix |⟨φi|φj⟩|² of shape (n, n).

        Returns:
            Symmetric joint probability matrix P of shape (n, n).
        """
        sq_distance = 1.0 - fidelity
        # Diagonal must be zero: floating-point noise can leave it slightly off.
        np.fill_diagonal(sq_distance, 0.0)
        return self.joint_probabilities(sq_distance, self.perplexity)

    def calc_probabilities_q(self, c_data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the t-SNE joint probability matrix Q from the low-dimensional embedding.
        Uses the Student's t-distribution as the similarity kernel.

        Args:
            c_data: Low-dimensional embedding (called y in the original paper),
                of shape (n_samples, n_components).

        Returns:
            Symmetric joint probability matrix Q of shape (n_samples, n_samples).
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
        """Compute the symmetric joint probability matrix from pairwise distances.

        Args:
            sq_distance: Pairwise distance matrix of shape (n_samples, n_samples).
            perplexity: Target perplexity for the conditional distributions.

        Returns:
            Symmetric joint probability matrix of shape (n_samples, n_samples).
        """
        conditional_P = self.binary_search_perplexity(sq_distance, perplexity)
        P = conditional_P + conditional_P.T
        P /= np.sum(P)
        return P

    def binary_search_perplexity(self, sq_distance: NDArray[np.float64], perplexity: int):
        """Find the Gaussian kernel bandwidth for each point via binary search
        so that the perplexity of the conditional distribution matches the target.
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
        """Compute the KL divergence KL(P || Q).

        Args:
            p_probs: Reference probability matrix P.
            q_probs: Approximate probability matrix Q.

        Returns:
            Scalar KL divergence value.
        """
        C = p_probs * np.log(p_probs / q_probs)
        c = np.sum(C)
        return c

    def cdist(self, X: NDArray[np.float64], X_tr: NDArray[np.float64]):
        """Compute pairwise SQUARED Euclidean distances between rows of X and X_tr.

        t-SNE uses squared distances in both the high-dimensional Gaussian kernel
        (``exp(-||x_i - x_j||^2 * beta)``) and the low-dimensional Student-t kernel
        (``(1 + ||y_i - y_j||^2)^-1``), so callers expect ``sq_distance`` here.

        Args:
            X: Array of shape (n_samples, n_features).
            X_tr: Array of shape (m_samples, n_features).

        Returns:
            Squared-distance matrix of shape (n_samples, m_samples).
        """
        if X_tr is None:
            raise ValueError("X_tr is None")
        sq_distance = distance.cdist(X, X_tr, metric="sqeuclidean")
        return sq_distance


class quantum_kernel_tsne:
    """t-SNE using a quantum kernel as the similarity measure in the high-dimensional space."""

    def __init__(self, perplexity=30, max_iter=400):
        self.perplexity = perplexity
        self.max_iter = max_iter
        self.tsne = TSNE(perplexity)
        self.optimizer = Adam(ftol=1e-12)
        self.X_train = None
        # Constant KL term sum_ij p log p; set in train() once P is known.
        self._p_log_p_sum = 0.0

    def init(self, pqc_f: Callable[[], LearningCircuit], theta: NDArray[np.float64]) -> None:
        """Set up the parametric quantum circuit used to encode input data.

        Args:
            pqc_f: A factory function that returns a new LearningCircuit instance.
            theta: Parameter vector for the quantum circuit.
        """
        self.pqc_f = pqc_f
        self.pqs_f = partial(self.input_quantum_state, pqc_f=self.pqc_f, theta=theta)
        self.pqs_f_helper = pqc_f_helper(self.pqs_f)

    def calc_loss(self, p_prob: NDArray[np.float64], q_prob: NDArray[np.float64]):
        """Compute the KL divergence loss KL(P || Q) used as the optimization objective.

        Args:
            p_prob: High-dimensional joint probability matrix P.
            q_prob: Low-dimensional joint probability matrix Q.

        Returns:
            Scalar KL divergence loss value.
        """
        p_prob = np.maximum(p_prob, EPS_abs)
        q_prob = np.maximum(q_prob, EPS_abs)
        loss = self.tsne.kldiv(p_prob, q_prob)
        return loss

    @_quiet_accelerate_fp
    def _kl_loss_from_y(self, y: NDArray[np.float64], p_prob: NDArray[np.float64]) -> np.float64:
        """KL(P || Q) loss evaluated directly from the low-dimensional embedding ``y``.

        Uses the identity (with ``Q`` the Student-t joint distribution)

            KL(P || Q) = sum_ij p log p  +  sum_ij p log(1 + d_ij^2)  +  log Z,

        where ``d_ij^2 = ||y_i - y_j||^2`` and ``Z = sum_{i!=j} 1/(1 + d_ij^2)``.
        The first term is constant during optimization and is cached in
        ``self._p_log_p_sum``. This avoids forming ``Q`` explicitly each call
        (no normalization, no ``p/q`` division, no clamping, a single ``log1p``),
        which is the dominant cost of the Powell/COBYLA inner loop.

        Args:
            y: Low-dimensional embedding of shape (n_samples, 2).
            p_prob: High-dimensional joint probability matrix P (normalized, sum 1).

        Returns:
            Scalar KL divergence loss value.
        """
        t = self.tsne.cdist(y, y)  # squared distances d^2 (reused as 1 + d^2 below)
        t += 1.0  # t = 1 + d^2
        num = 1.0 / t
        np.fill_diagonal(num, 0.0)
        Z = num.sum()
        np.log(t, out=t)  # in-place: t = log(1 + d^2); avoids a temporary array
        # Diagonal contributes nothing: log(1) = 0 and p has a zero diagonal.
        # np.dot over raveled arrays is a single BLAS reduction (no temporary).
        cross = np.dot(p_prob.ravel(), t.ravel())
        return self._p_log_p_sum + cross + np.log(Z)

    @_quiet_accelerate_fp
    def _grad_y(self, y: NDArray[np.float64], p_prob: NDArray[np.float64]) -> NDArray[np.float64]:
        """Analytic gradient of KL(P || Q) with respect to the embedding ``y``.

        Standard t-SNE gradient (van der Maaten & Hinton, 2008):

            dC/dy_i = 4 * sum_j (p_ij - q_ij) * (1 + ||y_i - y_j||^2)^-1 * (y_i - y_j)

        with ``q_ij = num_ij / Z``, ``num_ij = (1 + ||y_i - y_j||^2)^-1`` and
        ``Z = sum_{k!=l} num_kl``. Computed in O(n^2) closed form (no finite differences).

        Args:
            y: Low-dimensional embedding of shape (n_samples, 2).
            p_prob: High-dimensional joint probability matrix P (normalized, sum 1).

        Returns:
            Gradient dC/dy of shape (n_samples, 2).
        """
        num = 1.0 / (1.0 + self.tsne.cdist(y, y))
        np.fill_diagonal(num, 0.0)
        Z = num.sum()
        q = num / Z
        # L_ij = (p_ij - q_ij) * num_ij; the gradient is 4 * sum_j L_ij (y_i - y_j).
        L = (p_prob - q) * num
        return 4.0 * (L.sum(axis=1)[:, None] * y - L @ y)

    @_quiet_accelerate_fp
    def calc_grad(
        self, alpha: NDArray[np.float64], p_prob: NDArray[np.float64], fidelity: NDArray[np.float64]
    ):
        """Analytic gradient of the loss with respect to the embedding coefficients ``alpha``.

        Since ``y = fidelity @ alpha``, the chain rule gives
        ``dC/dalpha = fidelity^T @ dC/dy`` (``fidelity`` is symmetric for the train kernel).
        This replaces the former central-difference gradient, which cost O(n) loss
        evaluations per gradient and made gradient-based optimizers impractical.

        Args:
            alpha: Flattened embedding coefficients of shape (n_samples * 2,).
            p_prob: High-dimensional joint probability matrix P.
            fidelity: Pairwise fidelity matrix of shape (n_samples, n_samples).

        Returns:
            Flattened gradient of the same shape as ``alpha``.
        """
        y = self.calc_y(fidelity, alpha.reshape(len(alpha) // 2, 2))
        grad_y = self._grad_y(y, p_prob)
        grad_alpha = fidelity.T @ grad_y
        return grad_alpha.ravel()

    @_quiet_accelerate_fp
    def calc_loss_grad(
        self, alpha: NDArray[np.float64], p_prob: NDArray[np.float64], fidelity: NDArray[np.float64]
    ):
        """Joint loss and gradient w.r.t. ``alpha``, for gradient-based optimizers.

        Computing both together shares the ``d^2`` / ``num`` / ``Z`` work, so a
        gradient step costs a single ``cdist(y, y)`` instead of two (one for the
        value and one for the jacobian). Used by the ``L-BFGS-B`` path via
        ``scipy.optimize.minimize(..., jac=True)``.

        Args:
            alpha: Flattened embedding coefficients of shape (n_samples * 2,).
            p_prob: High-dimensional joint probability matrix P (normalized, sum 1).
            fidelity: Pairwise fidelity matrix of shape (n_samples, n_samples).

        Returns:
            Tuple ``(loss, grad_alpha)`` where ``grad_alpha`` is flattened like ``alpha``.
        """
        y = self.calc_y(fidelity, alpha.reshape(len(alpha) // 2, 2))
        d2 = self.tsne.cdist(y, y)
        num = 1.0 / (1.0 + d2)
        np.fill_diagonal(num, 0.0)
        Z = num.sum()
        # loss = sum p log p + sum p log(1 + d^2) + log Z  (see _kl_loss_from_y)
        loss = self._p_log_p_sum + np.dot(p_prob.ravel(), np.log1p(d2).ravel()) + np.log(Z)
        # grad: dC/dy_i = 4 sum_j (p_ij - q_ij) num_ij (y_i - y_j),  then chain rule.
        L = (p_prob - num / Z) * num
        grad_y = 4.0 * (L.sum(axis=1)[:, None] * y - L @ y)
        return loss, (fidelity.T @ grad_y).ravel()

    def cost_f(
        self,
        alpha: NDArray[np.float64],
        p_prob: NDArray[np.float64],
        fidelity: NDArray[np.float64],
    ):
        """Cost function passed to the optimizer.

        Args:
            alpha: Flattened embedding coefficients of shape (n_samples * 2,).
                The optimizer passes a 1-D array; it is reshaped to (n_samples, 2) internally.
            p_prob: High-dimensional joint probability matrix P.
            fidelity: Pairwise fidelity matrix of shape (n_samples, n_samples).

        Returns:
            Scalar KL divergence loss value.
        """
        # Reshape from 1-D (as passed by the optimizer) to (n_samples, 2)
        y = self.calc_y(fidelity, alpha.reshape(len(alpha) // 2, 2))
        return self._kl_loss_from_y(y, p_prob)

    def generate_X_train_state(self, X_train: NDArray[np.float64]):
        """Generate quantum states for all training inputs using the cached circuit evaluator.

        Args:
            X_train: Training input array of shape (n_samples, n_features).

        Returns:
            Array of GeneralCircuitQuantumState objects of shape (n_samples,).
        """
        X_train_state = np.zeros(len(X_train), dtype=object)
        for i in range(len(X_train)):
            X_train_state[i] = self.pqs_f_helper.get(X_train[i])
        return X_train_state

    def train(self, X_train: NDArray[np.float64], y_label: NDArray[np.int8], method="Powell"):
        """Fit the quantum kernel t-SNE embedding.

        Args:
            X_train: Training input array of shape (n_samples, n_features).
            y_label: Class labels of shape (n_samples,). Used only for plotting.
            method: Optimization method. One of ``"L-BFGS-B"``, ``"adam"``, ``"COBYLA"``,
                or ``"Powell"``. ``"L-BFGS-B"`` and ``"adam"`` use the analytic
                gradient and converge in far fewer evaluations than the gradient-free
                ``"Powell"`` / ``"COBYLA"``. Defaults to ``"Powell"``.
        """
        if self.pqc_f is None:
            raise ValueError("please call 'init' before training")
        # transformで使う
        self.X_train = X_train
        n_data = X_train.shape[0]
        print("calculating fidelity")
        # The fidelity Gram matrix is both the high-dimensional similarity used for
        # P and the embedding kernel, so compute it once and reuse it.
        start = time.perf_counter()
        fidelity = self.calc_fidelity(X_train, X_train, self.pqs_f_helper)
        print(f"elapsed time:{time.perf_counter() - start}")
        print("calculating p_ij")
        p_probs = self.tsne.calc_probabilities_p_from_fidelity(fidelity)
        # Cache the constant term sum_ij p log p of the KL divergence (p is fixed
        # during optimization) so the inner loop only recomputes the y-dependent part.
        mask = p_probs > 0.0
        self._p_log_p_sum = float(np.sum(p_probs[mask] * np.log(p_probs[mask])))
        cost_f = partial(self.cost_f, p_prob=p_probs, fidelity=fidelity)
        # d=2次元に落とすので2倍
        alpha = np.random.rand(n_data * 2)
        self.plot(self.calc_y(fidelity, alpha.reshape(n_data, 2)), y_label, "before")
        if method == "adam":
            self.optimizer_state = self.optimizer.get_init_state(alpha)
            for n_epoch in range(self.max_iter):
                if n_epoch % 10 == 0:
                    print(f"epoch:{n_epoch} loss:{self.optimizer_state.cost}")

                def grad_f(alpha):
                    return self.calc_grad(alpha, p_probs, fidelity)

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
        elif method == "L-BFGS-B":
            from scipy.optimize import minimize

            # jac=True: one function returns (loss, grad), sharing the d^2/num/Z work.
            loss_grad = partial(self.calc_loss_grad, p_prob=p_probs, fidelity=fidelity)
            result = minimize(
                loss_grad, alpha, method="L-BFGS-B", jac=True, options={"maxiter": self.max_iter}
            )
            print(result)
            self.trained_alpha = result.x
        else:
            raise ValueError(f"unknown optimization method: {method!r}")

        y = self.calc_y(fidelity, self.trained_alpha.reshape(n_data, 2))
        self.plot(y, y_label, "after")

    def transform(self, X_test: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the low-dimensional embedding for test data using the trained alpha.

        Args:
            X_test: Test input array of shape (n_samples, n_features).

        Returns:
            Low-dimensional embedding of shape (n_samples, 2).
        """
        fidelity = self.calc_fidelity_all(X_test, self.X_train, self.pqs_f_helper)
        y = self.calc_y(fidelity, self.trained_alpha.reshape(len(self.trained_alpha) // 2, 2))
        return y

    @_quiet_accelerate_fp
    def calc_y(
        self, fidelity: NDArray[np.float64], alpha: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute the low-dimensional embedding y = fidelity @ alpha.

        Args:
            fidelity: Pairwise fidelity matrix |⟨φi|φj⟩|² of shape (n_data, n_data).
            alpha: Embedding coefficients of shape (n_data, 2).

        Returns:
            Low-dimensional embedding of shape (n_data, 2).
        """
        return fidelity @ alpha

    def input_quantum_state(
        self,
        input: NDArray[np.float64],
        pqc_f: Callable[[], LearningCircuit],
        theta: NDArray[np.float64],
    ) -> GeneralCircuitQuantumState:
        """Compute the quantum state |φ(input, θ)⟩ for the given input and circuit parameters.

        Args:
            input: Input data array.
            pqc_f: Factory function that returns a new LearningCircuit instance.
            theta: Parameter vector for the circuit.

        Returns:
            Bound quantum state corresponding to the input and parameters.
        """
        qc = pqc_f()
        bind_params = qc.generate_bound_params(input, theta)
        circuit_state = quantum_state(n_qubits=qc.n_qubits, circuit=qc.circuit).bind_parameters(
            bind_params
        )
        return circuit_state

    def calc_fidelity(self, data, data_tr, pqs_f_helper: pqc_f_helper):
        """Compute the full symmetric fidelity matrix when data == data_tr.

        Args:
            data: Input array.
            data_tr: Must be identical to data.
            pqs_f_helper: Cached quantum state evaluator.

        Returns:
            Symmetric fidelity matrix of shape (n_data, n_data).

        Raises:
            ValueError: If data and data_tr are not identical.
        """
        if not np.array_equal(data, data_tr):
            raise ValueError("data and data_tr must be the same")
        states = [pqs_f_helper.get(x) for x in data]
        return fidelity_gram(states)

    def calc_fidelity_all(self, data, data_tr, pqs_f_helper: pqc_f_helper):
        """Compute the fidelity matrix when data != data_tr (e.g. train vs test).

        Args:
            data: Query data array of shape (n_data, n_features).
            data_tr: Reference data array of shape (n_data_tr, n_features).
            pqs_f_helper: Cached quantum state evaluator.

        Returns:
            Fidelity matrix of shape (n_data, n_data_tr).
        """
        states = [pqs_f_helper.get(x) for x in data]
        states_tr = [pqs_f_helper.get(x) for x in data_tr]
        return fidelity_cross(states, states_tr)

    def plot(self, y: NDArray[np.float64], y_label: NDArray[np.int64], title: str):
        """Plot the 2-D embedding with class labels.

        Args:
            y: 2-D embedding of shape (n_samples, 2).
            y_label: Class labels of shape (n_samples,).
            title: Plot title.

        Note:
            matplotlib is imported here rather than at module scope. It is a
            convenience method, and scikit-quri does not declare matplotlib as a
            runtime dependency — importing it at module scope made
            ``import scikit_quri.qnn.kernel_tsne`` depend on a plotting stack that
            only happens to be installed because another package pulls it in.
        """
        import matplotlib.pyplot as plt

        for i in np.unique(y_label):
            plt.scatter(y[:, 0][y_label == i], y[:, 1][y_label == i])
        plt.title(title)
        plt.show()
