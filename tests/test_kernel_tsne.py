"""Tests for the quantum kernel t-SNE implementation (scikit_quri.qnn.kernel_tsne).

These tests pin down the mathematical correctness of the pieces that can be
checked independently of the (plotting-heavy) ``train`` loop:

- ``overlap_estimator``: fidelity |<phi_i|phi_j>|^2 between quantum states.
- ``quantum_kernel_tsne.calc_fidelity`` / ``calc_fidelity_all``: the Gram matrix.
- ``TSNE``: the classical t-SNE probability math (P, Q, perplexity search).
"""

import numpy as np
import pytest
from quri_parts.circuit import H

from scikit_quri.circuit import LearningCircuit
from scikit_quri.qnn.kernel_tsne import TSNE, overlap_estimator, quantum_kernel_tsne


def _circuit_factory(n_qubits: int):
    """Return a factory producing a data-encoding LearningCircuit (no learnable params).

    Each qubit gets an H followed by an input-driven RY rotation, so distinct
    inputs map to distinct (generally non-orthogonal) quantum states.
    """

    def factory() -> LearningCircuit:
        qc = LearningCircuit(n_qubits)
        for i in range(n_qubits):
            qc.add_gate(H(i))
        for i in range(n_qubits):
            qc.add_input_RY_gate(i, lambda x, i=i: x[i % len(x)])
        return qc

    return factory


@pytest.fixture
def fitted_states():
    """Build a quantum_kernel_tsne with cached states for a small random dataset."""
    rng = np.random.default_rng(0)
    n_qubits = 3
    n_data = 6
    X = rng.uniform(-1.0, 1.0, size=(n_data, n_qubits))
    qk = quantum_kernel_tsne(perplexity=3)
    qk.init(_circuit_factory(n_qubits), np.array([]))
    states = qk.generate_X_train_state(X)
    return qk, X, list(states)


def _reference_gram(states):
    """|<phi_i|phi_j>|^2 computed directly from state vectors via one BLAS call."""
    est = overlap_estimator(states)
    est.calc_all_qula_states()
    vectors = np.stack([qs.get_vector() for qs in est.qula_states])
    overlap = vectors.conj() @ vectors.T
    return np.abs(overlap) ** 2


# --- overlap_estimator -----------------------------------------------------


def test_overlap_estimator_identity_and_symmetry(fitted_states):
    _, _, states = fitted_states
    est = overlap_estimator(states)
    est.calc_all_qula_states()
    n = len(states)
    for i in range(n):
        # A normalized state has unit self-overlap.
        assert est.estimate(i, i) == pytest.approx(1.0, abs=1e-9)
        for j in range(n):
            assert est.estimate(i, j) == pytest.approx(est.estimate(j, i), abs=1e-12)
            assert -1e-12 <= est.estimate(i, j) <= 1.0 + 1e-9


def test_overlap_estimator_matches_blas_gram(fitted_states):
    _, _, states = fitted_states
    est = overlap_estimator(states)
    est.calc_all_qula_states()
    ref = _reference_gram(states)
    n = len(states)
    got = np.array([[est.estimate(i, j) for j in range(n)] for i in range(n)])
    assert np.allclose(got, ref, atol=1e-10)


# --- quantum_kernel_tsne.calc_fidelity -------------------------------------


def test_calc_fidelity_symmetric_diagonal_one(fitted_states):
    qk, X, _ = fitted_states
    fidelity = qk.calc_fidelity(X, X, qk.pqs_f_helper)
    n = len(X)
    assert fidelity.shape == (n, n)
    assert np.allclose(np.diag(fidelity), 1.0, atol=1e-9)
    assert np.allclose(fidelity, fidelity.T, atol=1e-12)
    assert np.all(fidelity >= -1e-9)


def test_calc_fidelity_matches_gram(fitted_states):
    """The pair-by-pair loop must agree with the vectorized Gram matrix."""
    qk, X, states = fitted_states
    fidelity = qk.calc_fidelity(X, X, qk.pqs_f_helper)
    ref = _reference_gram(states)
    assert np.allclose(fidelity, ref, atol=1e-10)


def test_calc_fidelity_all_shape_and_consistency(fitted_states):
    """Train-vs-test fidelity equals the corresponding block of the full Gram."""
    qk, X, _ = fitted_states
    X_test = X[:2]
    fidelity = qk.calc_fidelity_all(X_test, X, qk.pqs_f_helper)
    assert fidelity.shape == (2, len(X))
    # Cross fidelity of X_test (subset of X) against X is the top rows of the Gram.
    ref = _reference_gram(list(qk.generate_X_train_state(np.concatenate([X_test, X]))))
    assert np.allclose(fidelity, ref[: len(X_test), len(X_test) :], atol=1e-10)


# --- classical t-SNE probability math --------------------------------------


def test_joint_probabilities_valid_distribution():
    rng = np.random.default_rng(1)
    sq_distance = np.abs(rng.uniform(0, 5, size=(8, 8)))
    sq_distance = (sq_distance + sq_distance.T) / 2
    np.fill_diagonal(sq_distance, 0.0)
    P = TSNE(perplexity=3).joint_probabilities(sq_distance, 3)
    assert P.shape == (8, 8)
    assert P.sum() == pytest.approx(1.0)
    assert np.allclose(P, P.T)
    assert np.all(P >= 0.0)


def test_probabilities_q_valid_distribution():
    rng = np.random.default_rng(2)
    y = rng.uniform(-3, 3, size=(7, 2))
    Q = TSNE().calc_probabilities_q(y)
    assert Q.shape == (7, 7)
    assert Q.sum() == pytest.approx(1.0)
    assert np.allclose(Q, Q.T)
    assert np.allclose(np.diag(Q), 0.0)
    assert np.all(Q >= 0.0)


def test_probabilities_q_uses_squared_distance():
    """Student's t kernel must use SQUARED Euclidean distance: q_ij ~ (1 + ||y_i - y_j||^2)^-1.

    Regression guard for the bug where the non-squared distance was used.
    """
    rng = np.random.default_rng(3)
    y = rng.uniform(-3, 3, size=(5, 2))
    diff = y[:, None, :] - y[None, :, :]
    d2 = (diff**2).sum(axis=-1)
    num = 1.0 / (1.0 + d2)
    np.fill_diagonal(num, 0.0)
    expected = num / num.sum()

    Q = TSNE().calc_probabilities_q(y)
    assert np.allclose(Q, expected, atol=1e-12)


def test_fused_kl_loss_matches_explicit_kl():
    """The fused KL loss must equal the explicit sum_ij p log(p/q) to machine precision.

    KL(P||Q) = sum p log p + sum p log(1 + d^2) + log Z  (with sum_ij p = 1).
    """
    rng = np.random.default_rng(8)
    n = 50
    X = rng.uniform(-2, 2, size=(n, 6))
    tsne = TSNE(perplexity=12)
    P = tsne.joint_probabilities(tsne.cdist(X, X), 12)
    assert P.sum() == pytest.approx(1.0)

    y = rng.uniform(-3, 3, size=(n, 2))
    Q = tsne.calc_probabilities_q(y)
    mask = (P > 0) & (Q > 0)
    kl_explicit = np.sum(P[mask] * np.log(P[mask] / Q[mask]))

    qk = quantum_kernel_tsne(perplexity=12)
    qk.tsne = tsne
    m = P > 0
    qk._p_log_p_sum = float(np.sum(P[m] * np.log(P[m])))
    kl_fused = qk._kl_loss_from_y(y, P)

    assert kl_fused == pytest.approx(kl_explicit, abs=1e-10)


def test_analytic_gradient_matches_numerical():
    """The analytic dC/dalpha must match a central finite-difference gradient.

    Holds for any symmetric kernel ``fidelity`` and any fixed normalized P, since
    the t-SNE gradient assumes only that Q is the Student-t distribution of y = K @ alpha.
    """
    rng = np.random.default_rng(9)
    n = 10
    # Symmetric kernel matrix and a valid normalized, zero-diagonal P.
    K = rng.uniform(0, 1, size=(n, n))
    K = (K + K.T) / 2
    P = rng.uniform(0, 1, size=(n, n))
    P = (P + P.T) / 2
    np.fill_diagonal(P, 0.0)
    P /= P.sum()

    qk = quantum_kernel_tsne(perplexity=3)
    m = P > 0
    qk._p_log_p_sum = float(np.sum(P[m] * np.log(P[m])))

    alpha = rng.uniform(-1, 1, size=n * 2)
    analytic = qk.calc_grad(alpha, P, K)

    def loss(a):
        y = qk.calc_y(K, a.reshape(n, 2))
        return qk._kl_loss_from_y(y, P)

    dx = 1e-6
    numeric = np.zeros_like(alpha)
    for i in range(len(alpha)):
        a = alpha.copy()
        a[i] += dx
        lp = loss(a)
        a[i] -= 2 * dx
        lm = loss(a)
        numeric[i] = (lp - lm) / (2 * dx)

    assert np.allclose(analytic, numeric, rtol=1e-4, atol=1e-6)


def test_fused_loss_grad_matches_separate():
    """calc_loss_grad must return the same loss and gradient as the separate paths."""
    rng = np.random.default_rng(11)
    n = 10
    K = rng.uniform(0, 1, size=(n, n))
    K = (K + K.T) / 2
    P = rng.uniform(0, 1, size=(n, n))
    P = (P + P.T) / 2
    np.fill_diagonal(P, 0.0)
    P /= P.sum()

    qk = quantum_kernel_tsne(perplexity=3)
    m = P > 0
    qk._p_log_p_sum = float(np.sum(P[m] * np.log(P[m])))
    alpha = rng.uniform(-1, 1, size=n * 2)

    loss, grad = qk.calc_loss_grad(alpha, P, K)
    y = qk.calc_y(K, alpha.reshape(n, 2))
    assert loss == pytest.approx(qk._kl_loss_from_y(y, P), abs=1e-12)
    assert np.allclose(grad, qk.calc_grad(alpha, P, K), atol=1e-12)


def test_lbfgs_training_reduces_loss_and_separates_classes():
    """Gradient-based L-BFGS-B training should converge and cluster a toy 2-class set."""
    rng = np.random.default_rng(10)
    n_qubits = 4
    # Two well-separated clusters in input space.
    X = np.vstack(
        [rng.normal(0.3, 0.05, size=(12, n_qubits)), rng.normal(1.2, 0.05, size=(12, n_qubits))]
    )
    y_label = np.array([0] * 12 + [1] * 12)

    qk = quantum_kernel_tsne(perplexity=5, max_iter=500)
    qk.plot = lambda *a, **k: None  # silence plotting
    qk.init(_circuit_factory(n_qubits), np.array([]))
    qk.train(X, y_label, method="L-BFGS-B")

    emb = qk.transform(X)
    assert np.all(np.isfinite(emb))
    # The two clusters should be farther apart than the within-cluster spread.
    c0, c1 = emb[y_label == 0], emb[y_label == 1]
    between = np.linalg.norm(c0.mean(0) - c1.mean(0))
    within = (c0.std(0).mean() + c1.std(0).mean()) / 2
    assert between > within


def test_binary_search_perplexity_achieves_target():
    """The per-point Gaussian bandwidth should yield the requested perplexity."""
    rng = np.random.default_rng(4)
    X = rng.uniform(-2, 2, size=(40, 5))
    diff = X[:, None, :] - X[None, :, :]
    sq_distance = (diff**2).sum(axis=-1)
    target = 10
    cond_P = TSNE(perplexity=target).binary_search_perplexity(sq_distance, target)
    # Perplexity = exp(H) with H the Shannon entropy (nats) of each row.
    p = np.clip(cond_P, 1e-12, None)
    entropy = -(cond_P * np.log(p)).sum(axis=1)
    perplexity = np.exp(entropy)
    assert np.allclose(perplexity, target, rtol=0.05)
