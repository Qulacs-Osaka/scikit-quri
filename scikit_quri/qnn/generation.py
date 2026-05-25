# mypy: ignore-errors
"""Quantum Circuit Born Machine (QCBM) generative model.

Implements the MMD-based training algorithm from
Liu & Wang, "Differentiable Learning of Quantum Circuit Born Machines",
Phys. Rev. A 98, 062324 (2018), arXiv:1804.04168.

The model samples bit strings z from ``p_theta(z) = |⟨z|psi(theta)⟩|^2``
(Born rule); training minimizes the squared maximum mean discrepancy MMD^2
between model samples and target samples in a reproducing kernel Hilbert
space. Both cost and gradient estimators are sample-based, so the same code
runs on a state-vector simulator (``QulacsSampler``), a noisy simulator, or
real hardware (``OqtopusSampler``).
"""

from functools import partial
from typing import Callable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from quri_parts.algo.optimizer import Optimizer, OptimizerStatus

from scikit_quri.backend import BaseSampler
from scikit_quri.circuit import LearningCircuit


def default_gaussian_mixture_kernel(
    sigmas: Sequence[float] = (0.25, 1.0, 4.0),
) -> Callable[[NDArray, NDArray], NDArray]:
    """Gaussian-mixture kernel on integer bit-string distances.

    ``K(x_i, y_j) = (1/|sigmas|) * sum_sigma exp(-(x_i - y_j)^2 / (2 sigma^2))``.

    Liu & Wang recommend mixtures of bandwidths so the kernel captures both
    local and global differences between distributions. The default values
    are reasonable for low-qubit problems; for larger bit-string ranges
    consider scaling sigmas with the support size.
    """
    inv_2sigma2 = np.asarray([1.0 / (2.0 * s * s) for s in sigmas], dtype=np.float64)

    def kernel(x: NDArray, y: NDArray) -> NDArray:
        diff = x.astype(np.float64)[:, None] - y.astype(np.float64)[None, :]
        diff_sq = diff * diff
        # (n_x, n_y, |sigmas|) -> mean over sigma dim -> (n_x, n_y)
        return np.exp(-diff_sq[:, :, None] * inv_2sigma2[None, None, :]).mean(axis=-1)

    return kernel


def _mmd_squared(
    model_samples: NDArray[np.int_],
    target_samples: NDArray[np.int_],
    kernel: Callable[[NDArray, NDArray], NDArray],
) -> float:
    """Sample-based estimator of MMD^2(p_model, p_target).

    ``MMD^2 = E_{x,x'~p}[K(x,x')] - 2 E_{x~p, y~q}[K(x,y)] + E_{y,y'~q}[K(y,y')]``.

    Uses the biased estimator (allowing i == j) for simplicity; the bias
    vanishes as n_shots grows and does not affect the gradient (the third
    term is independent of theta).
    """
    return float(
        kernel(model_samples, model_samples).mean()
        - 2 * kernel(model_samples, target_samples).mean()
        + kernel(target_samples, target_samples).mean()
    )


class QNNGenerator:
    """Quantum Circuit Born Machine trained with MMD loss.

    Args:
        circuit: Parametric circuit (ansatz). The input portion of the
            circuit is bound to a constant ``np.array([0])`` placeholder —
            this class learns an unconditional distribution, so any
            ``add_input_*`` gates should be avoided.
        solver: Optimizer driving theta updates.
        sampler: Sampling backend implementing :class:`BaseSampler`.
        n_shots: Number of measurement shots per circuit evaluation.
            Used for cost, gradient (per shift), and predict.
        kernel: Kernel ``K(x, y) -> (n_x, n_y)`` for the MMD loss. ``x`` and
            ``y`` are arrays of bit-string integers. Defaults to a Gaussian
            mixture from :func:`default_gaussian_mixture_kernel`.
        fitting_qubit: Number of qubits used to represent the output
            distribution. When less than ``circuit.n_qubits`` the higher
            qubits are marginalized out (``z mod 2^fitting_qubit``).
            Defaults to ``circuit.n_qubits``.

    Notes:
        Parameter-shift gradients are computed at the learning-parameter
        level (length = ``circuit.learning_params_count``). This is exact
        when each learning parameter controls a single Pauli rotation gate;
        circuits using ``share_with`` to share one learning parameter
        across multiple gates will receive an approximate gradient — the
        cost function itself is unaffected.
    """

    def __init__(
        self,
        circuit: LearningCircuit,
        solver: Optimizer,
        sampler: BaseSampler,
        n_shots: int = 1024,
        kernel: Optional[Callable[[NDArray, NDArray], NDArray]] = None,
        fitting_qubit: Optional[int] = None,
    ) -> None:
        self.n_qubit: int = circuit.n_qubits
        self.circuit = circuit
        self.solver = solver
        self.sampler = sampler
        self.n_shots = n_shots
        self.kernel: Callable[[NDArray, NDArray], NDArray] = (
            kernel if kernel is not None else default_gaussian_mixture_kernel()
        )
        self.fitting_qubit: int = fitting_qubit if fitting_qubit is not None else self.n_qubit
        if not 0 < self.fitting_qubit <= self.n_qubit:
            raise ValueError(
                f"fitting_qubit must be in (0, {self.n_qubit}], got {self.fitting_qubit}"
            )
        self.trained_param: Optional[NDArray[np.float64]] = None

    # --- Training ---------------------------------------------------------

    def fit(self, train_data: NDArray[np.int_], maxiter: int = 100) -> None:
        """Train against a sample-list target distribution.

        Args:
            train_data: Array of bit-string integers; the empirical
                distribution of these is the target.
            maxiter: Maximum optimizer iterations.
        """
        train_samples = np.asarray(train_data, dtype=np.int64)
        if self.fitting_qubit < self.n_qubit:
            train_samples = train_samples % (2**self.fitting_qubit)
        self._fit_inner(train_samples, maxiter)

    def fit_direct_distribution(
        self,
        p: NDArray[np.float64],
        maxiter: int = 100,
        n_target_samples: int = 10000,
        seed: int = 0,
    ) -> None:
        """Train against a target probability vector.

        Internally samples ``n_target_samples`` bit strings from ``p`` and
        delegates to :meth:`fit`. The MMD estimator is sample-based.

        Args:
            p: Target probability vector of length ``2^fitting_qubit``.
            maxiter: Maximum optimizer iterations.
            n_target_samples: Number of target-distribution samples.
            seed: Seed for the target sampler.
        """
        if len(p) != 2**self.fitting_qubit:
            raise ValueError(
                f"Probability vector length {len(p)} != 2^fitting_qubit ({2**self.fitting_qubit})"
            )
        rng = np.random.default_rng(seed)
        train_samples = rng.choice(len(p), size=n_target_samples, p=p).astype(np.int64)
        self._fit_inner(train_samples, maxiter)

    def _fit_inner(self, train_samples: NDArray[np.int64], maxiter: int) -> None:
        n_params = self.circuit.learning_params_count
        theta_init = 2 * np.pi * np.random.random(n_params)
        opt_state = self.solver.get_init_state(theta_init)

        cost_fn = partial(self.cost_func, train_samples=train_samples)
        grad_fn = partial(self._cost_func_grad, train_samples=train_samples)

        c = 0
        while c < maxiter:
            opt_state = self.solver.step(opt_state, cost_fn, grad_fn)
            # opt_state.cost can be None when the optimizer reports CONVERGED/FAILED
            cost_str = f"{opt_state.cost:.6f}" if opt_state.cost is not None else "n/a"
            print(f"\riter:{c}/{maxiter} cost:{cost_str}", end="", flush=True)
            if opt_state.status == OptimizerStatus.CONVERGED:
                break
            if opt_state.status == OptimizerStatus.FAILED:
                break
            c += 1
        print()
        self.trained_param = opt_state.params

    # --- Prediction -------------------------------------------------------

    def predict(self, n_shots: Optional[int] = None) -> NDArray[np.float64]:
        """Estimate the model's output probability vector via sampling.

        Args:
            n_shots: Override sampling shots. Defaults to ``self.n_shots``.

        Returns:
            Empirical probability vector of length ``2^fitting_qubit``.
            Has shot noise of order ``1/sqrt(n_shots)``.
        """
        if self.trained_param is None:
            raise ValueError("Call fit() before predict()")
        n = n_shots if n_shots is not None else self.n_shots
        samples = self._sample(self.trained_param, n)
        if self.fitting_qubit < self.n_qubit:
            samples = samples % (2**self.fitting_qubit)
        counts = np.bincount(samples, minlength=2**self.fitting_qubit)
        return counts.astype(np.float64) / len(samples)

    # --- Cost & gradient --------------------------------------------------

    def cost_func(self, theta: NDArray[np.float64], train_samples: NDArray[np.int64]) -> float:
        """Estimate MMD^2(model, target) with ``self.n_shots`` model samples."""
        model_samples = self._sample(theta, self.n_shots)
        if self.fitting_qubit < self.n_qubit:
            model_samples = model_samples % (2**self.fitting_qubit)
        return _mmd_squared(model_samples, train_samples, self.kernel)

    def _cost_func_grad(
        self, theta: NDArray[np.float64], train_samples: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        """Parameter-shift gradient of MMD^2 w.r.t. theta.

        Liu & Wang eq. (12): for each learning parameter theta_l,
        d MMD^2 / d theta_l =
              E_{p_+, p}[K] - E_{p_-, p}[K]
            - E_{p_+, q}[K] + E_{p_-, q}[K]
        where p_+ / p_- are the distributions at theta +/- (pi/2) e_l, p is
        p_theta, q is the target.
        """
        n_params = len(theta)
        grad = np.zeros(n_params, dtype=np.float64)
        shift = np.pi / 2

        model_samples = self._sample(theta, self.n_shots)
        marginalize = self.fitting_qubit < self.n_qubit
        mod = 2**self.fitting_qubit if marginalize else None
        if marginalize:
            model_samples = model_samples % mod

        for j in range(n_params):
            theta_plus = theta.copy()
            theta_plus[j] += shift
            theta_minus = theta.copy()
            theta_minus[j] -= shift

            plus_samples = self._sample(theta_plus, self.n_shots)
            minus_samples = self._sample(theta_minus, self.n_shots)
            if marginalize:
                plus_samples = plus_samples % mod
                minus_samples = minus_samples % mod

            k_pm = self.kernel(plus_samples, model_samples).mean()
            k_mm = self.kernel(minus_samples, model_samples).mean()
            k_pt = self.kernel(plus_samples, train_samples).mean()
            k_mt = self.kernel(minus_samples, train_samples).mean()

            grad[j] = (k_pm - k_mm) - (k_pt - k_mt)
        return grad

    # --- Sampling ---------------------------------------------------------

    def _sample(self, theta: NDArray[np.float64], n_shots: int) -> NDArray[np.int64]:
        """Sample ``n_shots`` bit strings from the model at parameter theta.

        Returns an array of integers (each representing a bit string).
        """
        bound = self.circuit.bind_input_and_parameters(np.array([0]), np.asarray(theta))
        counts = next(iter(self.sampler.sample([(bound, n_shots)])))
        # MeasurementCounts is dict[int, int]; expand to a flat int array
        out = np.empty(int(sum(counts.values())), dtype=np.int64)
        idx = 0
        for bitstr, c in counts.items():
            out[idx : idx + c] = bitstr
            idx += c
        return out[:idx]
