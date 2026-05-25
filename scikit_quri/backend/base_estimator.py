from abc import ABCMeta, abstractmethod
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from quri_parts.circuit import ParametricQuantumCircuitProtocol
from quri_parts.core.estimator import Estimatable, Estimate
from quri_parts.qulacs import QulacsStateT


class BaseEstimator(metaclass=ABCMeta):
    """Estimatorを実行する際の基底クラス
    estimateメソッドに対するinterfaceを定義
    """

    @abstractmethod
    def estimate(
        self, operators: Sequence[Estimatable], states: Sequence[QulacsStateT]
    ) -> Iterable[Estimate[complex]]:
        """
        operatorsとstatesの組み合わせに対して期待値を計算する
        operatorsまたはstatesのどちらかが1つの場合、もう一方の数に合わせて繰り返す
        もしくは、両方の数が同じ場合、1対1で対応させる
        それ以外の場合、ValueErrorを投げる

        Args:
            operators: 期待値を計算する演算子のリスト
            states: 期待値を計算する状態のリスト

        Returns:
            operatorsとstatesの組み合わせに対する期待値のリスト
        """


class BatchedSimEstimator(BaseEstimator, metaclass=ABCMeta):
    """Capability extension for simulation backends that natively batch
    expectation-value evaluation over a single parametric circuit with many
    parameter vectors.

    A ``BatchedSimEstimator`` evaluates M operators against one parametric
    circuit bound with N parameter vectors (one per input sample) in a single
    backend call, producing an (M, N) matrix. This is the natural API for
    state-vector simulators (scaluq, future GPU backends) that amortize
    circuit setup across the batch.

    Hardware backends (e.g. OQTOPUS) cannot exploit this shape and should not
    inherit from this class; ``_qnn_common`` dispatches via
    ``isinstance(estimator, BatchedSimEstimator)``.
    """

    @abstractmethod
    def estimate_batched(
        self,
        operators: Sequence[Estimatable],
        circuit: ParametricQuantumCircuitProtocol,
        params: NDArray[np.float64],
    ) -> list[list[float]]:
        """Compute batched expectation values for a parametric circuit.

        Args:
            operators: List of measurement operators. Length: n_operators.
            circuit: Parametric quantum circuit (e.g. from
                ``LearningCircuit.to_batched``).
            params: Per-sample bound parameter matrix.
                Shape: ``(n_samples, parameter_count)``.

        Returns:
            Nested list of shape ``(n_operators, n_samples)`` containing real
            expectation values.
        """

    @abstractmethod
    def estimate_grad_batched(
        self,
        operators: Sequence[Estimatable],
        circuit: ParametricQuantumCircuitProtocol,
        shifted_params: NDArray[np.float64],
        n_samples: int,
        n_learning_params: int,
        delta: float = 1e-5,
    ) -> NDArray[np.float64]:
        """Compute batched numerical gradients via central differences.

        Args:
            operators: List of measurement operators. Length: n_operators.
            circuit: Parametric quantum circuit (e.g. from
                ``LearningCircuit.to_batched_for_gradient``).
            shifted_params: Shifted-parameter matrix.
                Shape: ``(n_samples * 2 * n_learning_params, parameter_count)``.
                Row layout matches ``LearningCircuit.to_batched_for_gradient``.
            n_samples: Number of input samples.
            n_learning_params: Number of unique learning parameters.
            delta: Finite-difference step size.

        Returns:
            Gradient tensor. Shape:
            ``(n_samples, n_operators, n_learning_params)``.
        """
