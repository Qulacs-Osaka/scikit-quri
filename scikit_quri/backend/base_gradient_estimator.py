from abc import ABCMeta, abstractmethod
from typing import Sequence

from scikit_quri.circuit import LearningCircuit
from quri_parts.core.estimator import Estimatable, Estimates
from quri_parts.core.estimator.gradient import _ParametricStateT


class BaseGradientEstimator(metaclass=ABCMeta):
    """Gradient Estimatorを実行する際の基底クラス
    estimate_gradientメソッドに対するinterfaceを定義
    """

    @abstractmethod
    def estimate_gradient(
        self,
        operators: Estimatable,
        state: _ParametricStateT,
        params: Sequence[float],
    ) -> Estimates[complex]:
        """
        operatorsとstatesの組み合わせに対して勾配を計算する
        operatorsまたはstatesのどちらかが1つの場合、もう一方の数に合わせて繰り返す
        もしくは、両方の数が同じ場合、1対1で対応させる
        それ以外の場合、ValueErrorを投げる

        Args:
            operators: 勾配を計算する演算子のリスト
            states: 勾配を計算する状態のリスト
            params: 勾配を計算するパラメータのリスト

        Returns:
            operatorsとstatesの組み合わせに対する勾配のリスト
        """

    @abstractmethod
    def estimate_learning_param_gradient(
        self,
        operators: Estimatable,
        circuit: LearningCircuit,
        params: Sequence[float],
    ) -> Sequence[complex]:
        """
        学習パラメータに対する勾配を計算する
        operatorsまたはstatesのどちらかが1つの場合、もう一方の数に合わせて繰り返す
        もしくは、両方の数が同じ場合、1対1で対応させる
        それ以外の場合、ValueErrorを投げる

        Args:
            operators: 勾配を計算する演算子のリスト
            states: 勾配を計算する状態のリスト
            params: 勾配を計算するパラメータのリスト

        Returns:
            operatorsとstatesの組み合わせに対する学習パラメータの勾配のリスト
        """
