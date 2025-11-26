from abc import ABCMeta, abstractmethod
from typing import Sequence, Iterable

from quri_parts.qulacs import QulacsStateT
from quri_parts.core.estimator import (
    Estimatable,
)
from quri_parts.core.estimator import Estimate


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
        pass
