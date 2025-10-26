from .base_estimator import BaseEstimator
from typing import Sequence, Iterable, Optional
from quri_parts_oqtopus.backend import OqtopusConfig, OqtopusEstimationBackend
from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.core.operator import Operator, PauliLabel
from quri_parts.core.estimator import Estimatable, Estimate


class OqtopusEstimator(BaseEstimator):
    """quri-parts-oqtopusを用いて実機で期待値を計算するEstimator Class
    実行には`~/.oqtopus`の設定が必要
    https://quri-parts-oqtopus.readthedocs.io/en/stable/usage/getting_started/#prepare-oqtopus-configuration-file

    Args:
        device_id: 実行するデバイスのID
        shots: ショット数. Defaults to 1000.
        config: OqtopusのConfig. Defaults to None.
    """

    def __init__(
        self, device_id: str, shots: int = 1000, config: Optional[OqtopusConfig] = None
    ) -> None:
        self.backend = OqtopusEstimationBackend(config)
        self.device_id = device_id
        self.shots = shots

    def estimate(self, operators, states):
        """operatorsとstatesの組み合わせに対して期待値を計算する
        operatorsまたはstatesのどちらかが1つの場合、もう一方の数に合わせて繰り返す
        もしくは、両方の数が同じ場合、1対1で対応させる
        それ以外の場合、ValueErrorを投げる

        Args:
            operators: 期待値を計算する演算子のリスト
            states: 期待値を計算する状態のリスト

        Returns:
            operatorsとstatesの組み合わせに対する期待値のリスト

        Raises:
            ValueError: operatorsまたはstatesが空、もしくは両方の数が異なる場合
            BackendError: Oqtopusでの実行に失敗した場合
        """
        num_ops = len(operators)
        num_states = len(states)

        # operatorが1つもしくはstateが1つの場合は、もう一方の数に合わせて繰り返す
        if num_ops == 0:
            raise ValueError("No operator specified.")

        if num_states == 0:
            raise ValueError("No state specified.")

        if num_ops > 1 and num_states > 1 and num_ops != num_states:
            raise ValueError(
                f"Number of operators ({num_ops}) does not matchnumber of states ({num_states})."
            )

        if num_states == 1:
            # memory節約のため、shallow copy
            circuits = [states[0].circuit] * num_ops
            return self._estimate_concurrently(operators, circuits)
        else:
            if num_ops == 1:
                operators = [next(iter(operators))] * num_states
            circuits = [state.circuit for state in states]
            return self._estimate_concurrently(operators, circuits)

    def _estimate_concurrently(
        self, operators: Sequence[Estimatable], circuits: Sequence[NonParametricQuantumCircuit]
    ) -> Iterable[Estimate[complex]]:
        """
        operatorsとcircuitsの1対1の組み合わせに対して期待値を計算する
        Args:
            operators: 期待値を計算する演算子のリスト
            circuits: 期待値を計算する量子回路のリスト

        Returns:
            operatorsとcircuitsの組み合わせに対する期待値のリスト
        Raises:
            BackendError: Oqtopusでの実行に失敗した場合
        """
        results = []
        for circuit, operator in zip(circuits, operators):
            # EstimatableをOperatorに統一する
            if isinstance(operator, PauliLabel):
                operator = Operator({operator: 1.0})
            job = self.backend.estimate(
                circuit, operator=operator, device_id=self.device_id, shots=self.shots
            )
            result = job.result()
            exp_real = result.exp_value
            # * success以外の場合、例外を投げるため、exp_valueがNoneになることはない
            if not exp_real:
                exp_real = 0.0
            results.append(complex(exp_real, 0.0))
        return results
