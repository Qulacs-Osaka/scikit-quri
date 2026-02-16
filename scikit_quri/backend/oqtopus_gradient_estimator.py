from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

# from quri_parts.circuit.parameter_shift import ShiftedParameters
from quri_parts.core import quantum_state
from quri_parts.core.estimator import Estimatable, Estimate

from scikit_quri.circuit import LearningCircuit

from .base_gradient_estimator import BaseGradientEstimator
from .oqtopus_estimator import OqtopusEstimator


@dataclass
class LearningCircuitParameter:
    """学習回路のパラメータを保持するデータクラス。

    Attributes:
        input_param: 入力データに対応するパラメータ。
        learning_param: 最適化対象の学習パラメータ。
    """

    input_param: NDArray[np.float64]
    learning_param: NDArray[np.float64]


def _parametric_estimate(
    op_state: tuple[Estimatable, LearningCircuit], params: Sequence[LearningCircuitParameter]
) -> Sequence[Estimate[complex]]:
    """複数のパラメータセットに対して期待値を一括計算する。

    Args:
        op_state: (演算子, 学習回路)のタプル。
        params: パラメータセットのシーケンス。

    Returns:
        各パラメータセットに対する期待値のEstimateオブジェクトのシーケンス。
    """
    operator, circuit = op_state
    estimator = OqtopusEstimator("qulacs")
    n_qubits = circuit.n_qubits
    estimates = []
    states = []
    for param in params:
        bind_circuit = circuit.bind_input_and_parameters(param.input_param, param.learning_param)
        state = quantum_state(n_qubits=n_qubits, circuit=bind_circuit)
        states.append(state)
    estimates = estimator.estimate([operator], states)
    return list(estimates)


def numerical_gradient_estimates(
    op: Estimatable,
    circuit: LearningCircuit,
    params: LearningCircuitParameter,
    delta: float,
) -> Sequence[complex]:
    """数値微分により勾配を計算する。

    中心差分法を用いて各学習パラメータに対する勾配を計算する。
    grad[i] = (f(θ_i + δ/2) - f(θ_i - δ/2)) / δ

    Args:
        op: 期待値を計算する演算子。
        circuit: 学習回路。
        params: 入力パラメータと学習パラメータ。
        delta: 数値微分の刻み幅。

    Returns:
        各学習パラメータに対する勾配のシーケンス。
    """
    v = []
    input_param = params.input_param
    for i in range(len(params.learning_param)):
        current_learning_param = params.learning_param.copy()
        current_learning_param[i] += delta * 0.5
        v.append(LearningCircuitParameter(input_param, current_learning_param))
        current_learning_param = params.learning_param.copy()
        current_learning_param[i] -= delta * 0.5
        v.append(LearningCircuitParameter(input_param, current_learning_param))
    estimates = _parametric_estimate((op, circuit), v)
    grad = []
    for i in range(len(params.learning_param)):
        d = estimates[2 * i].value - estimates[2 * i + 1].value
        grad.append(d / delta)
    return grad


# def parameter_shift_gradient_estimates(
#     op: Estimatable,
#     circuit: LearningCircuit,
#     params: LearningCircuitParameter,
# ) -> Sequence[complex]:
#     """パラメータシフト法により勾配を計算する。

#     Note:
#         現在実装中。

#     Args:
#         op: 期待値を計算する演算子。
#         circuit: 学習回路。
#         params: 入力パラメータと学習パラメータ。

#     Returns:
#         各学習パラメータに対する勾配のシーケンス。
#     """
# gen_params = circuit.generate_bound_params(
#     np.array(params.input_param), np.array(params.learning_param)
# )
# param_mapping = circuit.circuit.param_mapping
# parameter_shift = ShiftedParameters(param_mapping)
# derivatives = parameter_shift.get_derivatives()
# shifted_params_and_coefs = [d.get_shifted_parameters_and_coef(gen_params) for d in derivatives]

# gate_params = set()
# for params_and_coefs in shifted_params_and_coefs:
#     for p, _ in params_and_coefs:
#         gate_params.add(p)
# gate_params_list = list(gate_params)
# pass


class OqtopusGradientEstimator(BaseGradientEstimator):
    """Oqtopusを用いて勾配を計算するGradient Estimator Class。

    OqtopusEstimatorを内部で使用し、数値微分により勾配を計算する。
    """

    def estimate_gradient(self, operators, state, params):
        """全パラメータに対する勾配を計算する。

        Note:
            現在未実装。

        Raises:
            NotImplementedError: このメソッドは未実装。
        """
        raise NotImplementedError("OqtopusGradientEstimator is not implemented yet.")

    def estimate_learning_param_gradient(self, operators, circuit, params) -> Sequence[complex]:
        """学習パラメータに対する勾配を計算する。

        入力パラメータは固定し、学習パラメータのみに対する勾配を計算する。

        Args:
            operators: 期待値を計算する演算子。
            circuit: 学習回路。
            params: 全パラメータ（入力+学習）の値。

        Returns:
            各学習パラメータに対する勾配のシーケンス。
        """
        learning_params = LearningCircuitParameter(
            input_param=np.array([params[i] for i in circuit.get_input_params_indexes()]),
            learning_param=np.array([params[i] for i in circuit.get_learning_params_indexes()]),
        )
        estimate_gradients = numerical_gradient_estimates(
            operators, circuit, learning_params, delta=1e-5
        )
        return estimate_gradients
