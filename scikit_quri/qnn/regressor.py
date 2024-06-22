from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from quri_parts.algo.optimizer import Adam
from quri_parts.circuit.circuit_parametric import UnboundParametricQuantumCircuitProtocol
from quri_parts.core.estimator import (
    ConcurrentParametricQuantumEstimator,
    Estimatable,
    ParametricQuantumEstimator,
)
from quri_parts.core.state import ParametricCircuitQuantumState,CircuitQuantumState
from quri_parts.qulacs import QulacsParametricStateT
from quri_parts.circuit import UnboundParametricQuantumCircuit
from scikit_quri.circuit import LearningCircuit
from typing import List
from numpy.random import default_rng
# ! Will remove
from quri_parts.qulacs.estimator import create_qulacs_vector_estimator

@dataclass
class QNNRegressor:
    n_qubits: int
    ansatz: UnboundParametricQuantumCircuitProtocol | LearningCircuit
    estimator: ParametricQuantumEstimator[QulacsParametricStateT]
    gradient_estimator: ConcurrentParametricQuantumEstimator[QulacsParametricStateT]
    optimizer: Adam
    operator: Estimatable
    # * TODO: scaled_x
    # * batchすればいけそう
    # * learning_parameter_list => update_parameters(theta)
    # * input_parameter_list => 
    # * calc_gradどうやってbind分けるか考えといて
    # ? add_input_RX_gateだと，inputが増えていく
    # ? 一気にbindするしかなさそう
    # ? 理想をいうなら，inputとparameterを分けて，parameterに従属な回路を作ってEstimateしたい
    # ? Unboundである必要性がわからん！

    def __post_init__(self) -> None:
        pass

    def fit(self, x_train: NDArray[np.float_], y_train: NDArray[np.float_],batch_size=32) -> None:
        if x_train.ndim == 1:
            x_train = x_train.reshape((-1, 1))

        self.x_train = x_train
        self.y_train = y_train
        parameter_count = 0
        if isinstance(self.ansatz, LearningCircuit):
            parameter_count = self.ansatz.parameter_count
        else:
            parameter_count = self.ansatz.parameter_count
        print(f"{parameter_count=}")
        np.random.seed(0) 
        init_params = np.random.random(parameter_count)
        optimizer_state = self.optimizer.get_init_state(init_params)
        x_split = list(self._batch(x_train,batch_size))
        y_spilt = list(self._batch(y_train,batch_size))

        while True:
            for x_batched,y_batched in zip(x_split,y_spilt):
                cost_fn_batched = lambda params: self.cost_fn(x_batched,y_batched,params)
                grad_fn_batched = lambda params: self.grad_fn(x_batched,params)
                optimizer_state = self.optimizer.step(
                    optimizer_state,
                    cost_fn_batched, 
                    grad_fn_batched
                )
                print(f"{optimizer_state.cost=}")
                break
                if optimizer_state.status == "CONVERGED":
                    break

                if optimizer_state.status == "FAILED":
                    break
            break

    def run(self, x_train: NDArray[np.float_]) -> NDArray[np.float_]:
        # self.ansatz += 1
        pass

    def cost_fn(self,x_batched:NDArray[np.float_],y_batched:NDArray[np.float_], params: Sequence[float]) -> float:
        y_pred = self._predict_inner(x_batched,params)
        # Case of MSE
        cost = np.mean((y_batched - y_pred) ** 2)
        # cost = self._log_loss(y_batched,y_pred)
        return cost

    def grad_fn(self, x_batched:NDArray[np.float_] ,param_values: Sequence[Sequence[float]]) -> np.ndarray:
        grads = []
        if isinstance(self.ansatz, LearningCircuit):
            n_params = self.ansatz.parameter_count
            n_all_params = self.ansatz.n_parameters
            for x in x_batched:
                params = self.ansatz.generate_bound_params(x,param_values)
                params[0] = 0;params[1] = 0
                if grads == []:
                    print(f"{params=}")
                circuit = ParametricCircuitQuantumState(self.n_qubits, self.ansatz.circuit)
                estimate = self.gradient_estimator(self.operator,circuit,params)
                # input用のparamsを取り除く
                grads.append(estimate.values[n_all_params-n_params:])
        else:
            embed_circuits = self._embed_x_circuit(x_batched)
            for circuit_state in embed_circuits:
                grad_estimate = self.gradient_estimator(
                    self.operator,
                    circuit_state,
                    param_values,
                )
                grads.append(grad_estimate.values)
        grads = np.asarray(grads)
        print(f"{grads[0]=}")
        grads = np.mean(grads, axis=0)
        grads = np.asarray([g.real for g in grads])


        return grads
    
    def _embed_x_circuit(self,x_scaled: NDArray[np.float_]) -> List[ParametricCircuitQuantumState]:
        circuits = []
        for x in x_scaled:
            circuit = UnboundParametricQuantumCircuit(self.n_qubits)
            for i in range(self.n_qubits):
                circuit.add_RY_gate(i, np.arcsin(x) * 2)
                circuit.add_RZ_gate(i, np.arccos(x * x) * 2)
            
            # bind_circuit = self.ansatz.bind_parameters(params)
            circuit = circuit.combine(self.ansatz)
            parametric_circuit = ParametricCircuitQuantumState(self.n_qubits, circuit)
            circuits.append(parametric_circuit)

        return circuits

    def _batch(self,data:NDArray[np.float_],batch_size:int):
        for i in range(0,len(data),batch_size):
            if i+batch_size > len(data):
                yield data[i:]
            else:
                yield data[i:i+batch_size]

    def _log_loss(self,y_true, y_pred, epsilon=1e-15):
        """
        ロジスティック損失を計算する関数

        Parameters:
        y_true (numpy.ndarray): 実際のラベル（0または1）
        y_pred (numpy.ndarray): 予測された確率（0から1の範囲）
        epsilon (float): 数値安定性のための小さな値（デフォルトは1e-15）

        Returns:
        float: ロジスティック損失
        """
        # 予測された確率をクリップして数値安定性を確保
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # ロジスティック損失の計算
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        return loss

    def _predict_inner(self, x_scaled: NDArray[np.float_], params: Sequence[float]) -> NDArray[np.float_]:
        res = []
        from quri_parts.circuit.utils.circuit_drawer import draw_circuit
        if isinstance(self.ansatz, LearningCircuit):
            for x in x_scaled:
                params = self.ansatz.generate_bound_params(x,params)
                parametric_circuit = ParametricCircuitQuantumState(self.n_qubits, self.ansatz.circuit)
                estimate = self.estimator(self.operator,parametric_circuit,params)
                res.append(estimate.value.real)
            

        else:
            for x in x_scaled:
                circuit = UnboundParametricQuantumCircuit(self.n_qubits)
                for i in range(self.n_qubits):
                    circuit.add_RY_gate(i, np.arcsin(x) * 2)
                    circuit.add_RZ_gate(i, np.arccos(x * x) * 2)
                
                # bind_circuit = self.ansatz.bind_parameters(params)
                circuit = circuit.combine(self.ansatz)
                parametric_circuit = ParametricCircuitQuantumState(self.n_qubits, circuit)
                estimate = self.estimator(self.operator,parametric_circuit,params)
                res.append(estimate.value.real)
        
        return res 
            

