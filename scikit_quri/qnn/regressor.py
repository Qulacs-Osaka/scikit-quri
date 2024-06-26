from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from quri_parts.algo.optimizer import Adam
from quri_parts.circuit.circuit_parametric import UnboundParametricQuantumCircuitProtocol
from quri_parts.core.estimator import (
    ConcurrentParametricQuantumEstimator,
    Estimatable,
    ParametricQuantumEstimator,
)
from quri_parts.algo.optimizer import OptimizerStatus
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

    x_norm_range: float = field(default=1.0)
    y_norm_range: float = field(default=0.7)

    do_x_scale: bool = field(default=True)
    do_y_scale: bool = field(default=True)
    n_outputs: int = field(default=1)
    # * TODO: scaled_x
    #  batchすればいけそう
    #  learning_parameter_list => update_parameters(theta)
    #  input_parameter_list => 
    # * calc_gradどうやってbind分けるか考えといて
    # * xとしてのParamsと，ParametricとしてのParamsが混合してる
    # * i番目のQubitに対して，x[i]をbindする
    # ! add_inputで，indexに対して，同じxをbindする必要がある
    # ! 現状,RYとRZがあるが,両方arccos(x**2)でbindされている
    # ! gradで，スライスが恒等である保障はない
    # ! EstimatorをConcurrentにしたほうが，predict_innnerで多次元outputに対応できる
    # ? add_input_RX_gateだと，inputが増えていく
    # ? 一気にbindするしかなさそう
    # ? 理想をいうなら，inputとparameterを分けて，parameterに従属な回路を作ってEstimateしたい
    # ? Unboundである必要性がわからん！

    def minMaxScaler(self,X:NDArray[np.float_],feature_range:tuple[int,int]=(0, 1)):
        (min,max) = feature_range
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min
        return X_scaled

    def __post_init__(self) -> None:
        pass

    def fit(self, x_train: NDArray[np.float_], y_train: NDArray[np.float_],batch_size=32) -> None:
        if x_train.ndim == 1:
            x_train = x_train.reshape((-1, 1))
        
        if y_train.ndim == 1:
            y_train = y_train.reshape((-1, 1))
        
        if self.do_x_scale:
            x_scaled = self.minMaxScaler(x_train,feature_range=(-self.x_norm_range,self.x_norm_range))
        else:
            x_scaled = x_train
        
        if self.do_y_scale:
            y_scaled = self.minMaxScaler(y_train,feature_range=(-self.y_norm_range,self.y_norm_range))
        else:
            y_scaled = y_train
        
        self.n_outputs = y_scaled.shape[1]


        self.x_train = x_scaled
        self.y_train = y_scaled

        print(f"{self.x_train=}")

        parameter_count = 0
        if isinstance(self.ansatz, LearningCircuit):
            parameter_count = self.ansatz.theta_count
        else:
            parameter_count = self.ansatz.parameter_count
        print(f"{parameter_count=}")
        init_params = np.random.random(parameter_count)
        print(f"{init_params=}")
        optimizer_state = self.optimizer.get_init_state(init_params)
        x_split = list(self._batch(self.x_train,batch_size))
        y_spilt = list(self._batch(self.y_train,batch_size))

        while True:
            for x_batched,y_batched in zip(x_split,y_spilt):
                cost_fn_batched = lambda params: self.cost_fn(x_batched,y_batched,params)
                grad_fn_batched = lambda params: self.grad_fn(x_batched,y_batched,params)
                optimizer_state = self.optimizer.step(
                    optimizer_state,
                    cost_fn_batched, 
                    grad_fn_batched
                )
                print(f"{optimizer_state.cost=}")
                # break
            if optimizer_state.status == OptimizerStatus.CONVERGED:
                break

            if optimizer_state.status == OptimizerStatus.FAILED:
                break

        self.trained_param = optimizer_state.params
        print(f"{self.trained_param=}")
            # break

    def run(self, x_train: NDArray[np.float_]) -> NDArray[np.float_]:
        # self.ansatz += 1
        pass

    def cost_fn(self,x_batched:NDArray[np.float_],y_batched:NDArray[np.float_], params: Sequence[float]) -> float:
        y_pred = self._predict_inner(x_batched,params)
        # Case of MSE
        cost = np.mean((y_batched - y_pred) ** 2)
        # cost = self._log_loss(y_batched,y_pred)
        # print(f"{cost=}")
        return cost
    
    def predict(self,x_test:NDArray[np.float_]) -> NDArray[np.float_]:
        """Predict outcome for each input data in `x_test`.

        Arguments:
            x_test: Input data whose shape is (n_samples, n_features).

        Returns:
            y_pred: Predicted outcome.
        """
        if self.trained_param is None:
            raise ValueError("Model is not trained yet.")
        if x_test.ndim == 1:
            x_test = x_test.reshape((-1, 1))
        
        if self.do_x_scale:
            x_scaled = self.minMaxScaler(x_test,feature_range=(-self.x_norm_range,self.x_norm_range))
        else:
            x_scaled = x_test
        
        if self.do_y_scale:
            y_pred = self.minMaxScaler(
                self._predict_inner(x_scaled,self.trained_param),
                feature_range=(-self.y_norm_range,self.y_norm_range)
            )
        else:
            y_pred = self._predict_inner(x_scaled,self.trained_param)
        
        return y_pred

    def grad_fn(self, x_batched:NDArray[np.float_], y_batched:NDArray[np.float_] ,param_values: Sequence[Sequence[float]]) -> np.ndarray:
        if isinstance(self.ansatz, LearningCircuit):
            # ? shape=(batch_size,)
            y_pred = self._predict_inner(x_batched,param_values)
            circuit_grads = self._estimate_grad(x_batched,param_values)
            grads = 2*(y_pred-y_batched) * circuit_grads
            grads = np.mean(grads, axis=0)
        else:
            embed_circuits = self._embed_x_circuit(x_batched)
            for circuit_state in embed_circuits:
                grad_estimate = self.gradient_estimator(
                    self.operator,
                    circuit_state,
                    param_values,
                )
                grads.append(grad_estimate.values)
        grads = np.asarray([g.real for g in grads])
        # print(f"{grads=}")

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

    def _estimate_grad(self, x_scaled: NDArray[np.float_], params: Sequence[float]) -> NDArray[np.float_]:
        grads = []
        n_all_params = self.ansatz.n_parameters
        n_params = self.ansatz.parameter_count
        for x in x_scaled:
            circuit_params = self.ansatz.generate_bound_params(x,params)
            circuit = ParametricCircuitQuantumState(self.n_qubits, self.ansatz.circuit)
            estimate = self.gradient_estimator(self.operator,circuit,circuit_params)
            # input用のparamsを取り除く
            grad = estimate.values[n_all_params-n_params:]
            grads.append(grad)
        return np.asarray(grads)

    def _predict_inner(self, x_scaled: NDArray[np.float_], params: Sequence[float]) -> NDArray[np.float_]:
        res = []
        from quri_parts.circuit.utils.circuit_drawer import draw_circuit
        if isinstance(self.ansatz, LearningCircuit):
            for x in x_scaled:
                circuit_params = self.ansatz.generate_bound_params(x,params)
                c = self.ansatz.bind_input_and_parameters(x,params)
                parametric_circuit = ParametricCircuitQuantumState(self.n_qubits, self.ansatz.circuit)
                # if res == []:
                #     for gate in c.gates:
                #         print(f"name:{gate.name} params:{gate.params}")
                estimate = self.estimator(self.operator,parametric_circuit,circuit_params)
                # ? これはout_dim == 1のみの応急処理
                res.append([estimate.value.real])
            # print(f"{res=}")


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
        
        # print(f"{res=}")
        return np.asarray(res)
            

