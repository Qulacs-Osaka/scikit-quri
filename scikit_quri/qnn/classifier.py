from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from quri_parts.algo.optimizer import Adam
from quri_parts.core.estimator import (
    ConcurrentQuantumEstimator,
    Estimatable,
    GradientEstimator,
)
from quri_parts.core.estimator.gradient import _ParametricStateT
from quri_parts.algo.optimizer import OptimizerStatus
from quri_parts.qulacs import QulacsStateT
from scikit_quri.circuit import LearningCircuit
from typing import List, Optional
from numpy.random import default_rng
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss
from quri_parts.core.state import quantum_state
# ! Will remove
import time

@dataclass
class QNNClassifier:
    ansatz: LearningCircuit
    num_class: int
    estimator : ConcurrentQuantumEstimator[QulacsStateT]
    gradient_estimator : GradientEstimator[_ParametricStateT]
    optimizer : Adam
    operator : List[Estimatable]

    x_norm_range: float = field(default=1.0)
    y_norm_range: float = field(default=0.7)

    do_x_scale: bool = field(default=True)
    do_y_scale: bool = field(default=True)
    n_outputs: int = field(default=1)
    y_exp_ratio: float = field(default=2.2)

    trained_param: Sequence[float] = field(default=None)

    n_qubit: int = field(init=False)


    
    def __post_init__(self) -> None:
        self.n_qubit = self.ansatz.n_qubits
        if self.do_x_scale:
            self.scale_x_scaler = MinMaxScaler(
                feature_range=(-self.x_norm_range, self.x_norm_range)
            )
    
    def softmax(self,x:NDArray[np.float_],axis=None)->NDArray[np.float_]:
        x_max = np.amax(x,axis=axis,keepdims=True)
        exp_x_shifted = np.exp(x-x_max)
        return exp_x_shifted/np.sum(exp_x_shifted,axis=axis,keepdims=True)
    
    def _cost(self,
        x_train:NDArray[np.float_],
        y_train:NDArray[np.int_],
        params:NDArray[np.float_]):

        if x_train.ndim == 1:
            x_train = x_train.reshape(-1, 1)
        
        if self.do_x_scale:
            x_scaled = self.scale_x_scaler.fit_transform(x_train)
        else:
            x_scaled = x_train
        
        cost = self.cost_func(x_scaled,y_train,params)
        return cost
    def fit(self,
        x_train:NDArray[np.float_],
        y_train:NDArray[np.int_],
        maxiter:Optional[int] = 100):

        if x_train.ndim == 1:
            x_train = x_train.reshape(-1, 1)
        
        if self.do_x_scale:
            x_scaled = self.scale_x_scaler.fit_transform(x_train)
        else:
            x_scaled = x_train
        
        parameter_count = self.ansatz.learning_params_count
        init_params = 2*np.pi * np.random.random(parameter_count)
        print(f"{init_params=}")
        optimizaer_state = self.optimizer.get_init_state(init_params)

        cost_func = lambda params: self.cost_func(x_scaled,y_train,params)
        grad_func = lambda params: self.cost_func_grad(x_scaled,y_train,params)
        print(f"{cost_func(init_params)=}")
        print(f"{self._estimate_grad(x_scaled,init_params)=}")
        c = 0
        while maxiter > c:
            optimizaer_state = self.optimizer.step(
                optimizaer_state,
                cost_func,
                grad_func
            )
            print(f"{optimizaer_state.cost=}")

            if optimizaer_state.status == OptimizerStatus.CONVERGED:
                break
            if optimizaer_state.status == OptimizerStatus.FAILED:
                break

            c += 1
        
        print(f"{optimizaer_state.cost=}")
        self.trained_param = optimizaer_state.params

    def predict(self,x_test:NDArray[np.float_])->NDArray[np.float_]:
        if self.trained_param is None:
            raise ValueError("Model is not trained.")
        
        if x_test.ndim == 1:
            x_test = x_test.reshape(-1,1)
        if self.do_x_scale:
            x_scaled = self.scale_x_scaler.transform(x_test)
        else:
            x_scaled = x_test
        y_pred = self._predict_inner(x_scaled,self.trained_param)#.argmax(axis=1)
        return y_pred

    def _predict_inner(self, x_scaled: NDArray[np.float_],params:NDArray[np.float_]) -> NDArray[np.float_]:
        """
        Predict inner function.

        Parameters:
            x_scaled: Input data whose shape is (batch_size, n_features).
            params: Parameters for the quantum circuit.
        
        Returns:
            res: Predicted outcome.
        """
        res = np.zeros((len(x_scaled), self.num_class))
        circuit_states = []
        # 入力ごとのcircuit_state生成
        for x in x_scaled:
            circuit_params = self.ansatz.generate_bound_params(x,params)
            circuit_state = quantum_state(n_qubits=self.n_qubit,circuit=self.ansatz.circuit).bind_parameters(circuit_params)
            circuit_states.append(circuit_state)

        for i in range(self.num_class):
            op = self.operator[i]
            estimates = self.estimator(op,circuit_states)
            estimates = [e.value.real*self.y_exp_ratio for e in estimates]
            res[:,i] = estimates.copy()

        return res
    
    def cost_func(self,x_scaled:NDArray[np.float_],y_scaled:NDArray[np.int_],params:NDArray[np.float_])->float:
        y_pred = self._predict_inner(x_scaled,params)
        # Case of log_logg
        # softmax
        y_pred_sm = self.softmax(y_pred,axis=1)
        loss = log_loss(y_scaled,y_pred_sm)
        # print(f"{params[:4]=}")
        return loss
    def cost_func_grad(self,x_scaled:NDArray[np.float_],y_train:NDArray[np.float_],params:NDArray[np.float_])->float:
        # start = time.perf_counter()
        y_pred = self._predict_inner(x_scaled,params)
        y_pred_sm = self.softmax(y_pred,axis=1)
        raw_grads = self._estimate_grad(x_scaled,params)
        # print(f"{raw_grads.shape=}")
        grads = np.zeros(self.ansatz.n_learning_params)
        # print(f"{raw_grads=}")
        for sample_index in range(len(x_scaled)):
            for current_class in range(self.num_class):
                expected = 1.0 if current_class == y_train[sample_index] else 0.0
                coef = self.y_exp_ratio*(-expected + y_pred_sm[sample_index][current_class])
                grads += coef*raw_grads[sample_index][current_class]
        grads /= len(x_scaled)
        # print(f"{time.perf_counter()-start=}")

        return grads

    def _estimate_grad(self,x_scaled:NDArray[np.float_],params:NDArray[np.float_])->NDArray[np.float_]:
        grads = []
        learning_param_indexes = self.ansatz.get_learning_param_indexes()
        start = time.perf_counter() 
        for x in x_scaled:
            _grads = []
            for op in self.operator:
                circuit_params = self.ansatz.generate_bound_params(x,params)
                param_state = quantum_state(n_qubits=self.n_qubit,circuit=self.ansatz.circuit)
                estimate = self.gradient_estimator(op,param_state,circuit_params)
                # input用のparamsを取り除く
                grad = np.array(estimate.values)[learning_param_indexes]
                _grads.append([g.real for g in grad])
            grads.append(_grads)
        return np.asarray(grads)