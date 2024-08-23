from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from quri_parts.algo.optimizer import Adam
from quri_parts.circuit.circuit_parametric import UnboundParametricQuantumCircuitProtocol
from quri_parts.core.estimator import (
    ConcurrentParametricQuantumEstimator,
    ConcurrentQuantumEstimator,
    Estimatable,
    ParametricQuantumEstimator,
)
from quri_parts.algo.optimizer import OptimizerStatus
from quri_parts.core.state import ParametricCircuitQuantumState,CircuitQuantumState
from quri_parts.qulacs import QulacsParametricStateT,QulacsStateT
from quri_parts.circuit import UnboundParametricQuantumCircuit
from scikit_quri.circuit import LearningCircuit
from typing import List, Optional
from numpy.random import default_rng
from quri_parts.core.operator import Operator, pauli_label
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss
from quri_parts.core.state import quantum_state
from scipy.optimize import minimize
# ! Will remove
from quri_parts.qulacs.estimator import create_qulacs_vector_estimator

@dataclass
class QNNClassifier:
    ansatz: LearningCircuit
    num_class: int
    estimator : ConcurrentQuantumEstimator[QulacsStateT]
    gradient_estimator : ConcurrentParametricQuantumEstimator[QulacsParametricStateT]
    optimizer : Adam
    operator : Estimatable

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
    
    def fit(self,
        x_train:NDArray[np.float_],
        y_train:NDArray[np.int_],
        maxiter:Optional[int] = 500):

        if x_train.ndim == 1:
            x_train = x_train.reshape(-1, 1)
        
        if self.do_x_scale:
            x_scaled = self.scale_x_scaler.fit_transform(x_train)
        else:
            x_scaled = x_train
        
        parameter_count = self.ansatz.theta_count
        init_params = 2*np.pi * np.random.random(parameter_count)
        # init_params = np.zeros(parameter_count)
        print(f"{init_params=}")
        cost = self.cost_func(x_scaled,y_train,init_params)
        print(f"{cost=}")
        optimizaer_state = self.optimizer.get_init_state(init_params)


        cost_func = lambda params: self.cost_func(x_scaled,y_train,params)
        result = minimize(cost_func,init_params,method="COBYLA",options={"maxiter":maxiter})
        print(result)

        # while True:
        #     cost_func = lambda params: self.cost_func(x_scaled,y_train,params)
        #     grad_func = lambda params: self.cost_func_grad(x_scaled,y_train,params)
        #     optimizaer_state = self.optimizer.step(
        #         optimizaer_state,
        #         cost_func,
        #         grad_func
        #     )
        #     print(f"{optimizaer_state.cost=}")

        #     if optimizaer_state.status == OptimizerStatus.CONVERGED:
        #         break
        #     if optimizaer_state.status == OptimizerStatus.FAILED:
        #         break
    

    def predict(self,x_scaled:NDArray[np.float_])->NDArray[np.float_]:
        y_pred = self._predict_inner(x_scaled,params)
        pass

    def _predict_inner(self, x_scaled: NDArray[np.float_],params:NDArray[np.float_]) -> NDArray[np.float_]:
        """
        Predict inner function.

        Parameters:
            x_scaled: Input data whose shape is (batch_size, n_features).
            params: Parameters for the quantum circuit.
        
        Returns:
            res: Predicted outcome.
        """
        res = []
        for x in x_scaled:
            circuit_params = self.ansatz.generate_bound_params(x,params)
            # print(f"{len(circuit_params)=}")
            parametric_circuit = ParametricCircuitQuantumState(self.n_qubit, self.ansatz.circuit)
            general_state = parametric_circuit.bind_parameters(circuit_params)
            # if res == []:
            #     from quri_parts.circuit.utils.circuit_drawer import draw_circuit
            #     for i,gate in enumerate(general_state.circuit.gates):
            #         if gate.params is not None:
            #             print(f"{i}: {gate.name}: {gate.params}")
            #     # print(f"{x=}")
            #     # print(f"{circuit_params=}")
            #     draw_circuit(general_state.circuit)
            estimate = self.estimator(self.operator,[general_state])
            res.append([e.value.real*self.y_exp_ratio for e in estimate])
        return np.asarray(res)
    
    def cost_func(self,x_scaled:NDArray[np.float_],y_scaled:NDArray[np.int_],params:NDArray[np.float_])->float:
        y_pred = self._predict_inner(x_scaled,params)
        # Case of log_logg
        # softmax
        y_pred_sm = self.softmax(y_pred,axis=1)
        loss = log_loss(y_scaled,y_pred_sm)
        print(f"{params[:4]=}")
        return loss
    
    def cost_func_grad(self,x_scaled:NDArray[np.float_],y_scaled:NDArray[np.float_],params:NDArray[np.float_])->float:
        y_pred = self._predict_inner(x_scaled,params)
        y_pred_sm = self.softmax(y_pred,axis=1)
        raw_grads = self._estimate_grad(x_scaled,params)
        grads = np.zeros(self.ansatz.n_thetas)
        for sample_index in range(len(x_scaled)):
            for current_class in range(self.num_class):
                expected = 1.0 if current_class == y_scaled[sample_index] else 0.0
                coef = self.y_exp_ratio*(-expected + y_pred_sm[sample_index][current_class])
                grads += coef*raw_grads[sample_index][current_class]
        grads /= len(x_scaled)
        # for sample_index in range(len(x_scaled)):
        #     correct_index = y_scaled[sample_index]
        #     ops = []
        #     for current_class in range(self.num_class):
        #         expected = 1.0 if current_class == correct_index else 0.0
        #         coef = self.y_exp_ratio*(-expected + y_pred_sm[sample_index][current_class])
        #         op = Operator({pauli_label(f"Z {current_class}"):coef})
        #         ops.append(op)
        #     grad = self._estimate_grad(x_scaled[sample_index],ops,params)
        #     print(f"{grad=}")

        return grads

    def _estimate_grad(self,x_scaled:NDArray[np.float_],params:NDArray[np.float_])->NDArray[np.float_]:
        grads = []
        n_all_params = self.ansatz.n_parameters
        n_params = self.ansatz.parameter_count
        for x in x_scaled:
            circuit_params = self.ansatz.generate_bound_params(x,params)
            circuit = ParametricCircuitQuantumState(self.n_qubit,self.ansatz.circuit)
            _grads = []
            for current_class in range(self.num_class):
                op = self.operator[current_class]
                estimate = self.gradient_estimator(op,circuit,circuit_params)
                grad = estimate.values[n_all_params-n_params:]
                _grads.append([g.real for g in grad])
            # input用のparamsを取り除く
            grads.append(_grads)
        
        return np.asarray(grads)