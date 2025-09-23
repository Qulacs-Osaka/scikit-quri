from .base_estimator import BaseEstimator

from quri_parts.qulacs.estimator import (
    create_qulacs_vector_concurrent_estimator
)

class SimEstimator(BaseEstimator):
    """quri-parts-qulacsを用いて期待値を計算するSimulation用Estimator Class
    """
    def estimate(self, operators, states):
        estimator = create_qulacs_vector_concurrent_estimator()
        return estimator(operators, states)