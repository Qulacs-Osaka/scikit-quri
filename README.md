# scikit-quri
scikit-quri is a library for quantum neural network. This library is based on [quri-parts](https://quri-parts.qunasys.com/) and named after scikit-learn.

# Requirements
- pip
- Python >= 3.10

# Installation
### Build from Source
Install uv before building package.
```
git clone https://github.com/Qulacs-Osaka/scikit-quri.git
cd scikit-quri
uv sync
```
# Documentation
API Documentation: [scikit-quri documentation](https://qulacs-osaka.github.io/scikit-quri/)  
[Sample notebooks](https://github.com/Qulacs-Osaka/scikit-quri/tree/main/samples)

# Sample Code
```python
from quri_parts.core.estimator.gradient import (
    create_numerical_gradient_estimator,
)
from quri_parts.qulacs.estimator import (
    create_qulacs_vector_concurrent_estimator,
    create_qulacs_vector_concurrent_parametric_estimator,
)
from quri_parts.algo.optimizer import Adam

from scikit_quri.circuit import create_qcl_ansatz
from scikit_quri.qnn.classifier import QNNClassifier

circuit = create_qcl_ansatz(n_qubits, 3, 1.0)
num_class=3
estimator = create_qulacs_vector_concurrent_estimator()
gradient_estimator = create_numerical_gradient_estimator(
    create_qulacs_vector_concurrent_parametric_estimator(), delta=1e-10
)
adam = Adam()

qnn = QNNClassifier(circuit, num_class, estimator, gradient_estimator, adam)
qnn.fit(x_train, y_train)
y_pred = qnn.predict(x_test)
```
