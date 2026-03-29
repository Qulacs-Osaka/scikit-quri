# scikit-quri

scikit-quri is a quantum machine learning library built on [quri-parts](https://quri-parts.qunasys.com/), designed with a scikit-learn-like API.
It provides quantum neural network classifiers/regressors, quantum SVMs, and quantum kernel methods that integrate naturally with the scikit-learn ecosystem.

## Features

- **QNN Classifier / Regressor** — Variational quantum circuit-based models with `fit` / `predict` interface
- **Quantum SVM / Kernel Ridge Regression** — Quantum kernel methods compatible with scikit-learn
- **Scaluq backend support** — Batched state vector simulation via [scaluq](https://github.com/qulacs/scaluq) for faster training

## Requirements

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) (package manager)

## Installation

```bash
git clone https://github.com/Qulacs-Osaka/scikit-quri.git
cd scikit-quri
uv sync
```

## Quick Start

```python
from quri_parts.core.estimator.gradient import create_numerical_gradient_estimator
from quri_parts.qulacs.estimator import create_qulacs_vector_concurrent_parametric_estimator
from quri_parts.algo.optimizer import Adam

from scikit_quri.backend import SimEstimator
from scikit_quri.circuit import create_qcl_ansatz
from scikit_quri.qnn.classifier import QNNClassifier

n_qubits = 5
num_class = 3

circuit = create_qcl_ansatz(n_qubits, 3, 1.0)

# use_scaluq=True enables batched estimation via the scaluq backend
estimator = SimEstimator(use_scaluq=True)
gradient_estimator = create_numerical_gradient_estimator(
    create_qulacs_vector_concurrent_parametric_estimator(), delta=1e-10
)
adam = Adam()

qnn = QNNClassifier(circuit, num_class, estimator, gradient_estimator, adam)
qnn.fit(x_train, y_train, maxiter=50)
y_pred = qnn.predict(x_test)
```

### Scaluq backend

Setting `SimEstimator(use_scaluq=True)` uses [scaluq](https://github.com/qulacs/scaluq)'s `StateVectorBatched` for expectation value computation and numerical gradient estimation.
By batching circuit execution across all samples, larger datasets benefit from greater speedups.

| Batch size | predict_inner speedup | Full training speedup |
|------------|----------------------|----------------------|
| 16         | 1.4x                 | 2.2x                 |
| 128        | 11.1x                | 2.7x                 |
| 512        | 19.7x                | 2.5x                 |

## Documentation

- API Documentation: [scikit-quri documentation](https://qulacs-osaka.github.io/scikit-quri/)
- [Sample notebooks](https://github.com/Qulacs-Osaka/scikit-quri/tree/main/samples)

## License

MIT License
