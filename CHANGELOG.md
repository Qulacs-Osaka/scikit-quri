# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Versions before 2.0.0 were never tagged or published; `1.1.0` sat in
`pyproject.toml` from 2025-09-24 (`cd0b996`) through a year of breaking refactors, so
that number does not identify a release. 2.0.0 is the first version intended to be
released.

## [2.0.0] - 2026-09-02

### Wrong results (all of these returned plausible numbers with no error)

- **QCNN circuits were partly inert.** `add_parametric_multi_Pauli_rotation_gate` added
  its gate directly to the underlying circuit without registering it, so its slot was
  never filled by the learning template and stayed at angle 0 regardless of `theta`.
  `create_qcnn_ansatz` uses it for every XX/YY/ZZ rotation in the convolution blocks —
  9 of 76 slots in a 4-qubit QCNN were the identity. Measured on the 6-qubit task, the
  model could not learn at all: f1 0.1667 at 20, 60 and 150 iterations, exactly the
  majority-class baseline. After the fix it reaches 0.5688; on the 8-qubit task in
  `test_qcnn`, 0.9376.
- **`share_with` gradients were dropped.** `hadamard_gradient`,
  `SimGradientEstimator` and `OqtopusGradientEstimator` returned the raw per-gate-position
  array: coefficients were not applied, shared positions were not summed, and the
  result did not even have length `learning_params_count`. All gradient paths now
  aggregate through `ParameterRegistry`.
- **The chain rule for `add_parametric_input_*` gates was missing.** With
  `angle = f(theta, x)`, the reported gradient was `d<O>/d(angle)` — off by `df/dtheta`,
  which varies per sample, so the descent direction itself was wrong (measured: exactly
  a factor of `x` for `f = theta * x`). Three gradient paths disagreed; backprop
  returned 0. None of the bundled ansätze use these gates, so only user-written
  data-reuploading circuits were affected.
- **The Hadamard test mis-inverted gates.** `U†` was built by negating `params`, which
  leaves S, T, SqrtX and UnitaryMatrix unchanged, so the "inverse" segment was the
  forward one. Measured error up to 1.79 with a sign flip for SqrtX. Now uses
  quri-parts' `inverse_gate`, and raises instead of guessing on unknown gates.
- **`QNNRegressor.grad_fn` was the cost gradient divided by `y_exp_ratio`** (2.2 by
  default). `_predict_inner` applies the factor and `_estimate_grad` does not;
  `QNNClassifier` already compensated. L-BFGS hid it because the direction is unchanged.
- **The prediction cache could return the previous batch.** It keyed on
  `(id(x_scaled), shape, dtype)` without holding a reference to the array. `x_scaled`
  is a temporary built by the scaler inside `predict`, so it was freed on return and
  the next batch could reuse the address; shape and dtype match in the ordinary case.
  The cache now holds the array and compares identity, then content.
- **Hardware gradients were noise.** `OqtopusGradientEstimator` took a central
  difference with a hardcoded `delta=1e-5` on shot-based expectation values. Against a
  true gradient of 0.456, 1000-shot estimates had a spread of 3175 — 191,919x the
  spread of the parameter-shift rule, which is exact for these gates and costs the same
  two executions per gate position.
- **`register_input_param` did not invalidate the template cache**, so binding once and
  then adding an input gate raised `IndexError`.
- **`create_ibm_embedding_circuit` was not the ZZ feature map it claims to be.** The
  second-order angle was written `pi - x_i * (pi - x_j)` instead of
  `(pi - x_i) * (pi - x_j)` — a parenthesis moved while porting from scikit-qulacs,
  whose copy has the published form. The angle multiplies `Z_i Z_j` and so has to be
  symmetric in `i` and `j`; the shipped one was not (for `x_i=-0.5, x_j=0.8` it gave
  4.31 one way and 0.23 the other). On 4 qubits the encoded state had a mean fidelity
  of 0.086 against the intended one. Both forms are valid kernels and the accidental one
  happens to score better on the bundled toy tasks (`test_noisy_sine` MSE, median of 25
  runs: 0.0035 before, 0.0075 after; iris f1 0.9482 → 0.9468 — both still far from the
  0.0306 and 0.3769 that the trivial baselines score). What was wrong is not the accuracy
  but that the function did not implement the paper it cites.
  The docstring also pointed at arXiv:1802.06002 (Farhi & Neven), which does not define
  this feature map; it is Havlíček et al., [arXiv:1804.11326](https://arxiv.org/abs/1804.11326)
  (Nature 567, 209). The term is now `scikit_quri.circuit.pre_defined.zz_data_map`, with
  a test pinning both the value and the symmetry.

### Added

- `use_adjoint_gradient` (default `True`) on `QNNClassifier` / `QNNRegressor`. On exact
  statevector backends the gradient is computed by the adjoint method — one forward
  pass plus one backpropagation per observable instead of `2 * parameter_count`
  simulations. Exact rather than `O(delta^2)`, and 10–140x faster.
- `ExactStatevectorEstimator` capability marker. The adjoint method cannot run on
  hardware, so device backends fall through to the supplied `gradient_estimator`; a
  hardware run is never silently replaced by a local simulation.
- `parameter_shift_gradient` in `scikit_quri.circuit.gradient`, backend-agnostic and
  the correct rule for hardware.
- `seed` on `QNNClassifier` / `QNNRegressor`. Initial parameters were drawn from the
  unseeded global RNG, so no fit was reproducible.
- `job_name`, `poll_interval`, `timeout` and `max_submit_workers` on `OqtopusEstimator`;
  `job_name` on `OqtopusSampler`.
- `LICENSE` (MIT) and `NOTICE`. The project declared MIT but shipped no license text.
- `CHANGELOG.md`, and `__version__` in `scikit_quri/__init__.py` (previously empty).

### Changed — breaking

| Before | After |
| --- | --- |
| `SimEstimator(use_scaluq=True/False)` | `ScaluqEstimator()` / `QulacsEstimator()`. `SimEstimator` still works and warns. |
| `hadamard_gradient(...)` returned one entry per gate position | one entry per learning parameter (`learning_params_count`) |
| `SimGradientEstimator.estimate_learning_param_gradient(op, circuit, params)` | takes `x=` and `theta=` as well; required when the circuit has parametric-input gates |
| `OqtopusGradientEstimator()` | `OqtopusGradientEstimator(device_id="qulacs", shots=1000)`; `estimate_learning_param_gradient` requires `x=` and `theta=` |
| `add_parametric_multi_Pauli_rotation_gate` returned a `Parameter` | returns the `parameter_id` (`int`), like the other `add_parametric_*` methods, and accepts `share_with` / `share_with_coef` |
| `OqtopusSampler(device_id, config)` | `config` now defaults to `None` |
| `quri-parts-oqtopus==1.0.3` | `>=1.1.5,<2` — 1.0.3's generated client rejects the current API's job response |
| `scaluq>=0.1.0` (no upper bound) | `>=0.1.0,<0.3`; both 0.1 and 0.2 are supported and tested |

Changes from the 2026-05-25 refactor, which predates this changelog but was never
released, are breaking relative to the last published state as well:

| Before | After |
| --- | --- |
| gradient helpers on `LearningCircuit` | `scikit_quri.circuit.gradient` (`backprop_inner_product`, `hadamard_gradient`, and now `adjoint_expectation_gradients`, `parameter_shift_gradient`) |
| parameter bookkeeping inside `LearningCircuit` | `ParameterRegistry`, reachable through `LearningCircuit.input_chain_factors` / `aggregate_gate_gradients` |
| `preprocess_x` | `scikit_quri.circuit.encoding` |
| samplers passed ad hoc | `BaseSampler` (`QulacsSampler`, `OqtopusSampler`), used by QSVM / QKRR / `OverlapEstimator` / `QNNGenerator` |

### Fixed — packaging

- **`pip install scikit-quri` was unusable.** hatchling ships only the package matching
  the project name, so the vendored `quri_parts_scaluq/` was left out of the wheel while
  `backend/__init__` imports it unconditionally: 5 of the 6 subpackages raised
  `ModuleNotFoundError`. Editable installs put the repository root on `sys.path`, which
  hid this for the life of the project. CI now builds the wheel and imports every
  subpackage from a clean environment.
- **scaluq 0.2 support.** A fresh install resolved to 0.2.0, which the vendored adapter
  did not handle: the precision backend module is no longer importable by path (only as
  an attribute), `Circuit()` lost its qubit-count argument, and `Operator`'s integer
  overload changed meaning from qubit count to term count — so the old call still
  succeeded and built a different operator. All three are handled; both versions are
  tested in CI.

### Performance

| | Before | After |
| --- | --- | --- |
| test suite | 103.4s | 34.3s |
| `test_dqn_cl` | 24.8s | 1.6s |
| `test_qcnn` | 168.6s (failing) | 2.9s |
| OQTOPUS parameter-shift gradient, 6 jobs | 120s | 27.6s |

Besides the adjoint gradient, the qulacs circuit and the observables are now converted
once per batch instead of once per sample (`convert_gate` was being called 1,063,040
times in `test_qcnn`), and OQTOPUS jobs are submitted before results are collected,
through a thread pool, with a 1s poll interval instead of the SDK's 10s.

### Tests and CI

- `tests/test_gradient.py` and `tests/test_estimator.py` were skipped wholesale since
  2026-02-17; in `test_gradient.py` the functions doing the work were also prefixed with
  `_`, so they were never collected even without the skip, and `test_grad` called them
  directly — bypassing the `oqtopus` marker and reaching the cloud. Both files are
  rewritten to compare against analytic or finite-difference values instead of
  `is not None` and `len(...) == n`.
- `tests/test_qcnn.py` is no longer skipped.
- Accuracy thresholds are set against measured baselines rather than guessed. For iris,
  the majority class scores f1 0.0907 and logistic regression 0.9740; the threshold
  moved 0.94 → 0.95 and `maxiter` 10 → 30, because at 10 iterations 1 seed in 8 landed
  at 0.9217. For wine, majority 0.3769 and logistic regression 0.9693; 0.8 → 0.90. For
  the noisy-sine regression, 0.03 → 0.012 against a measured 0.0027–0.0073 — at 0.03 the
  2.2x gradient error above passed both before and after being fixed. `test_noisy_sine`
  is sampler-based and `QulacsSampler` cannot be seeded, so its threshold is set against
  the measured shot-noise distribution (p50 0.0075, max 0.0090 over 25 runs) rather than
  a single observation: 0.008 sat at the p90 and flaked on CI.
- OQTOPUS tests are deselected by default (`make test`, `make cov`, `make cov_ci`);
  `make test-oqtopus` runs them. The repository secret is never available to pull
  requests from forks, so those runs could not be made green.
- CI triggers no longer filter on `paths`, which had excluded `pyproject.toml` — the
  file whose changes break packaging.
