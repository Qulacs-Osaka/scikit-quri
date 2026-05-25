"""Verify benchmark results: cold vs warm, random order, numerical accuracy."""

import time
import numpy as np
import subprocess
import sys
import os

# --- Settings ---
n_qubits = 8
num_class = 3
y_exp_ratio = 2.2
n_runs_cold = 5
n_runs_warm = 5
depth = 3
batch_list = [16, 64, 128, 256, 512]

# Import after settings
from sklearn.preprocessing import MinMaxScaler
from quri_parts.core.operator import Operator, pauli_label
from scikit_quri.circuit import create_qcl_ansatz
from scikit_quri.backend import SimEstimator
from scikit_quri.qnn._qnn_common import predict_inner

operators = [Operator({pauli_label(f"Z {i}"): 1.0}) for i in range(num_class)]


def make_dummy_data(n_samples: int, n_features: int = 2) -> np.ndarray:
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    data = np.random.random((n_samples, n_features))
    return scaler.fit_transform(data)


def bench_single(circuit, estimator, x_scaled, params) -> float:
    """Single cold-run timing (no warmup)."""
    start = time.perf_counter()
    predict_inner(circuit, estimator, operators, x_scaled, params, y_exp_ratio)
    return time.perf_counter() - start


def bench_warm(
    circuit, estimator, x_scaled, params, n_runs: int = 5
) -> tuple[float, float, list[float]]:
    """Warmup + multiple runs (original method)."""
    # warmup
    predict_inner(circuit, estimator, operators, x_scaled, params, y_exp_ratio)
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        predict_inner(circuit, estimator, operators, x_scaled, params, y_exp_ratio)
        times.append(time.perf_counter() - start)
    return float(np.mean(times)), float(np.std(times)), times


# ============================================================
# 1. Warm vs Cold comparison (fixed batch=512, same circuit)
# ============================================================
print("=" * 70)
print("TEST 1: Cold vs Warm start comparison (batch=512)")
print("=" * 70)

np.random.seed(42)
circuit = create_qcl_ansatz(n_qubits, depth, 1.0)
params = 2 * np.pi * np.random.random(circuit.learning_params_count)
x_scaled = make_dummy_data(512)

estimator_q = SimEstimator(use_scaluq=False)
estimator_s = SimEstimator(use_scaluq=True)

# --- Cold runs (fresh circuit each time) ---
cold_times_q = []
cold_times_s = []
for i in range(n_runs_cold):
    c = create_qcl_ansatz(n_qubits, depth, 1.0)
    p = 2 * np.pi * np.random.random(c.learning_params_count)
    eq = SimEstimator(use_scaluq=False)
    es = SimEstimator(use_scaluq=True)
    cold_times_q.append(bench_single(c, eq, x_scaled, p))
    cold_times_s.append(bench_single(c, es, x_scaled, p))
    print(f"  Cold run {i + 1}: qulacs={cold_times_q[-1]:.4f}s, scaluq={cold_times_s[-1]:.4f}s")

cold_q = np.mean(cold_times_q)
cold_s = np.mean(cold_times_s)
print(f"  Cold avg: qulacs={cold_q:.4f}s, scaluq={cold_s:.4f}s, speedup={cold_q / cold_s:.2f}x")

# --- Warm runs (same circuit, warmup + 5 runs) ---
mean_q, std_q, times_q = bench_warm(circuit, estimator_q, x_scaled, params, n_runs_warm)
mean_s, std_s, times_s = bench_warm(circuit, estimator_s, x_scaled, params, n_runs_warm)
print(
    f"  Warm avg: qulacs={mean_q:.4f}s ±{std_q:.4f}, scaluq={mean_s:.4f}s ±{std_s:.4f}, speedup={mean_q / mean_s:.2f}x"
)
print(f"  Warm raw times (qulacs): {[f'{t:.4f}' for t in times_q]}")
print(f"  Warm raw times (scaluq): {[f'{t:.4f}' for t in times_s]}")

# Check if warm has significant speedup over cold
print(f"  qulacs cold/warm ratio: {cold_q / mean_q:.2f}x")
print(f"  scaluq cold/warm ratio: {cold_s / mean_s:.2f}x")
print()

# ============================================================
# 2. Numerical correctness check
# ============================================================
print("=" * 70)
print("TEST 2: Numerical correctness")
print("=" * 70)

np.random.seed(42)
circuit = create_qcl_ansatz(n_qubits, depth, 1.0)
params = 2 * np.pi * np.random.random(circuit.learning_params_count)

for bs in [16, 128, 512]:
    x_scaled = make_dummy_data(bs)
    eq = SimEstimator(use_scaluq=False)
    es = SimEstimator(use_scaluq=True)

    res_q = predict_inner(circuit, eq, operators, x_scaled, params, y_exp_ratio)
    res_s = predict_inner(circuit, es, operators, x_scaled, params, y_exp_ratio)

    max_diff = np.max(np.abs(res_q - res_s))
    mean_diff = np.mean(np.abs(res_q - res_s))
    print(f"  batch={bs:4d}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")


# ============================================================
# 3. Batch sweep with random order + cold start
# ============================================================
print()
print("=" * 70)
print("TEST 3: Batch size sweep (randomized order, cold start, fresh circuit each time)")
print("=" * 70)

# Shuffle batch order
batch_order = batch_list.copy()
np.random.shuffle(batch_order)
print(f"  Batch order: {batch_order}")

results = {}
for bs in batch_order:
    np.random.seed(42)
    circuit = create_qcl_ansatz(n_qubits, depth, 1.0)
    params = 2 * np.pi * np.random.random(circuit.learning_params_count)

    # Fresh circuit + data for each batch
    x_scaled = make_dummy_data(bs)

    # Fresh estimators
    eq = SimEstimator(use_scaluq=False)
    es = SimEstimator(use_scaluq=True)

    # Single cold run
    tq = bench_single(circuit, eq, x_scaled, params)
    ts = bench_single(circuit, es, x_scaled, params)

    speedup = tq / ts if ts > 0 else 0
    results[bs] = {"qulacs": tq, "scaluq": ts, "speedup": speedup}
    print(f"  batch={bs:4d}: qulacs={tq:.4f}s, scaluq={ts:.4f}s, speedup={speedup:.2f}x")

print()

# ============================================================
# 4. Original-style warm measurement (for comparison to README)
# ============================================================
print("=" * 70)
print("TEST 4: Original-style warm measurement (sequential, small→large)")
print("=" * 70)

np.random.seed(42)
circuit = create_qcl_ansatz(n_qubits, depth, 1.0)
params = 2 * np.pi * np.random.random(circuit.learning_params_count)
estimator_q = SimEstimator(use_scaluq=False)
estimator_s = SimEstimator(use_scaluq=True)

for bs in batch_list:
    x_scaled = make_dummy_data(bs)

    tq_mean, tq_std, tq_raw = bench_warm(circuit, estimator_q, x_scaled, params, n_runs_warm)
    ts_mean, ts_std, ts_raw = bench_warm(circuit, estimator_s, x_scaled, params, n_runs_warm)

    speedup = tq_mean / ts_mean if ts_mean > 0 else 0
    print(
        f"  batch={bs:4d}: qulacs={tq_mean:.4f}s ±{tq_std:.4f}, scaluq={ts_mean:.4f}s ±{ts_std:.4f}, speedup={speedup:.2f}x"
    )
    print(f"           qulacs raw: {[f'{t:.4f}' for t in tq_raw]}")
    print(f"           scaluq raw: {[f'{t:.4f}' for t in ts_raw]}")

# ============================================================
# 5. Subprocess isolation (full cold-start per batch)
# ============================================================
print()
print("=" * 70)
print("TEST 5: Subprocess isolation (truly fresh start per batch)")
print("=" * 70)


def run_in_subprocess(bs: int, use_scaluq: bool) -> float:
    """Run a single predict_inner in a subprocess and return the time."""
    code = f"""
import time, numpy as np
from sklearn.preprocessing import MinMaxScaler
from quri_parts.core.operator import Operator, pauli_label
from scikit_quri.circuit import create_qcl_ansatz
from scikit_quri.backend import SimEstimator
from scikit_quri.qnn._qnn_common import predict_inner

np.random.seed(42)
n_qubits = {n_qubits}
depth = {depth}
num_class = {num_class}
y_exp_ratio = {y_exp_ratio}
operators = [Operator({{pauli_label(f"Z {{i}}"): 1.0}}) for i in range(num_class)]

circuit = create_qcl_ansatz(n_qubits, depth, 1.0)
params = 2 * np.pi * np.random.random(circuit.learning_params_count)

scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
data = np.random.random(({bs}, 2))
x_scaled = scaler.fit_transform(data)

estimator = SimEstimator(use_scaluq={use_scaluq})

start = time.perf_counter()
predict_inner(circuit, estimator, operators, x_scaled, params, y_exp_ratio)
elapsed = time.perf_counter() - start
print(elapsed)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    try:
        return float(result.stdout.strip().split("\n")[-1])
    except (ValueError, IndexError):
        print(f"  STDOUT: {result.stdout}")
        print(f"  STDERR: {result.stderr}")
        return -1.0


for bs in [16, 128, 512]:
    tq = run_in_subprocess(bs, use_scaluq=False)
    ts = run_in_subprocess(bs, use_scaluq=True)
    speedup = tq / ts if ts > 0 else 0
    print(f"  batch={bs:4d}: qulacs={tq:.4f}s, scaluq={ts:.4f}s, speedup={speedup:.2f}x")

print()
print("Done! ✅")
