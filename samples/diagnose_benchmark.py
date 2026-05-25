"""Diagnose why scaluq batched is faster: thread scaling + cache analysis."""

import time
import os
import subprocess
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from quri_parts.core.operator import Operator, pauli_label
from scikit_quri.circuit import create_qcl_ansatz
from scikit_quri.backend import SimEstimator
from scikit_quri.qnn._qnn_common import predict_inner

n_qubits = 8
depth = 3
num_class = 3
y_exp_ratio = 2.2
batch_list = [16, 64, 128, 256, 512]
n_runs = 3
operators = [Operator({pauli_label(f"Z {i}"): 1.0}) for i in range(num_class)]


def make_dummy_data(n_samples: int) -> np.ndarray:
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    data = np.random.random((n_samples, 2))
    return scaler.fit_transform(data)


def bench(circuit, estimator, x_scaled, params) -> float:
    # warmup first
    predict_inner(circuit, estimator, operators, x_scaled, params, y_exp_ratio)
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        predict_inner(circuit, estimator, operators, x_scaled, params, y_exp_ratio)
        times.append(time.perf_counter() - start)
    return float(np.mean(times))


print("=" * 70)
print("DIAGNOSTIC 1: Thread scaling (OMP_NUM_THREADS)")
print("=" * 70)
print()

np.random.seed(42)
x_scaled = make_dummy_data(512)
circuit = create_qcl_ansatz(n_qubits, depth, 1.0)
params = 2 * np.pi * np.random.random(circuit.learning_params_count)

for n_threads in [1, 2, 4, 8]:
    os.environ["OMP_NUM_THREADS"] = str(n_threads)

    eq = SimEstimator(use_scaluq=False)
    es = SimEstimator(use_scaluq=True)

    # Cold start each thread count (fresh estimator)
    tq = bench(circuit, eq, x_scaled, params)
    ts = bench(circuit, es, x_scaled, params)

    speedup = tq / ts if ts > 0 else 0
    print(
        f"  OMP_NUM_THREADS={n_threads:2d}: qulacs={tq:.4f}s, scaluq={ts:.4f}s, speedup={speedup:.2f}x"
    )

print()
print("=" * 70)
print("DIAGNOSTIC 2: Time breakdown — circuit conversion vs execution")
print("=" * 70)
print()

# For scaluq, let's check how much time is spent on circuit conversion vs actual execution
import time as _time
from scikit_quri.circuit.circuit import LearningCircuit

# We need to check the breakdown inside predict_inner for scaluq path
# Let's manually do what predict_inner does and time each step

np.random.seed(42)
x_scaled = make_dummy_data(512)
circuit = create_qcl_ansatz(n_qubits, depth, 1.0)
params = 2 * np.pi * np.random.random(circuit.learning_params_count)
es = SimEstimator(use_scaluq=True)

# Warmup
circuit, batched_params = circuit.to_batched(x_scaled, params)
_ = es.estimate_scaluq_batched(operators, circuit, batched_params)

# Timed runs
t_convert_list = []
t_estimate_list = []
for _ in range(10):
    np.random.seed(42)
    c = create_qcl_ansatz(n_qubits, depth, 1.0)
    p = 2 * np.pi * np.random.random(c.learning_params_count)
    es2 = SimEstimator(use_scaluq=True)
    x2 = make_dummy_data(512)

    t0 = _time.perf_counter()
    batched_circuit, batched_params = c.to_batched(x2, p)
    t1 = _time.perf_counter()
    _ = es2.estimate_scaluq_batched(operators, batched_circuit, batched_params)
    t2 = _time.perf_counter()

    t_convert_list.append(t1 - t0)
    t_estimate_list.append(t2 - t1)

print(
    f"  scaluq circuit conversion (to_batched):  {np.mean(t_convert_list) * 1000:.1f}ms ±{np.std(t_convert_list) * 1000:.1f}ms"
)
print(
    f"  scaluq batched estimation:               {np.mean(t_estimate_list) * 1000:.1f}ms ±{np.std(t_estimate_list) * 1000:.1f}ms"
)
print(
    f"  scaluq total:                             {np.mean(t_convert_list) * 1000 + np.mean(t_estimate_list) * 1000:.1f}ms"
)

# Also check qulacs breakdown
print()
t_setup_list = []
t_estimate_list_q = []
for _ in range(10):
    np.random.seed(42)
    c = create_qcl_ansatz(n_qubits, depth, 1.0)
    p = 2 * np.pi * np.random.random(c.learning_params_count)
    eq2 = SimEstimator(use_scaluq=False)
    x2 = make_dummy_data(512)

    t0 = _time.perf_counter()
    from scikit_quri.qnn._qnn_common import build_circuit_states, compute_expectations

    circuit_states = build_circuit_states(c, x2, p)
    t1 = _time.perf_counter()
    _ = compute_expectations(eq2, operators, circuit_states, y_exp_ratio)
    t2 = _time.perf_counter()

    t_setup_list.append(t1 - t0)
    t_estimate_list_q.append(t2 - t1)

print(
    f"  qulacs circuit setup (build_circuit_states): {np.mean(t_setup_list) * 1000:.1f}ms ±{np.std(t_setup_list) * 1000:.1f}ms"
)
print(
    f"  qulacs estimation (compute_expectations):    {np.mean(t_estimate_list_q) * 1000:.1f}ms ±{np.std(t_estimate_list_q) * 1000:.1f}ms"
)
print(
    f"  qulacs total:                                 {np.mean(t_setup_list) * 1000 + np.mean(t_estimate_list_q) * 1000:.1f}ms"
)

print()
print("=" * 70)
print("DIAGNOSTIC 3: Kokkos backend and config")
print("=" * 70)
print()

# Check Kokkos configuration
import ctypes

try:
    # Try to get Kokkos config from the running process
    import scaluq

    print(f"  scaluq version/import: OK")
    print(f"  scaluq location: {scaluq.__file__ if hasattr(scaluq, '__file__') else '?'}")
except Exception as e:
    print(f"  scaluq import error: {e}")

# Check Kokkos via env vars
print(f"  OMP_NUM_THREADS (current): {os.environ.get('OMP_NUM_THREADS', 'not set')}")
print(f"  OMP_PROC_BIND: {os.environ.get('OMP_PROC_BIND', 'not set')}")
print(f"  OMP_PLACES: {os.environ.get('OMP_PLACES', 'not set')}")

# Try checking Kokkos config via scaluq's _backend
try:
    from quri_parts_scaluq import _backend

    print(f"  _backend type: {type(_backend)}")
    # Check if there's any Kokkos-related info
    for attr in dir(_backend):
        if (
            "kokkos" in attr.lower()
            or "omp" in attr.lower()
            or "thread" in attr.lower()
            or "config" in attr.lower()
        ):
            print(f"  _backend.{attr}: {getattr(_backend, attr)}")
except Exception as e:
    print(f"  _backend inspection error: {e}")

print()
print("=" * 70)
print("DIAGNOSTIC 4: Strong scaling — fixed batch, varying qubits")
print("=" * 70)
print()

# Test with different qubit counts to see if speedup scales with problem size
qubit_list = [4, 6, 8, 10]
fixed_batch = 128

for nq in qubit_list:
    np.random.seed(42)
    c = create_qcl_ansatz(nq, depth, 1.0)
    p = 2 * np.pi * np.random.random(c.learning_params_count)
    x = make_dummy_data(fixed_batch)

    eq = SimEstimator(use_scaluq=False)
    es = SimEstimator(use_scaluq=True)

    tq = bench(c, eq, x, p)
    ts = bench(c, es, x, p)
    speedup = tq / ts if ts > 0 else 0
    print(
        f"  n_qubits={nq:2d} (batch={fixed_batch}): qulacs={tq:.4f}s, scaluq={ts:.4f}s, speedup={speedup:.2f}x"
    )

print()
print("=" * 70)
print("DIAGNOSTIC 5: Memory allocation / reuse effect")
print("=" * 70)
print()

# Compare: fresh StateVectorBatched each time vs reusing
from quri_parts_scaluq import _backend as _sq_backend
from quri_parts_scaluq.estimator import estimate as scaluq_estimate_v2
from quri_parts_scaluq.circuit import convert_parametric_circuit
from quri_parts_scaluq.operator import convert_operator

np.random.seed(42)
c = create_qcl_ansatz(n_qubits, depth, 1.0)
p = 2 * np.pi * np.random.random(c.learning_params_count)
x = make_dummy_data(512)

circuit_batched, batched_params = c.to_batched(x, p)


# Method A: Fresh state each time (current behavior)
def bench_fresh():
    state = _sq_backend.StateVectorBatched(len(x), n_qubits)
    state.set_zero_state()
    results = scaluq_estimate_v2(state, circuit_batched, operators, batched_params)
    return results


# Method B: Reuse state (set_zero_state between calls)
state_reuse = _sq_backend.StateVectorBatched(len(x), n_qubits)


def bench_reuse():
    state_reuse.set_zero_state()
    results = scaluq_estimate_v2(state_reuse, circuit_batched, operators, batched_params)
    return results


# Warmup
bench_fresh()
bench_reuse()

times_fresh = []
times_reuse = []
for _ in range(10):
    t0 = _time.perf_counter()
    bench_fresh()
    times_fresh.append(_time.perf_counter() - t0)

    t0 = _time.perf_counter()
    bench_reuse()
    times_reuse.append(_time.perf_counter() - t0)

print(
    f"  Fresh StateVectorBatched each call: {np.mean(times_fresh) * 1000:.1f}ms ±{np.std(times_fresh) * 1000:.1f}ms"
)
print(
    f"  Reuse StateVectorBatched + set_zero: {np.mean(times_reuse) * 1000:.1f}ms ±{np.std(times_reuse) * 1000:.1f}ms"
)
print(f"  Reuse speedup: {np.mean(times_fresh) / np.mean(times_reuse):.2f}x")
print(f"  (difference = allocation overhead of StateVectorBatched)")

print()
print("Done! ✅")
