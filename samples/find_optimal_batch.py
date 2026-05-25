"""Find the optimal batch size for scaluq batched estimation.

Sweeps batch sizes across multiple qubit counts, measures predict_inner time
on both qulacs and scaluq backends, and reports speedup, throughput
(samples/sec), and the actual peak RSS delta (psutil) observed during the
scaluq runs.

Outputs:
    samples/optimal_batch_results.csv  — raw per-row measurements
    samples/optimal_batch_plot.png     — 2x2 subplots (speedup, throughput, memory log, memory linear)

Usage:
    uv run samples/find_optimal_batch.py
"""

from __future__ import annotations

import csv
import gc
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
from matplotlib import font_manager
from quri_parts.core.operator import Operator, pauli_label
from sklearn.preprocessing import MinMaxScaler

from scikit_quri.backend import SimEstimator
from scikit_quri.circuit import create_qcl_ansatz
from scikit_quri.qnn._qnn_common import predict_inner

_PROC = psutil.Process()

# --- Settings ---
BATCH_LIST = [8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
QUBIT_LIST = [5, 8, 10]
DEPTH = 3
NUM_CLASS = 3
Y_EXP_RATIO = 2.2
N_RUNS = 3

OPERATORS = [Operator({pauli_label(f"Z {i}"): 1.0}) for i in range(NUM_CLASS)]

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "optimal_batch_results.csv"
PNG_PATH = SCRIPT_DIR / "optimal_batch_plot.png"


def make_data(n_samples: int) -> np.ndarray:
    rng = np.random.RandomState(42)
    data = rng.random((n_samples, 2))
    return MinMaxScaler(feature_range=(-1.0, 1.0)).fit_transform(data)


def bench(circuit, estimator, x, params, *, measure_memory: bool = False) -> tuple[float, float]:
    """Run predict_inner N_RUNS times.

    Returns (mean_time_s, peak_rss_delta_mb). With measure_memory=True a polling
    thread tracks peak RSS during the timed runs and the delta from a gc'd
    baseline is reported. With measure_memory=False the memory value is 0.0.
    """

    def fn() -> None:
        predict_inner(circuit, estimator, OPERATORS, x, params, Y_EXP_RATIO)

    fn()  # warmup

    stop = threading.Event()
    monitor_thread: threading.Thread | None = None
    baseline = 0
    peak = 0

    if measure_memory:
        gc.collect()
        baseline = _PROC.memory_info().rss
        peak = baseline

        def monitor() -> None:
            nonlocal peak
            while not stop.is_set():
                rss = _PROC.memory_info().rss
                if rss > peak:
                    peak = rss
                time.sleep(0.005)

        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.start()

    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)

    if monitor_thread is not None:
        stop.set()
        monitor_thread.join()
        delta_mb = max(0.0, (peak - baseline) / (1024**2))
    else:
        delta_mb = 0.0

    return float(np.mean(times)), delta_mb


def setup_jp_font() -> None:
    candidates = [
        "Hiragino Sans",
        "Hiragino Maru Gothic Pro",
        "Yu Gothic",
        "Noto Sans CJK JP",
        "IPAexGothic",
        "TakaoGothic",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for f in candidates:
        if f in available:
            plt.rcParams["font.family"] = f
            break
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    setup_jp_font()

    print("=" * 96)
    print(f"Sweep: batches={BATCH_LIST}")
    print(f"       qubits={QUBIT_LIST}, depth={DEPTH}, n_runs={N_RUNS} (+1 warmup)")
    print("=" * 96)
    header = (
        f"{'n_qubits':>8} | {'batch':>5} | {'qulacs (s)':>11} | {'scaluq (s)':>11} | "
        f"{'speedup':>9} | {'scaluq sps':>11} | {'qulacs sps':>11} | {'mem (MB)':>9}"
    )
    print(header)
    print("-" * len(header))

    rows: list[dict] = []
    for n_qubits in QUBIT_LIST:
        np.random.seed(42)
        circuit = create_qcl_ansatz(n_qubits, DEPTH, 1.0)
        params = 2 * np.pi * np.random.random(circuit.learning_params_count)

        estimator_q = SimEstimator(use_scaluq=False)
        estimator_s = SimEstimator(use_scaluq=True)

        for batch in BATCH_LIST:
            x = make_data(batch)

            t_q, _ = bench(circuit, estimator_q, x, params)
            t_s, mem = bench(circuit, estimator_s, x, params, measure_memory=True)

            speedup = t_q / t_s if t_s > 0 else float("nan")
            thr_q = batch / t_q if t_q > 0 else float("nan")
            thr_s = batch / t_s if t_s > 0 else float("nan")

            print(
                f"{n_qubits:>8} | {batch:>5} | {t_q:>11.4f} | {t_s:>11.4f} | "
                f"{speedup:>8.2f}x | {thr_s:>11.1f} | {thr_q:>11.1f} | {mem:>9.2f}"
            )

            rows.append(
                dict(
                    n_qubits=n_qubits,
                    batch_size=batch,
                    qulacs_time_s=t_q,
                    scaluq_time_s=t_s,
                    speedup=speedup,
                    qulacs_throughput_sps=thr_q,
                    scaluq_throughput_sps=thr_s,
                    memory_mb=mem,
                )
            )

    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved: {CSV_PATH}")

    # --- Plot: speedup, throughput, memory (log), memory (linear) ---
    fig, axes_grid = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes_grid.flatten()
    colors = {5: "#1f77b4", 8: "#ff7f0e", 10: "#2ca02c"}
    markers = {5: "o", 8: "s", 10: "^"}

    for nq in QUBIT_LIST:
        sub = [r for r in rows if r["n_qubits"] == nq]
        bs = [r["batch_size"] for r in sub]
        sp = [r["speedup"] for r in sub]
        thr_s = [r["scaluq_throughput_sps"] for r in sub]
        thr_q = [r["qulacs_throughput_sps"] for r in sub]
        mem = [r["memory_mb"] for r in sub]

        axes[0].plot(
            bs,
            sp,
            marker=markers[nq],
            color=colors[nq],
            linewidth=2,
            markersize=8,
            label=f"n_qubits={nq}",
        )
        axes[1].plot(
            bs,
            thr_s,
            marker=markers[nq],
            color=colors[nq],
            linewidth=2,
            markersize=8,
            label=f"scaluq (n_qubits={nq})",
        )
        axes[1].plot(
            bs,
            thr_q,
            marker=markers[nq],
            color=colors[nq],
            linewidth=1.5,
            markersize=6,
            linestyle="--",
            alpha=0.6,
            label=f"qulacs (n_qubits={nq})",
        )
        axes[2].plot(
            bs,
            mem,
            marker=markers[nq],
            color=colors[nq],
            linewidth=2,
            markersize=8,
            label=f"n_qubits={nq}",
        )
        axes[3].plot(
            bs,
            mem,
            marker=markers[nq],
            color=colors[nq],
            linewidth=2,
            markersize=8,
            label=f"n_qubits={nq}",
        )

    for ax in axes[:3]:
        ax.set_xscale("log", base=2)
        ax.set_xticks(BATCH_LIST)
        ax.set_xticklabels([str(b) for b in BATCH_LIST], rotation=45)
        ax.grid(True, alpha=0.3, linestyle="--", which="both")

    axes[0].axhline(1.0, color="gray", linewidth=0.8, alpha=0.5)
    axes[0].set_xlabel("バッチサイズ", fontsize=12)
    axes[0].set_ylabel("高速化倍率 (qulacs / scaluq)", fontsize=12)
    axes[0].set_title("バッチサイズ別 高速化倍率", fontsize=13, fontweight="bold")
    axes[0].legend(loc="best", fontsize=10)

    axes[1].set_yscale("log")
    axes[1].set_xlabel("バッチサイズ", fontsize=12)
    axes[1].set_ylabel("スループット (samples/秒)", fontsize=12)
    axes[1].set_title("バッチサイズ別 スループット", fontsize=13, fontweight="bold")
    axes[1].legend(loc="best", fontsize=9)

    axes[2].set_yscale("log")
    axes[2].set_xlabel("バッチサイズ", fontsize=12)
    axes[2].set_ylabel("プロセスメモリ実測 ΔRSS (MB)", fontsize=12)
    axes[2].set_title("バッチサイズ別 メモリ使用量 (実測・対数)", fontsize=13, fontweight="bold")
    axes[2].legend(loc="best", fontsize=10)

    axes[3].grid(True, alpha=0.3, linestyle="--")
    axes[3].set_xlabel("バッチサイズ", fontsize=12)
    axes[3].set_ylabel("プロセスメモリ実測 ΔRSS (MB)", fontsize=12)
    axes[3].set_title("バッチサイズ別 メモリ使用量 (実測・線形)", fontsize=13, fontweight="bold")
    axes[3].legend(loc="best", fontsize=10)

    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
    print(f"Plot saved:    {PNG_PATH}")

    # --- Recommendation ---
    print()
    print("=" * 96)
    print("RECOMMENDATION")
    print("=" * 96)
    for nq in QUBIT_LIST:
        sub = [r for r in rows if r["n_qubits"] == nq]
        peak = max(sub, key=lambda r: r["scaluq_throughput_sps"])
        # "knee": smallest batch reaching >=90% of peak scaluq throughput
        knee = next(
            (r for r in sub if r["scaluq_throughput_sps"] >= 0.9 * peak["scaluq_throughput_sps"]),
            peak,
        )
        best_speedup = max(sub, key=lambda r: r["speedup"])

        print(f"\n  n_qubits={nq}:")
        print(
            f"    最大スループット: batch={peak['batch_size']:4d}  "
            f"({peak['scaluq_throughput_sps']:.0f} samples/s, "
            f"speedup {peak['speedup']:.2f}x, "
            f"mem {peak['memory_mb']:.2f} MB)"
        )
        print(
            f"    最大高速化:       batch={best_speedup['batch_size']:4d}  "
            f"({best_speedup['speedup']:.2f}x, "
            f"scaluq {best_speedup['scaluq_throughput_sps']:.0f} samples/s, "
            f"mem {best_speedup['memory_mb']:.2f} MB)"
        )
        print(
            f"    推奨 (90%到達):   batch={knee['batch_size']:4d}  "
            f"({knee['scaluq_throughput_sps']:.0f} samples/s, "
            f"speedup {knee['speedup']:.2f}x, "
            f"mem {knee['memory_mb']:.2f} MB)"
        )


if __name__ == "__main__":
    main()
