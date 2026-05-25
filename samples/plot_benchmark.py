"""Visualize benchmark results comparing qulacs vs scaluq batched execution."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# --- Japanese font setup ---
_jp_candidates = [
    "Hiragino Sans",
    "Hiragino Maru Gothic Pro",
    "Yu Gothic",
    "Noto Sans CJK JP",
    "IPAexGothic",
    "TakaoGothic",
]
_available = {f.name for f in font_manager.fontManager.ttflist}
for _font in _jp_candidates:
    if _font in _available:
        plt.rcParams["font.family"] = _font
        break
plt.rcParams["axes.unicode_minus"] = False

# --- Benchmark data ---
batch_sizes = [16, 64, 128, 256, 512]

test3_qulacs = [0.0269, 0.1075, 0.2152, 0.4310, 0.8739]
test3_scaluq = [0.0181, 0.0197, 0.0360, 0.0287, 0.0368]

test4_qulacs = [0.0270, 0.1084, 0.2175, 0.4311, 0.8651]
test4_scaluq = [0.0163, 0.0208, 0.0200, 0.0284, 0.0441]

test5_batches = [16, 128, 512]
test5_qulacs = [0.0274, 0.2195, 0.8637]
test5_scaluq = [0.0170, 0.0231, 0.0521]

readme_batches = [16, 128, 512]
readme_predict_speedup = [1.4, 11.1, 19.7]


def speedup(qulacs_times, scaluq_times):
    return [q / s for q, s in zip(qulacs_times, scaluq_times)]


test3_speedup = speedup(test3_qulacs, test3_scaluq)
test4_speedup = speedup(test4_qulacs, test4_scaluq)
test5_speedup = speedup(test5_qulacs, test5_scaluq)

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel (a): predict_inner speedup comparison
ax1.plot(
    batch_sizes,
    test3_speedup,
    "o-",
    color="#1f77b4",
    linewidth=2,
    markersize=8,
    label="Test3 (Cold, random)",
)
ax1.plot(
    batch_sizes,
    test4_speedup,
    "s-",
    color="#ff7f0e",
    linewidth=2,
    markersize=8,
    label="Test4 (Warm, sequential)",
)
ax1.plot(
    test5_batches,
    test5_speedup,
    "^-",
    color="#2ca02c",
    linewidth=2,
    markersize=10,
    label="Test5 (Subprocess isolated)",
)
ax1.plot(
    readme_batches,
    readme_predict_speedup,
    "D--",
    color="#d62728",
    linewidth=2,
    markersize=8,
    alpha=0.8,
    label="README参考値 (predict_inner)",
)

ax1.set_xscale("log", base=2)
ax1.set_xticks(batch_sizes)
ax1.set_xticklabels([str(b) for b in batch_sizes])
ax1.set_xlabel("バッチサイズ", fontsize=12)
ax1.set_ylabel("高速化倍率 (qulacs / scaluq)", fontsize=12)
ax1.set_title("predict_inner 高速化倍率の比較", fontsize=13, fontweight="bold")
ax1.legend(loc="upper left", fontsize=10)
ax1.grid(True, alpha=0.3, linestyle="--")
ax1.axhline(y=1.0, color="gray", linewidth=0.8, alpha=0.5)

# Panel (b): Absolute time at batch=512
tests = ["Test3\n(Cold)", "Test4\n(Warm)", "Test5\n(Subprocess)"]
qulacs_512 = [test3_qulacs[-1], test4_qulacs[-1], test5_qulacs[-1]]
scaluq_512 = [test3_scaluq[-1], test4_scaluq[-1], test5_scaluq[-1]]

x = np.arange(len(tests))
width = 0.35

bars1 = ax2.bar(
    x - width / 2, qulacs_512, width, label="qulacs", color="#4c72b0", edgecolor="black"
)
bars2 = ax2.bar(
    x + width / 2, scaluq_512, width, label="scaluq", color="#dd8452", edgecolor="black"
)

for bars in (bars1, bars2):
    for bar in bars:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.015,
            f"{h:.4f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

for i, (q, s) in enumerate(zip(qulacs_512, scaluq_512)):
    ax2.text(
        i,
        max(q, s) + 0.08,
        f"{q / s:.1f}x",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="#c44e52",
    )

ax2.set_xticks(x)
ax2.set_xticklabels(tests)
ax2.set_ylabel("実行時間 (秒)", fontsize=12)
ax2.set_title("絶対実行時間の比較 (batch=512)", fontsize=13, fontweight="bold")
ax2.legend(loc="upper right", fontsize=11)
ax2.grid(True, alpha=0.3, linestyle="--", axis="y")
ax2.set_ylim(0, max(qulacs_512) * 1.2)

plt.tight_layout()
out_path = "samples/benchmark_plot.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Plot saved to {out_path}")
