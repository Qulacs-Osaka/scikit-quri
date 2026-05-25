"""Plot large batch profile results — check cache overflow cliff."""

import matplotlib.pyplot as plt
from matplotlib import font_manager

# Data from profile_large_batch.py (n_qubits=8)
batch_sizes = [128, 256, 512, 1024, 1536, 2048, 3072, 4096]
qulacs_times = [0.2148, 0.4471, 0.8677, 1.7302, 2.6007, 3.4558, 5.2002, 6.9325]
scaluq_times = [0.0195, 0.0251, 0.0415, 0.0751, 0.1092, 0.1486, 0.2218, 0.3553]
speedups = [11.02, 17.84, 20.91, 23.04, 23.83, 23.26, 23.45, 19.51]
sps_scaluq = [6570, 10216, 12338, 13636, 14072, 13784, 13850, 11529]
state_mem = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0]

# Japanese font setup
candidates = ["Hiragino Sans", "Yu Gothic", "Noto Sans CJK JP"]
available = {f.name for f in font_manager.fontManager.ttflist}
for f in candidates:
    if f in available:
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel 1: Speedup vs batch size
ax = axes[0]
ax.plot(batch_sizes, speedups, "o-", color="#1f77b4", linewidth=2, markersize=8)
ax.axvline(x=1536, color="green", linestyle="--", alpha=0.5, label="L2限界 (~12MB)")
ax.axvline(x=4096, color="red", linestyle=":", alpha=0.5, label="L2の4倍")
peak_idx = speedups.index(max(speedups))
ax.annotate(
    f"ピーク {speedups[peak_idx]:.2f}x\nbatch={batch_sizes[peak_idx]}",
    xy=(batch_sizes[peak_idx], speedups[peak_idx]),
    xytext=(batch_sizes[peak_idx] + 400, speedups[peak_idx] - 2),
    arrowprops=dict(arrowstyle="->", color="gray"),
    fontsize=10,
)
ax.set_xlabel("バッチサイズ", fontsize=11)
ax.set_ylabel("高速化倍率", fontsize=11)
ax.set_title("バッチサイズ vs 高速化倍率\n(n_qubits=8)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: Throughput (sps) vs batch size
ax = axes[1]
ax.plot(
    batch_sizes,
    sps_scaluq,
    "s-",
    color="#ff7f0e",
    linewidth=2,
    markersize=8,
    label="scaluq throughput",
)
ax.axvline(x=1536, color="green", linestyle="--", alpha=0.5, label="L2限界 (~12MB)")
ax.axvline(x=4096, color="red", linestyle=":", alpha=0.5, label="L2の4倍")
ax.set_xlabel("バッチサイズ", fontsize=11)
ax.set_ylabel("スループット (samples/s)", fontsize=11)
ax.set_title("バッチサイズ vs スループット", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: Throughput vs Memory footprint
ax = axes[2]
ax.plot(state_mem, sps_scaluq, "^--", color="#2ca02c", linewidth=2, markersize=8)
ax.axvline(x=6, color="green", linestyle="--", alpha=0.5, label="L2限界 (~12MB)")
# Annotate key points
for i in [0, 4, 7]:  # batch=128, 1536, 4096
    ax.annotate(
        f"batch={batch_sizes[i]}",
        xy=(state_mem[i], sps_scaluq[i]),
        xytext=(state_mem[i] + 0.5, sps_scaluq[i] - 500),
        fontsize=9,
    )
ax.set_xlabel("ステートベクトルメモリ (MB)", fontsize=11)
ax.set_ylabel("スループット (samples/s)", fontsize=11)
ax.set_title(
    "メモリ使用量 vs スループット\n(バッチサイズ増加に伴う変化)", fontsize=12, fontweight="bold"
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = "samples/large_batch_profile.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Plot saved to {out}")
