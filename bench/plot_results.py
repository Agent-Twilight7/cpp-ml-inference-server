"""Turns bench/results/*.json into the charts in docs/images/.

Usage: python bench/plot_results.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"
OUT = ROOT / "docs" / "images"

# Validated categorical palette (slots 1-2) and neutral inks.
ON_COLOR = "#2a78d6"   # batching on
OFF_COLOR = "#eb6834"  # batching off
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e7e6e2"

SERIES = {
    "on": dict(color=ON_COLOR, marker="o", label="Dynamic batching (max 16)"),
    "off": dict(color=OFF_COLOR, marker="s", label="No batching"),
}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "font.size": 10,
    "legend.frameon": False,
    "legend.labelcolor": INK_2,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


def load(name: str) -> list[dict]:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else []


def by(rows: list[dict], key: str, value) -> list[dict]:
    return [r for r in rows if r.get(key) == value]


def end_label(ax, x, y, text, dy=0):
    """Direct label at the end of a line, in ink (the line's own marker carries the color)."""
    ax.annotate(text, (x, y), xytext=(8, dy), textcoords="offset points", va="center", ha="left",
                color=INK_2, fontsize=9)


def plain_log_axis(ax, ticks):
    """Log scale with plain-number tick labels (5, 10, 20, ...) instead of 10^n."""
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(ticks)


def plot_concurrency() -> None:
    rows = load("concurrency")
    if not rows:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for key in ["off", "on"]:
        rs = sorted(by(rows, "batching", key), key=lambda r: r["connections"])
        x = [r["connections"] for r in rs]
        ax1.plot(x, [r["throughput_rps"] for r in rs], **SERIES[key])
        ax2.plot(x, [r["p99_ms"] for r in rs], **SERIES[key])
        end_label(ax1, x[-1], rs[-1]["throughput_rps"], f"{rs[-1]['throughput_rps']:,.0f} req/s")
        end_label(ax2, x[-1], rs[-1]["p99_ms"], f"{rs[-1]['p99_ms']:,.0f} ms")
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256], ["1", "2", "4", "8", "16", "32", "64", "128", "256"])
        ax.set_xlabel("Concurrent connections (closed loop)")
    ax1.set_title("Throughput")
    ax1.set_ylabel("Requests / second")
    ax1.set_ylim(bottom=0)
    ax2.set_title("p99 latency")
    ax2.set_ylabel("Milliseconds (log scale)")
    plain_log_axis(ax2, [5, 10, 20, 50, 100, 200, 500])
    ax1.legend(loc="upper left")
    fig.suptitle("ResNet-18 on an RTX 4060 Laptop GPU: batching vs concurrency", x=0.01, ha="left",
                 color=INK, fontsize=13, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(OUT / "concurrency.png")
    plt.close(fig)


def plot_open_loop() -> None:
    rows = load("open_loop")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for key in ["off", "on"]:
        rs = sorted(by(rows, "batching", key), key=lambda r: r["target_rps"])
        x = [r["target_rps"] for r in rs]
        ax.plot(x, [r["p99_ms"] for r in rs], **{**SERIES[key], "label": SERIES[key]["label"] + ", p99"})
        ax.plot(x, [r["p50_ms"] for r in rs], color=SERIES[key]["color"], marker=SERIES[key]["marker"],
                linestyle="--", linewidth=1.5, markersize=4, alpha=0.8, label=SERIES[key]["label"] + ", p50")
    plain_log_axis(ax, [5, 10, 20, 50, 100, 200, 500, 1000, 2000])
    ax.set_xlabel("Offered load (requests / second, Poisson arrivals)")
    ax.set_ylabel("Latency (ms, log scale)")
    ax.set_title("Latency vs offered load (open loop, coordinated-omission safe)")
    ax.set_xlim(left=0)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "latency_vs_load.png")
    plt.close(fig)


def plot_max_delay() -> None:
    rows = sorted(load("max_delay"), key=lambda r: r["max_delay_us"])
    if not rows:
        return
    x = [r["max_delay_us"] / 1000 for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    ax1.plot(x, [r["p99_ms"] for r in rows], color=ON_COLOR, marker="o", label="p99")
    ax1.plot(x, [r["p50_ms"] for r in rows], color=ON_COLOR, marker="o", linestyle="--", linewidth=1.5,
             markersize=4, alpha=0.8, label="p50")
    ax1.set_title("Latency at 1,000 req/s")
    ax1.set_ylabel("Milliseconds")
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="lower right")
    ax2.plot(x, [r["avg_batch"] for r in rows], color=ON_COLOR, marker="o")
    ax2.set_title("Average batch size")
    ax2.set_ylabel("Requests per batch")
    ax2.set_ylim(bottom=0)
    for ax in (ax1, ax2):
        ax.set_xlabel("--max-delay (ms)")
    fig.suptitle("The batching delay knob: waiting longer fills batches, at a latency cost", x=0.01, ha="left",
                 color=INK, fontsize=13, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(OUT / "max_delay.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_concurrency()
    plot_open_loop()
    plot_max_delay()
    print("charts written to", OUT)


if __name__ == "__main__":
    main()
