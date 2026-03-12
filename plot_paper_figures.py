from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List


def _ensure_mplconfig() -> None:
    cfg_dir = Path(__file__).resolve().parent / ".mplconfig"
    cache_dir = Path(__file__).resolve().parent / ".cache"
    cfg_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cfg_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


_ensure_mplconfig()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


def set_vldb_style() -> None:
    """Paper-style rcParams inspired by common database systems figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.constrained_layout.use": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _parse_json_cell(value: str, default: Any) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def load_timeseries(path: Path) -> Dict[str, List[float]]:
    rows = list(csv.DictReader(path.open()))
    data: Dict[str, List[Any]] = {
        "t": [],
        "n_requests": [],
        "workload": [],
        "hot_workload": [],
        "cold_workload": [],
        "primary_hot_workload": [],
        "secondary_hot_workload": [],
        "hot_probe_units": [],
        "cold_probe_units": [],
        "imbalance": [],
        "imbalance_pre": [],
        "fanout": [],
        "fanout_req_mean": [],
        "fanout_req_var": [],
        "fanout_req_min": [],
        "fanout_req_max": [],
        "fanout_req_samples": [],
        "fanout_var": [],
        "rebalance": [],
    }

    for row in rows:
        t = int(float(row["t"]))
        n_requests = float(row["n_requests"])
        workload = float(row["workload"])
        imbalance = float(row["imbalance"])
        imbalance_pre = float(row["imbalance_pre"])
        fanout = float(row["fanout"])
        fanout_var = float(row["fanout_var"])
        fanout_req_mean = float(row.get("fanout_req_mean") or fanout)
        fanout_req_var = float(row.get("fanout_req_var") or 0.0)
        fanout_req_min = float(row.get("fanout_req_min") or fanout_req_mean)
        fanout_req_max = float(row.get("fanout_req_max") or fanout_req_mean)
        fanout_req_samples = int(float(row.get("fanout_req_samples") or 0))
        rebalance = int(float(row["rebalance"]))
        work_units = float(row["work_units"])
        hot_probe_ratio = float(row["hot_probe_ratio"])

        hot_set_raw = _parse_json_cell(row.get("hot_set", ""), [])
        hot_set = [int(x) for x in hot_set_raw if isinstance(x, (int, float))]

        cluster_loads_raw = _parse_json_cell(row.get("cluster_loads", ""), {})
        cluster_loads: Dict[int, float] = {}
        if isinstance(cluster_loads_raw, dict):
            for k, v in cluster_loads_raw.items():
                try:
                    cluster_loads[int(k)] = float(v)
                except (TypeError, ValueError):
                    continue

        hot_loads = sorted((cluster_loads.get(k, 0.0) for k in hot_set), reverse=True)
        hot_workload = sum(hot_loads)
        cold_workload = max(0.0, workload - hot_workload)
        primary_hot_workload = hot_loads[0] if hot_loads else 0.0
        secondary_hot_workload = hot_loads[1] if len(hot_loads) > 1 else 0.0

        data["t"].append(t)
        data["n_requests"].append(n_requests)
        data["workload"].append(workload)
        data["hot_workload"].append(hot_workload)
        data["cold_workload"].append(cold_workload)
        data["primary_hot_workload"].append(primary_hot_workload)
        data["secondary_hot_workload"].append(secondary_hot_workload)
        data["hot_probe_units"].append(work_units * hot_probe_ratio)
        data["cold_probe_units"].append(max(0.0, work_units * (1.0 - hot_probe_ratio)))
        data["imbalance"].append(imbalance)
        data["imbalance_pre"].append(imbalance_pre)
        data["fanout"].append(fanout)
        data["fanout_req_mean"].append(fanout_req_mean)
        data["fanout_req_var"].append(fanout_req_var)
        data["fanout_req_min"].append(fanout_req_min)
        data["fanout_req_max"].append(fanout_req_max)
        data["fanout_req_samples"].append(fanout_req_samples)
        data["fanout_var"].append(fanout_var)
        data["rebalance"].append(rebalance)

    return data


def _rebalance_times(data: Dict[str, List[float]]) -> List[int]:
    return [int(t) for t, flag in zip(data["t"], data["rebalance"]) if int(flag) == 1]


def _add_rebalance_lines(ax: plt.Axes, rebalance_steps: List[int], label: str = "Rebalance") -> None:
    if not rebalance_steps:
        return
    first = True
    for t in rebalance_steps:
        ax.axvline(
            t,
            color="black",
            linewidth=0.8,
            linestyle=(0, (4, 4)),
            alpha=0.5,
            label=(label if first else None),
        )
        first = False


def _save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".png"))
    fig.savefig(base_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_workload_imbalance(zipf: Dict[str, List[float]], uniform: Dict[str, List[float]], outdir: Path) -> None:
    colors = {
        "requests": "#4c566a",
        "hot": "#d08770",
        "cold": "#88c0d0",
        "hot1": "#bf616a",
        "hot2": "#5e81ac",
        "pre": "#4c566a",
        "post": "#2e3440",
    }

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex="col")
    panels = [("Zipf Cold Tail", zipf), ("Uniform Cold Tail", uniform)]

    for col, (title, data) in enumerate(panels):
        t = data["t"]
        rebalance_steps = _rebalance_times(data)

        ax = axes[0][col]
        ax.plot(t, data["n_requests"], color=colors["requests"], label="Requests / timestep")
        ax.fill_between(t, data["n_requests"], color=colors["requests"], alpha=0.10)
        _add_rebalance_lines(ax, rebalance_steps)
        ax.set_title(title)
        ax.set_ylabel("Requests")
        ax.grid(True, axis="y")
        ax.legend(loc="upper right", frameon=False)

        ax = axes[1][col]
        ax.stackplot(
            t,
            data["hot_workload"],
            data["cold_workload"],
            colors=[colors["hot"], colors["cold"]],
            alpha=0.55,
            labels=["Hot-cluster workload (80% band)", "Cold-tail workload (20% band)"],
        )
        ax.plot(
            t,
            data["primary_hot_workload"],
            color=colors["hot1"],
            linestyle="--",
            linewidth=1.2,
            label="Most-loaded hot cluster",
        )
        ax.plot(
            t,
            data["secondary_hot_workload"],
            color=colors["hot2"],
            linestyle=":",
            linewidth=1.2,
            label="Second hot cluster",
        )
        _add_rebalance_lines(ax, rebalance_steps)
        ax.set_ylabel("Probe workload")
        ax.grid(True, axis="y")
        ax.legend(loc="upper right", frameon=False)

        ax = axes[2][col]
        ax.plot(t, data["imbalance_pre"], color=colors["pre"], linestyle="--", label="Imbalance before rebalance")
        ax.plot(t, data["imbalance"], color=colors["post"], label="Imbalance after timestep actions")
        _add_rebalance_lines(ax, rebalance_steps)
        ax.set_ylabel("Imbalance")
        ax.set_xlabel("Time step")
        ax.grid(True)
        ax.legend(loc="upper right", frameon=False)

    fig.suptitle("Time vs Request Volume, 80/20 Workload Split, and Imbalance", y=1.02)
    _save_figure(fig, outdir / "workload_imbalance_requests_comparison")


def plot_fanout(zipf: Dict[str, List[float]], uniform: Dict[str, List[float]], outdir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    panels = [
        ("Zipf Cold Tail", zipf, "#bf616a"),
        ("Uniform Cold Tail", uniform, "#5e81ac"),
    ]

    for ax, (title, data, color) in zip(axes, panels):
        t = data["t"]
        rebalance_steps = _rebalance_times(data)

        fan_line = ax.plot(t, data["fanout_req_mean"], color=color, label="Per-request fan-out (mean)")[0]
        _add_rebalance_lines(ax, rebalance_steps)
        ax.set_title(title)
        ax.set_ylabel("Per-request fan-out")
        ax.grid(True)

        ax2 = ax.twinx()
        var_line = ax2.plot(
            t,
            data["fanout_req_var"],
            color="#4c566a",
            linewidth=1.0,
            linestyle="--",
            label="Per-request fan-out variance",
        )[0]
        ax2.set_ylabel("Fan-out variance")

        handles = [
            fan_line,
            var_line,
            Line2D([0], [0], color="#8fbcbb", linestyle=":", linewidth=1.0, label="Aggregate timestep fan-out"),
            Line2D([0], [0], color="black", linestyle=(0, (4, 4)), linewidth=0.8, label="Rebalance"),
        ]
        ax.plot(
            t,
            data["fanout"],
            color="#8fbcbb",
            linewidth=1.0,
            linestyle=":",
            alpha=0.9,
        )
        ax.legend(handles=handles, loc="upper right", frameon=False)

    axes[-1].set_xlabel("Time step")
    fig.suptitle("Time vs Per-request Fan-out for Zipf and Uniform Workloads", y=1.02)
    _save_figure(fig, outdir / "fanout_time_comparison")


def _rolling_candles(values: List[float], window: int) -> Dict[str, List[float]]:
    lows: List[float] = []
    opens: List[float] = []
    closes: List[float] = []
    highs: List[float] = []

    for idx in range(len(values)):
        lo = max(0, idx - window + 1)
        segment = values[lo: idx + 1]
        mu = sum(segment) / len(segment)
        var = sum((x - mu) ** 2 for x in segment) / len(segment)
        sigma = math.sqrt(var)
        low = min(segment)
        high = max(segment)
        open_ = max(low, mu - sigma)
        close_ = min(high, mu + sigma)
        lows.append(low)
        opens.append(open_)
        closes.append(close_)
        highs.append(high)

    return {"low": lows, "open": opens, "close": closes, "high": highs}


def _request_candles(data: Dict[str, List[float]], window: int) -> Dict[str, List[float]]:
    samples = data.get("fanout_req_samples", [])
    if samples and any(int(x) > 0 for x in samples):
        mean_vals = data["fanout_req_mean"]
        var_vals = data["fanout_req_var"]
        std_vals = [math.sqrt(max(0.0, float(v))) for v in var_vals]
        return {
            "low": [float(x) for x in data["fanout_req_min"]],
            "open": [max(float(m) - s, float(lo)) for m, s, lo in zip(mean_vals, std_vals, data["fanout_req_min"])],
            "close": [min(float(m) + s, float(hi)) for m, s, hi in zip(mean_vals, std_vals, data["fanout_req_max"])],
            "high": [float(x) for x in data["fanout_req_max"]],
        }
    return _rolling_candles([float(x) for x in data["fanout_req_mean"]], window)


def _draw_candles(
    ax: plt.Axes,
    x: List[int],
    candles: Dict[str, List[float]],
    values: List[float],
    color_up: str,
    color_down: str,
) -> None:
    width = 0.70
    prev = values[0] if values else 0.0
    for idx, step in enumerate(x):
        low = candles["low"][idx]
        high = candles["high"][idx]
        open_ = candles["open"][idx]
        close_ = candles["close"][idx]
        color = color_up if values[idx] >= prev else color_down
        ax.vlines(step, low, high, color=color, linewidth=0.45, alpha=0.9)
        body_bottom = min(open_, close_)
        body_height = max(1e-9, abs(close_ - open_))
        ax.add_patch(
            Rectangle(
                (step - width / 2.0, body_bottom),
                width,
                body_height,
                facecolor=color,
                edgecolor=color,
                alpha=0.35,
                linewidth=0.35,
            )
        )
        prev = values[idx]


def plot_fanout_candles(
    zipf: Dict[str, List[float]],
    uniform: Dict[str, List[float]],
    outdir: Path,
    window: int,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    panels = [
        ("Zipf Cold Tail", zipf, "#d08770", "#bf616a"),
        ("Uniform Cold Tail", uniform, "#88c0d0", "#5e81ac"),
    ]

    for ax, (title, data, up_color, down_color) in zip(axes, panels):
        t = [int(x) for x in data["t"]]
        fanout = [float(x) for x in data["fanout_req_mean"]]
        candles = _request_candles(data, window)
        rebalance_steps = _rebalance_times(data)

        _draw_candles(ax, t, candles, fanout, up_color, down_color)
        ax.plot(t, fanout, color="#2e3440", linewidth=0.8, alpha=0.65, label="Per-request fan-out mean")
        _add_rebalance_lines(ax, rebalance_steps)
        ax.set_title(f"{title} (wick = per-step min/max, box = per-step mean ± 1 std)")
        ax.set_ylabel("Per-request fan-out")
        ax.grid(True, axis="y")
        handles = [
            Line2D([0], [0], color="#2e3440", linewidth=0.8, label="Per-request fan-out mean"),
            Patch(facecolor=up_color, edgecolor=up_color, alpha=0.35, label="Per-timestep mean ± 1 std"),
            Line2D([0], [0], color="black", linestyle=(0, (4, 4)), linewidth=0.8, label="Rebalance"),
        ]
        ax.legend(handles=handles, loc="upper right", frameon=False)

    axes[-1].set_xlabel("Time step")
    fig.suptitle("Per-request Fan-out Variance Candlestick View", y=1.02)
    _save_figure(fig, outdir / "fanout_variance_candlestick")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zipf-csv", type=str, default="timeseries_zipf_popular_growth_2000.csv")
    ap.add_argument("--uniform-csv", type=str, default="timeseries_uniform_popular_growth_2000.csv")
    ap.add_argument("--outdir", type=str, default="plots")
    ap.add_argument("--candle-window", type=int, default=50)
    args = ap.parse_args()

    set_vldb_style()

    zipf_path = Path(args.zipf_csv)
    uniform_path = Path(args.uniform_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    zipf = load_timeseries(zipf_path)
    uniform = load_timeseries(uniform_path)

    plot_workload_imbalance(zipf, uniform, outdir)
    plot_fanout(zipf, uniform, outdir)
    plot_fanout_candles(zipf, uniform, outdir, max(2, int(args.candle_window)))

    print("=== PLOTS READY ===")
    print(f"zipf_csv={zipf_path}")
    print(f"uniform_csv={uniform_path}")
    print(f"outdir={outdir}")


if __name__ == "__main__":
    main()
