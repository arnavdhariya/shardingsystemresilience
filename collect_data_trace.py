"""
collect_data_trace.py

Full-fidelity 2000-step runner for the sharding/search simulator.

This runner fixes the old step-by-step loop (which prevented periodic rebalance)
by using one continuous `run_timeseries()` call per scenario.

Outputs (by default, under ./results):
  - timeseries_zipf_full_2000_log.txt
  - trace_zipf_full_2000.jsonl
  - timeseries_uniform_full_2000_log.txt
  - trace_uniform_full_2000.jsonl
  - run_config_full_2000.json
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from server import Server, SimpleHeatModel
from search import IVFSearch
from workload import Workload, BurstyHotsetQueryDist, zipf_weights
from sharding import build_initial_state
from simulator import Simulator


@dataclass
class RunnerConfig:
    seed: int = 7
    steps: int = 2000
    rebalance_every: int = 50
    policy: str = "greedy"

    n_servers: int = 100
    n_keys: int = 100
    nprobe: int = 20
    server_capacity: float = 1e9

    n_hot: int = 2
    hot_mass: float = 0.8
    hot_skew: float = 1.2
    fixed_hotset: bool = True
    fixed_hot_ids: List[int] = field(default_factory=lambda: [0, 1])
    burst_mean: int = 200
    burst_std: int = 50
    burst_min: int = 20
    burst_max: int = 800

    zipf_s: float = 1.2
    allow_repeat: bool = True

    cluster_shards_min: int = 6
    cluster_shards_max: int = 10
    chunk_size: int = 5
    initial_slices: int = 64

    churn_budget: float = 50000.0
    migration_k: int = 50
    gurobi_time_limit_s: float = 10.0
    gurobi_mip_gap: float = 0.01

    requests_per_server: int = 10000
    requests_std_frac: float = 0.2
    requests_min: int = 1
    requests_max: int | None = None

    burn_in: int = 200
    fanout_var_window: int = 200
    fanout_request_samples: int = 128

    trace_state_every: int = 1
    reset_history_on_rebalance: bool = True
    include_state_in_rows: bool = False

    split_hot_slices: bool = True
    aggregate_requests: bool = True
    lock_payload_per_step: bool = True

    growth_on_rebalance: bool = True
    growth_popular_fraction: float = 0.02
    growth_popular_k: int = 0
    growth_frac: float = 0.05
    growth_add: float = 0.0
    growth_new_shards_max: int = 3
    print_current_tick: bool = True
    tick_print_every: int = 1

    write_csv: bool = False
    write_txt_log: bool = True

    outdir: str = "results"

    def request_volume(self) -> Tuple[int, int, int, int]:
        req_mean = max(1, self.n_servers * self.requests_per_server)
        req_std = max(0, int(round(req_mean * self.requests_std_frac)))
        req_min = max(1, req_mean - 3 * req_std)
        req_max = req_mean + 3 * req_std
        if self.requests_max is not None:
            req_max = int(self.requests_max)
        req_min = max(req_min, int(self.requests_min))
        req_max = max(req_max, req_min)
        return req_mean, req_std, req_min, req_max


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_sim(
    cfg: RunnerConfig,
    *,
    dist_mode: str,
) -> Tuple[Simulator, List[float], List[int]]:
    rng = random.Random(cfg.seed)

    servers = [Server(sid=i, capacity=cfg.server_capacity) for i in range(cfg.n_servers)]
    server_ids = sorted(s.sid for s in servers)
    key_sizes = [1.0 + 0.2 * rng.random() for _ in range(cfg.n_keys)]

    list_cost = [1.0 for _ in range(cfg.n_keys)]
    base_w = zipf_weights(cfg.n_keys, s=cfg.zipf_s, rng=rng)

    search = IVFSearch(
        rng=rng,
        n_lists=cfg.n_keys,
        nprobe=cfg.nprobe,
        list_cost=list_cost,
        list_weights=base_w,
        allow_repeat=cfg.allow_repeat,
    )

    workload = Workload(
        rng=rng,
        dist=BurstyHotsetQueryDist(
            n_lists=cfg.n_keys,
            n_hot=cfg.n_hot,
            hot_mass=cfg.hot_mass,
            burst_mean=cfg.burst_mean,
            burst_std=cfg.burst_std,
            burst_min=cfg.burst_min,
            burst_max=cfg.burst_max,
            dist_switch_prob=0.0,
            dist_start_mode=dist_mode,
            hot_skew=cfg.hot_skew,
            fixed_hotset=cfg.fixed_hotset,
            fixed_hot_ids=cfg.fixed_hot_ids,
        ),
    )

    state = build_initial_state(
        rng=rng,
        servers=servers,
        n_keys=cfg.n_keys,
        key_sizes=key_sizes,
        initial_slices=cfg.initial_slices,
        window=cfg.fanout_var_window,
        chunk_size=(cfg.chunk_size if cfg.chunk_size > 0 else None),
        cluster_shards_min=cfg.cluster_shards_min,
        cluster_shards_max=cfg.cluster_shards_max,
    )
    state.invariant_check(cfg.n_keys)

    server_model = SimpleHeatModel(alpha=0.0005, beta=0.01, decay=0.02)
    sim = Simulator(
        rng=rng,
        workload=workload,
        search=search,
        sharding_state=state,
        server_model=server_model,
        n_keys=cfg.n_keys,
    )
    return sim, key_sizes, server_ids


def _write_rows_csv(path: Path, rows: List[Dict], server_ids: List[int]) -> None:
    base_cols = [
        "t",
        "latency",
        "fanout",
        "fanout_req_mean",
        "fanout_req_var",
        "fanout_req_min",
        "fanout_req_max",
        "fanout_req_samples",
        "fanout_var",
        "imbalance",
        "imbalance_pre",
        "hottest_server",
        "coldest_server",
        "hottest_work_server",
        "coldest_work_server",
        "hottest_work",
        "coldest_work",
        "mean_work",
        "rebalance",
        "growth_event",
        "growth_added_total",
        "growth_slices",
        "growth_servers",
        "growth_clusters",
        "growth_actions",
        "burn_in",
        "n_requests",
        "work_units",
        "workload",
        "dist_mode",
        "dist_switch",
        "dist_id",
        "burst_id",
        "burst_start",
        "burst_mode",
        "burst_len",
        "burst_remaining",
        "hot_mass",
        "hot_probe_ratio",
        "hot_set",
        "hot_weights",
        "cluster_loads",
        "hottest_slice",
        "coldest_slice",
        "hottest_slice_load",
        "coldest_slice_load",
        "slice_loads",
        "n_slices_total",
        "n_clusters_total",
        "server_work_map",
        "server_heat_map",
        "server_util_map",
        "server_keyspace_map",
        "server_slice_counts",
        "rebalance_actions",
        "rebalance_diff",
        "rebalance_key_migrations",
        "rebalance_cluster_transitions",
        "rebalance_slice_added",
        "rebalance_slice_removed",
        "rebalance_slice_moved",
        "rebalance_key_migration_count",
        "rebalance_cluster_transition_count",
        "rebalance_slice_added_count",
        "rebalance_slice_removed_count",
        "rebalance_slice_moved_count",
        "state_key_to_slice",
        "state_key_to_server",
        "state_cluster_to_slices",
        "state_cluster_to_servers",
        "state_cluster_shard_sizes",
        "state_slices",
    ]
    per_server_cols = (
        [f"work_s{sid}" for sid in server_ids]
        + [f"keyspace_s{sid}" for sid in server_ids]
        + [f"heat_s{sid}" for sid in server_ids]
        + [f"util_s{sid}" for sid in server_ids]
        + [f"slicecount_s{sid}" for sid in server_ids]
    )
    cols = base_cols + per_server_cols

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})


def _parse_json_field(raw: Any, fallback: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if raw is None:
        return fallback
    txt = str(raw).strip()
    if not txt:
        return fallback
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return fallback


def _to_sid_float_map(raw: Any) -> Dict[int, float]:
    parsed = _parse_json_field(raw, {})
    out: Dict[int, float] = {}
    if not isinstance(parsed, dict):
        return out
    for k, v in parsed.items():
        try:
            sid = int(k)
            out[sid] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _fmt(x: float) -> str:
    ax = abs(float(x))
    if ax != 0.0 and ax < 1e-4:
        return f"{x:.6e}"
    return f"{x:.6f}"


def _top_k_items(d: Dict[int, float], k: int = 5) -> List[Tuple[int, float]]:
    return sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


def _write_rows_txt_log(
    path: Path,
    rows: List[Dict[str, Any]],
    server_ids: List[int],
    cfg: RunnerConfig,
    *,
    dist_mode: str,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("Sharding Simulation Narrative Log\n")
        f.write("=" * 88 + "\n")
        f.write(
            f"Scenario: {dist_mode} | Steps: {cfg.steps} | Rebalance every: {cfg.rebalance_every} ticks | "
            f"Servers: {cfg.n_servers} | Clusters: {cfg.n_keys}\n"
        )
        f.write(
            "Imbalance formula used each timestep: imbalance = max(utilization) / mean(utilization)\n"
        )
        f.write("=" * 88 + "\n\n")

        for row in rows:
            t = int(row.get("t", 0) or 0)
            n_requests = int(row.get("n_requests", 0) or 0)
            work_units = int(row.get("work_units", 0) or 0)
            workload = float(row.get("workload", 0.0) or 0.0)
            fanout = float(row.get("fanout", 0.0) or 0.0)
            fanout_req_mean = float(row.get("fanout_req_mean", 0.0) or 0.0)
            fanout_req_var = float(row.get("fanout_req_var", 0.0) or 0.0)
            imbalance_logged = float(row.get("imbalance", 0.0) or 0.0)
            imbalance_pre = float(row.get("imbalance_pre", 0.0) or 0.0)
            rebalance = int(row.get("rebalance", 0) or 0)
            growth_event = int(row.get("growth_event", 0) or 0)

            util_map = _to_sid_float_map(row.get("server_util_map"))
            work_map = _to_sid_float_map(row.get("server_work_map"))
            heat_map = _to_sid_float_map(row.get("server_heat_map"))
            keyspace_map = _to_sid_float_map(row.get("server_keyspace_map"))
            slice_count_raw = _parse_json_field(row.get("server_slice_counts"), {})

            utils = [util_map.get(sid, 0.0) for sid in server_ids]
            mean_util = (sum(utils) / len(utils)) if utils else 0.0
            if utils:
                hottest_sid = max(server_ids, key=lambda sid: util_map.get(sid, 0.0))
                coldest_sid = min(server_ids, key=lambda sid: util_map.get(sid, 0.0))
                max_util = util_map.get(hottest_sid, 0.0)
                min_util = util_map.get(coldest_sid, 0.0)
            else:
                hottest_sid = -1
                coldest_sid = -1
                max_util = 0.0
                min_util = 0.0
            imbalance_calc = (max_util / mean_util) if mean_util > 1e-12 else 0.0

            cluster_loads_raw = _parse_json_field(row.get("cluster_loads"), {})
            cluster_loads: Dict[int, float] = {}
            if isinstance(cluster_loads_raw, dict):
                for k, v in cluster_loads_raw.items():
                    try:
                        cluster_loads[int(k)] = float(v)
                    except (TypeError, ValueError):
                        continue
            top_clusters = _top_k_items(cluster_loads, k=5)

            rebalance_actions = _parse_json_field(row.get("rebalance_actions"), [])
            growth_actions = _parse_json_field(row.get("growth_actions"), [])
            rebalance_diff = _parse_json_field(row.get("rebalance_diff"), {})
            raw_blob = {
                "rebalance_actions": rebalance_actions,
                "growth_actions": growth_actions,
                "rebalance_diff": rebalance_diff,
                "cluster_loads": cluster_loads,
                "hot_set": _parse_json_field(row.get("hot_set"), []),
                "hot_weights": _parse_json_field(row.get("hot_weights"), []),
            }

            f.write(f"Time = {t}\n")
            f.write("-" * 88 + "\n")
            f.write(
                f"At Time = {t}, the system processed {n_requests} requests, generated {work_units} work units, "
                f"and assigned {_fmt(workload)} total workload across servers.\n"
            )
            f.write(
                f"Fan-out at this timestep was {_fmt(fanout)}; per-request fan-out mean was "
                f"{_fmt(fanout_req_mean)} with variance {_fmt(fanout_req_var)}.\n"
            )
            f.write(
                "Imbalance calculation: max_util / mean_util = "
                f"{_fmt(max_util)} / {_fmt(mean_util)} = {_fmt(imbalance_calc)}.\n"
            )
            f.write(
                f"Logged post-rebalance imbalance = {_fmt(imbalance_logged)}; "
                f"pre-rebalance imbalance = {_fmt(imbalance_pre)}.\n"
            )
            f.write(
                f"Max utilization came from server {hottest_sid} with util {_fmt(max_util)}; "
                f"minimum utilization was server {coldest_sid} with util {_fmt(min_util)}.\n"
            )
            f.write(
                f"Mean utilization used in the denominator was {_fmt(mean_util)} "
                f"({(mean_util * 100.0):.6f}% of capacity).\n"
            )
            if top_clusters:
                cluster_txt = ", ".join([f"cluster {k}: {_fmt(v)}" for k, v in top_clusters])
                f.write(f"Top cluster loads at this timestep: {cluster_txt}.\n")
            else:
                f.write("No cluster load was recorded for this timestep.\n")

            if rebalance:
                moved = int(row.get("rebalance_key_migration_count", 0) or 0)
                tr = int(row.get("rebalance_cluster_transition_count", 0) or 0)
                add = int(row.get("rebalance_slice_added_count", 0) or 0)
                rm = int(row.get("rebalance_slice_removed_count", 0) or 0)
                mv = int(row.get("rebalance_slice_moved_count", 0) or 0)
                f.write(
                    "Rebalance event: yes. "
                    f"Key migrations={moved}, cluster transitions={tr}, "
                    f"slice added={add}, removed={rm}, moved={mv}.\n"
                )
            else:
                f.write("Rebalance event: no.\n")

            if growth_event:
                f.write(
                    "Growth event: yes. "
                    f"Added total keyspace={_fmt(float(row.get('growth_added_total', 0.0) or 0.0))}, "
                    f"new slices={int(row.get('growth_slices', 0) or 0)}.\n"
                )
            else:
                f.write("Growth event: no.\n")

            f.write("\nServer Loads Used At This Timestep:\n")
            f.write("server | work | utilization | heat | keyspace | slice_count\n")
            for sid in server_ids:
                work = work_map.get(sid, 0.0)
                util = util_map.get(sid, 0.0)
                heat = heat_map.get(sid, 0.0)
                keyspace = keyspace_map.get(sid, 0.0)
                slices = 0
                if isinstance(slice_count_raw, dict):
                    slices = int(slice_count_raw.get(str(sid), slice_count_raw.get(sid, 0)) or 0)
                f.write(
                    f"{sid} | {_fmt(work)} | {_fmt(util)} | {_fmt(heat)} | {_fmt(keyspace)} | {slices}\n"
                )

            f.write("\nRaw transition payloads (JSON):\n")
            f.write(json.dumps(raw_blob, indent=2, sort_keys=True))
            f.write("\n\n")


def run_scenario(cfg: RunnerConfig, *, dist_mode: str) -> Dict[str, float]:
    outdir = Path(cfg.outdir)
    _ensure_dir(outdir)

    csv_path = outdir / f"timeseries_{dist_mode}_full_2000.csv"
    txt_path = outdir / f"timeseries_{dist_mode}_full_2000_log.txt"
    trace_path = outdir / f"trace_{dist_mode}_full_2000.jsonl"

    sim, key_sizes, server_ids = _build_sim(cfg, dist_mode=dist_mode)
    req_mean, req_std, req_min, req_max = cfg.request_volume()

    print(f"[run] {dist_mode} -> {txt_path.name}, {trace_path.name}")
    rows = sim.run_timeseries(
        steps=cfg.steps,
        rebalance_every=cfg.rebalance_every,
        policy=cfg.policy,
        churn_budget=cfg.churn_budget,
        key_sizes=key_sizes,
        migration_K=cfg.migration_k,
        gurobi_time_limit_s=cfg.gurobi_time_limit_s,
        gurobi_mip_gap=cfg.gurobi_mip_gap,
        burn_in=cfg.burn_in,
        fanout_var_window=cfg.fanout_var_window,
        trace_jsonl=str(trace_path),
        trace_state_every=cfg.trace_state_every,
        reset_history_on_rebalance=cfg.reset_history_on_rebalance,
        requests_mean=req_mean,
        requests_std=req_std,
        requests_min=req_min,
        requests_max=req_max,
        lock_payload_per_step=cfg.lock_payload_per_step,
        split_hot_slices=cfg.split_hot_slices,
        aggregate_requests=cfg.aggregate_requests,
        fanout_request_samples=cfg.fanout_request_samples,
        growth_on_rebalance=cfg.growth_on_rebalance,
        growth_popular_fraction=cfg.growth_popular_fraction,
        growth_popular_k=cfg.growth_popular_k,
        growth_frac=cfg.growth_frac,
        growth_add=cfg.growth_add,
        growth_new_shards_max=cfg.growth_new_shards_max,
        include_state_in_rows=cfg.include_state_in_rows,
        print_current_tick=cfg.print_current_tick,
        tick_print_every=cfg.tick_print_every,
    )

    if cfg.write_csv:
        _write_rows_csv(csv_path, rows, server_ids)
    if cfg.write_txt_log:
        _write_rows_txt_log(txt_path, rows, server_ids, cfg, dist_mode=dist_mode)

    final = rows[-1]
    print(
        f"[done] {dist_mode}: rows={len(rows)} final_imbalance={float(final['imbalance']):.3f} "
        f"fanout_req_mean={float(final['fanout_req_mean']):.3f}"
    )
    return {
        "rows": float(len(rows)),
        "final_imbalance": float(final["imbalance"]),
        "final_fanout_req_mean": float(final["fanout_req_mean"]),
        "final_fanout_req_var": float(final["fanout_req_var"]),
    }


def main() -> None:
    cfg = RunnerConfig()
    outdir = Path(cfg.outdir)
    _ensure_dir(outdir)

    summary = {
        "config": asdict(cfg),
        "scenarios": {},
    }

    for mode in ("zipf", "uniform"):
        summary["scenarios"][mode] = run_scenario(cfg, dist_mode=mode)

    config_path = outdir / "run_config_full_2000.json"
    config_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[summary] {config_path}")


if __name__ == "__main__":
    main()
