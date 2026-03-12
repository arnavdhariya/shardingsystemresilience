"""
collect_data.py

Batch ablation sweeps for the sharding/search simulator.

Outputs:
  results/
    sweep_servers.csv
    sweep_rebalance.csv
    sweep_steps.csv
    all_results_long.csv
"""

from __future__ import annotations
import csv
import os
import random
from typing import Dict, Any, List

from server import Server, SimpleHeatModel
from search import IVFSearch
from workload import Workload, BurstyHotsetQueryDist, zipf_weights
from sharding import build_initial_state
from simulator import Simulator


# =========================
# EDIT THESE ONLY
# =========================

OUTDIR = "results"

POLICY = "greedy"        # "greedy" or "ilp"
REPLICATES = 3        # runs per configuration
BASE_SEED = 42

# Fixed parameters
N_KEYS = 512
NPROBE = 6
ZIPF_S = 1.2
SERVER_CAPACITY = 1000.0
INITIAL_SLICES = 32
CHURN_BUDGET = 5.0
MIGRATION_K = 3

# ILP safety knobs
GUROBI_TIME_LIMIT = 10.0
GUROBI_MIP_GAP = 0.05

# Ablation ranges
SERVER_COUNTS = [8, 16, 32, 64, 128, 256]
REBALANCE_EVERY = [10, 25, 50, 100, 250, 500, 1000]
TOTAL_STEPS = [500, 1000, 2000, 4000, 8000, 16000]

# Baselines used when sweeping one variable
BASE_SERVERS = 32
BASE_REBALANCE = 50
BASE_STEPS = 2000


# =========================
# Helper functions
# =========================

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def run_sim(
    *,
    seed: int,
    n_servers: int,
    steps: int,
    rebalance_every: int,
) -> Dict[str, Any]:

    rng = random.Random(seed)

    # servers
    servers = [Server(sid=i, capacity=SERVER_CAPACITY) for i in range(n_servers)]

    # cluster sizes
    key_sizes = [1.0 + 0.2 * rng.random() for _ in range(N_KEYS)]

    # zipf distribution
    weights = zipf_weights(N_KEYS, s=ZIPF_S, rng=rng)

    # cost per cluster
    list_cost = [1.0 + 3.0 * weights[i] * N_KEYS for i in range(N_KEYS)]

    # IVF search
    search = IVFSearch(
        rng=rng,
        n_lists=N_KEYS,
        nprobe=NPROBE,
        list_cost=list_cost,
        list_weights=weights,
    )

    # bursty hotspot workload
    workload = Workload(
        rng,
        BurstyHotsetQueryDist(
            n_lists=N_KEYS,
            n_hot=2,
            hot_mass=0.8,
            burst_mean=200,
            burst_std=50,
            burst_min=20,
            burst_max=800,
        ),
    )

    # shard layout
    state = build_initial_state(
        rng=rng,
        servers=servers,
        n_keys=N_KEYS,
        key_sizes=key_sizes,
        initial_slices=INITIAL_SLICES,
        window=200,
    )

    state.invariant_check(N_KEYS)

    # server model
    server_model = SimpleHeatModel(
        alpha=0.0005,
        beta=0.01,
        decay=0.02,
    )

    # simulator
    sim = Simulator(
        rng=rng,
        workload=workload,
        search=search,
        sharding_state=state,
        server_model=server_model,
        n_keys=N_KEYS,
    )

    stats = sim.run(
        steps=steps,
        rebalance_every=rebalance_every,
        policy=POLICY,
        churn_budget=CHURN_BUDGET,
        key_sizes=key_sizes,
        migration_K=MIGRATION_K,
        gurobi_time_limit_s=GUROBI_TIME_LIMIT,
        gurobi_mip_gap=GUROBI_MIP_GAP,
    )

    return {
        "policy": POLICY,
        "n_servers": n_servers,
        "steps": steps,
        "rebalance_every": rebalance_every,
        "avg_latency": stats.avg_latency,
        "p95_latency": stats.p95_latency,
        "imbalance": stats.imbalance,
        "hottest_server": stats.hottest_server,
        "max_server_heat": stats.max_server_heat,
    }


def write_csv(path: str, rows: List[Dict[str, Any]]):

    if not rows:
        return

    # collect all possible keys
    fieldnames = set()
    for r in rows:
        fieldnames.update(r.keys())

    fieldnames = sorted(fieldnames)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def sweep(name: str, configs: List[Dict[str, int]]) -> List[Dict[str, Any]]:

    print(f"\n=== Sweep: {name} ===")

    rows = []

    for cfg in configs:
        for r in range(REPLICATES):

            seed = BASE_SEED + r

            try:

                out = run_sim(seed=seed, **cfg)

                out["replicate"] = r

                rows.append(out)

                print(" OK", cfg, f"rep={r}")

            except Exception as e:

                rows.append({
                    **cfg,
                    "replicate": r,
                    "error": str(e)
                })

                print(" FAIL", cfg, f"rep={r}", e)

    return rows


# =========================
# Main
# =========================

def main():

    ensure_dir(OUTDIR)

    all_rows = []

    # ---- sweep #servers ----
    server_cfgs = [
        dict(n_servers=s, steps=BASE_STEPS, rebalance_every=BASE_REBALANCE)
        for s in SERVER_COUNTS
    ]

    rows = sweep("servers", server_cfgs)

    write_csv(f"{OUTDIR}/sweep_servers.csv", rows)

    all_rows += rows


    # ---- sweep rebalance frequency ----
    rebalance_cfgs = [
        dict(n_servers=BASE_SERVERS, steps=BASE_STEPS, rebalance_every=r)
        for r in REBALANCE_EVERY
    ]

    rows = sweep("rebalance", rebalance_cfgs)

    write_csv(f"{OUTDIR}/sweep_rebalance.csv", rows)

    all_rows += rows


    # ---- sweep total steps ----
    step_cfgs = [
        dict(n_servers=BASE_SERVERS, steps=s, rebalance_every=BASE_REBALANCE)
        for s in TOTAL_STEPS
    ]

    rows = sweep("steps", step_cfgs)

    write_csv(f"{OUTDIR}/sweep_steps.csv", rows)

    all_rows += rows


    # ---- combined ----
    write_csv(f"{OUTDIR}/all_results_long.csv", all_rows)

    print("\nAll sweeps complete.")
    print(f"Results written to ./{OUTDIR}/")


if __name__ == "__main__":
    main()