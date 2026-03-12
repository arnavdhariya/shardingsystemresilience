# main.py (patched for per-timestep CSV + bursty workloads)
import argparse
import random
import csv
from pathlib import Path

from server import Server, SimpleHeatModel
from search import IVFSearch
from workload import Workload, BurstyHotsetQueryDist, zipf_weights
from sharding import build_initial_state
from simulator import Simulator


def main():
    ap = argparse.ArgumentParser()

    # core sim
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--policy", choices=["greedy", "ilp"], default="greedy")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--rebalance-every", type=int, default=50)

    # scale
    ap.add_argument("--n-servers", type=int, default=100)
    ap.add_argument("--n-keys", type=int, default=100, help="logical clusters (keys)")

    # IVF probing (fan-out driver)
    ap.add_argument("--nprobe", type=int, default=20)

    # initial sharding ("split 100 into chunks of 5, round robin")
    ap.add_argument("--chunk-size", type=int, default=5,
                    help="if >0, initial slices are contiguous chunks of this size")
    ap.add_argument("--initial-slices", type=int, default=64,
                    help="used only when --chunk-size <= 0")
    ap.add_argument("--cluster-shards-min", type=int, default=2,
                    help="minimum initial sub-shards per logical cluster (key)")
    ap.add_argument("--cluster-shards-max", type=int, default=5,
                    help="maximum initial sub-shards per logical cluster (key)")

    # hotspot model (bursty, non-sinusoidal)
    ap.add_argument("--n-hot", type=int, default=2, help="how many clusters are hot during a burst")
    ap.add_argument("--hot-mass", type=float, default=0.8, help="fraction of probes drawn from hot clusters")
    ap.add_argument("--hot-skew", type=float, default=1.2,
                    help="Zipf-like skew across the hot clusters themselves (higher => one hot cluster dominates more)")
    ap.add_argument("--burst-mean", type=int, default=200, help="mean burst length (timesteps)")
    ap.add_argument("--burst-std", type=int, default=50, help="stddev of burst length (normal)")
    ap.add_argument("--burst-min", type=int, default=20, help="min burst length (timesteps)")
    ap.add_argument("--burst-max", type=int, default=800, help="max burst length (timesteps)")
    ap.add_argument("--alt-burst-prob", type=float, default=0.0,
                    help="probability a burst uses alternate hotset params")
    ap.add_argument("--alt-n-hot", type=int, default=None,
                    help="alternate hotset size (defaults to --n-hot)")
    ap.add_argument("--alt-hot-mass", type=float, default=None,
                    help="alternate hot-mass (defaults to --hot-mass)")
    ap.add_argument("--dist-switch-prob", type=float, default=0.05,
                    help="probability of switching query distribution mode each timestep")
    ap.add_argument("--dist-start-mode", type=str, default="zipf",
                    choices=["zipf", "uniform"], help="initial query distribution mode")
    ap.add_argument("--zipf-s", type=float, default=1.2,
                    help="zipf exponent used for cold distribution in zipf mode")

    # server + balancing
    ap.add_argument("--server-capacity", type=float, default=1e9)
    ap.add_argument("--churn-budget", type=float, default=5000.0)
    ap.add_argument("--migration-K", type=int, default=50)
    ap.add_argument("--gurobi-time-limit", type=float, default=10.0)
    ap.add_argument("--gurobi-mip-gap", type=float, default=0.01)

    # measurement
    ap.add_argument("--burn-in", type=int, default=200, help="early timesteps to treat as warmup")
    ap.add_argument("--fanout-var-window", type=int, default=200, help="rolling window for variance")
    ap.add_argument("--out-csv", type=str, default="timeseries.csv")
    ap.add_argument("--trace-jsonl", type=str, default="trace.jsonl",
                    help="write detailed trace (state/plan/server/slice) to JSONL; empty to disable")
    ap.add_argument("--trace-state-every", type=int, default=1,
                    help="snapshot full cluster/slice state every N steps in trace (1 = every step)")
    ap.add_argument("--reset-history-on-rebalance", action="store_true", default=True,
                    help="clear imbalance history after each rebalance for sharp drops")
    ap.add_argument("--no-reset-history-on-rebalance", action="store_false",
                    dest="reset_history_on_rebalance",
                    help="keep imbalance history across rebalances")

    ap.add_argument("--allow-repeat", action="store_true", default=True,
                    help="allow repeated probes so hotsets can dominate (80/20 even with small n_hot)")
    ap.add_argument("--no-allow-repeat", action="store_false", dest="allow_repeat",
                    help="force unique probes only")

    # requests per timestep (bursty load volume)
    ap.add_argument("--requests-per-server", type=int, default=10000,
                    help="default per-server request budget; if --requests-mean is omitted, mean becomes n_servers * this value")
    ap.add_argument("--requests-mean", type=int, default=None,
                    help="mean requests per timestep (default: n_servers * requests_per_server)")
    ap.add_argument("--requests-std", type=int, default=None,
                    help="stddev of requests per timestep (default: 20%% of requests_mean)")
    ap.add_argument("--requests-min", type=int, default=None,
                    help="min requests per timestep (default: max(1, requests_mean - 3*requests_std))")
    ap.add_argument("--requests-max", type=int, default=None,
                    help="max requests per timestep (default: requests_mean + 3*requests_std)")
    ap.add_argument("--lock-payload-per-step", action="store_true", default=True,
                    help="keep same hotset for all requests in a timestep")
    ap.add_argument("--no-lock-payload-per-step", action="store_false",
                    dest="lock_payload_per_step",
                    help="allow hotset to evolve within a timestep")
    ap.add_argument("--split-hot-slices", action="store_true", default=True,
                    help="split slices covering hot keys before rebalancing")
    ap.add_argument("--no-split-hot-slices", action="store_false",
                    dest="split_hot_slices",
                    help="disable hot-slice splitting")
    ap.add_argument("--aggregate-requests", action="store_true", default=True,
                    help="aggregate per-timestep requests into key counts for speed")
    ap.add_argument("--no-aggregate-requests", action="store_false",
                    dest="aggregate_requests",
                    help="simulate each request explicitly (slow)")
    ap.add_argument("--fanout-request-samples", type=int, default=128,
                    help="per-timestep request samples used to estimate per-request fan-out when aggregate mode is enabled")
    ap.add_argument("--include-state-in-csv", action="store_true", default=False,
                    help="embed full cluster/slice state JSON in each CSV row (very large files)")
    ap.add_argument("--print-current-tick", action="store_true", default=False,
                    help="print current simulation tick while running")
    ap.add_argument("--tick-print-every", type=int, default=1,
                    help="print every N ticks when --print-current-tick is enabled")

    # Step 1.5: add data to the most popular clusters
    ap.add_argument("--growth-on-rebalance", action="store_true", default=True,
                    help="add new shards for the most popular clusters before each rebalance")
    ap.add_argument("--no-growth-on-rebalance", action="store_false",
                    dest="growth_on_rebalance",
                    help="disable popular-cluster growth")
    ap.add_argument("--growth-popular-fraction", type=float, default=0.02,
                    help="fraction of highest-load clusters to receive new data")
    ap.add_argument("--growth-popular-k", type=int, default=0,
                    help="override popular-cluster count (0 => use fraction)")
    ap.add_argument("--growth-cold-fraction", type=float, dest="growth_popular_fraction",
                    help=argparse.SUPPRESS)
    ap.add_argument("--growth-cold-k", type=int, dest="growth_popular_k",
                    help=argparse.SUPPRESS)
    ap.add_argument("--growth-frac", type=float, default=0.05,
                    help="base fractional data increase per growth event, scaled by cluster popularity")
    ap.add_argument("--growth-add", type=float, default=0.0,
                    help="base absolute data increase per growth event, scaled by cluster popularity")
    ap.add_argument("--growth-new-shards-max", type=int, default=3,
                    help="max number of new shards added per grown cluster in one event")

    args = ap.parse_args()
    rng = random.Random(args.seed)

    if args.requests_mean is None:
        args.requests_mean = max(1, int(args.n_servers) * int(args.requests_per_server))
    if args.requests_std is None:
        args.requests_std = max(0, int(round(args.requests_mean * 0.2)))
    if args.requests_min is None:
        args.requests_min = max(1, int(args.requests_mean - 3 * args.requests_std))
    if args.requests_max is None:
        args.requests_max = int(args.requests_mean + 3 * args.requests_std)
    if args.requests_max < args.requests_min:
        args.requests_max = args.requests_min

    servers = [Server(sid=j, capacity=args.server_capacity) for j in range(args.n_servers)]

    # churn weights per key
    key_sizes = [1.0 + 0.2 * rng.random() for _ in range(args.n_keys)]

    # base per-list compute costs (kept simple; can be customized)
    list_cost = [1.0 for _ in range(args.n_keys)]

    # base weights used for zipf-mode cold distribution
    base_w = zipf_weights(args.n_keys, s=args.zipf_s, rng=rng)

    search = IVFSearch(
        rng=rng,
        n_lists=args.n_keys,
        nprobe=args.nprobe,
        list_cost=list_cost,
        list_weights=base_w,
        allow_repeat=args.allow_repeat,

    )
    #search.probe_jitter = 2

    workload = Workload(
        rng=rng,
        dist=BurstyHotsetQueryDist(
            n_lists=args.n_keys,
            n_hot=args.n_hot,
            hot_mass=args.hot_mass,
            burst_mean=args.burst_mean,
            burst_std=args.burst_std,
            burst_min=args.burst_min,
            burst_max=args.burst_max,
            alt_prob=args.alt_burst_prob,
            alt_n_hot=args.alt_n_hot,
            alt_hot_mass=args.alt_hot_mass,
            dist_switch_prob=args.dist_switch_prob,
            dist_start_mode=args.dist_start_mode,
            hot_skew=args.hot_skew,
        ),
    )

    state = build_initial_state(
        rng=rng,
        servers=servers,
        n_keys=args.n_keys,
        key_sizes=key_sizes,
        initial_slices=args.initial_slices,
        window=args.fanout_var_window,
        chunk_size=(args.chunk_size if args.chunk_size > 0 else None),
        cluster_shards_min=args.cluster_shards_min,
        cluster_shards_max=args.cluster_shards_max,
    )
    state.invariant_check(args.n_keys)

    server_model = SimpleHeatModel(alpha=0.0005, beta=0.01, decay=0.02)

    sim = Simulator(rng=rng, workload=workload, search=search, sharding_state=state,
                    server_model=server_model, n_keys=args.n_keys)

    rows = sim.run_timeseries(
        steps=args.steps,
        rebalance_every=args.rebalance_every,
        policy=args.policy,
        churn_budget=args.churn_budget,
        key_sizes=key_sizes,
        migration_K=args.migration_K,
        gurobi_time_limit_s=args.gurobi_time_limit,
        gurobi_mip_gap=args.gurobi_mip_gap,
        burn_in=args.burn_in,
        fanout_var_window=args.fanout_var_window,
        trace_jsonl=(args.trace_jsonl if args.trace_jsonl else None),
        trace_state_every=args.trace_state_every,
        reset_history_on_rebalance=args.reset_history_on_rebalance,
        requests_mean=args.requests_mean,
        requests_std=args.requests_std,
        requests_min=args.requests_min,
        requests_max=args.requests_max,
        lock_payload_per_step=args.lock_payload_per_step,
        split_hot_slices=args.split_hot_slices,
        aggregate_requests=args.aggregate_requests,
        fanout_request_samples=args.fanout_request_samples,
        growth_on_rebalance=args.growth_on_rebalance,
        growth_popular_fraction=args.growth_popular_fraction,
        growth_popular_k=args.growth_popular_k,
        growth_frac=args.growth_frac,
        growth_add=args.growth_add,
        growth_new_shards_max=args.growth_new_shards_max,
        include_state_in_rows=args.include_state_in_csv,
        print_current_tick=args.print_current_tick,
        tick_print_every=args.tick_print_every,
    )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # stable column order: base cols then per-server cols
    server_ids = sorted(state.servers.keys())
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

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    # quick console summary (post burn-in)
    last = rows[-1]
    print("=== DONE ===")
    print(f"out_csv={out_path}")
    if args.trace_jsonl:
        print(f"trace_jsonl={args.trace_jsonl}")
    print(f"policy={args.policy} steps={args.steps} rebalance_every={args.rebalance_every}")
    print(f"final_imbalance={last['imbalance']:.3f} hottest_server={last['hottest_server']}")


if __name__ == "__main__":
    main()
