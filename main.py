# main.py
import argparse
import random

from server import Server, SimpleHeatModel
from search import IVFSearch
from workload import Workload, RandomVectorQueryDist, zipf_weights
from sharding import build_initial_state
from simulator import Simulator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--policy", choices=["greedy", "ilp"], default="greedy")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--rebalance-every", type=int, default=50)

    ap.add_argument("--n-servers", type=int, default=8)
    ap.add_argument("--n-keys", type=int, default=2000)
    ap.add_argument("--nprobe", type=int, default=20)
    ap.add_argument("--zipf-s", type=float, default=1.1)

    ap.add_argument("--server-capacity", type=float, default=1e9)
    ap.add_argument("--initial-slices", type=int, default=64)

    ap.add_argument("--churn-budget", type=float, default=5000.0)
    ap.add_argument("--migration-K", type=int, default=50)
    ap.add_argument("--gurobi-time-limit", type=float, default=10.0)
    ap.add_argument("--gurobi-mip-gap", type=float, default=0.01)

    args = ap.parse_args()
    rng = random.Random(args.seed)

    servers = [Server(sid=j, capacity=args.server_capacity) for j in range(args.n_servers)]

    # key sizes = capacity/churn resource r_i
    key_sizes = [1.0 + 0.2 * rng.random() for _ in range(args.n_keys)]

    # hotspot skew across IVF lists
    w = zipf_weights(args.n_keys, s=args.zipf_s, rng=rng)

    # per-list compute cost (can correlate with skew)
    list_cost = [1.0 + 3.0 * w[i] * args.n_keys for i in range(args.n_keys)]

    search = IVFSearch(rng=rng, n_lists=args.n_keys, nprobe=args.nprobe, list_cost=list_cost, list_weights=w)
    workload = Workload(rng=rng, dist=RandomVectorQueryDist(dim=4))
    state = build_initial_state(rng=rng, servers=servers, n_keys=args.n_keys,
                                key_sizes=key_sizes, initial_slices=args.initial_slices, window=200)
    state.invariant_check(args.n_keys)

    server_model = SimpleHeatModel(alpha=0.0005, beta=0.01, decay=0.02)

    sim = Simulator(rng=rng, workload=workload, search=search, sharding_state=state,
                    server_model=server_model, n_keys=args.n_keys)

    stats = sim.run(
        steps=args.steps,
        rebalance_every=args.rebalance_every,
        policy=args.policy,
        churn_budget=args.churn_budget,
        key_sizes=key_sizes,
        migration_K=args.migration_K,
        gurobi_time_limit_s=args.gurobi_time_limit,
        gurobi_mip_gap=args.gurobi_mip_gap,
    )

    print("=== DONE ===")
    print(f"policy={args.policy} steps={args.steps} rebalance_every={args.rebalance_every}")
    print(f"avg_latency={stats.avg_latency:.3f} p95_latency={stats.p95_latency:.3f}")
    print(f"imbalance={stats.imbalance:.3f} hottest_server={stats.hottest_server}")
    print(f"max_server_heat={stats.max_server_heat:.3f}")


if __name__ == "__main__":
    main()
