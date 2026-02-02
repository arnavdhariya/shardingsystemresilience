# simulator.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import random

from server import ServerModel
from search import SearchAlgorithm, Query
from workload import Workload
from sharding import (
    ShardState,
    greedy_slicer_balance,
    ilp_rebalance_keys_gurobi,
    rebuild_as_single_key_slices,
)


@dataclass
class SimStats:
    avg_latency: float
    p95_latency: float
    imbalance: float
    hottest_server: int
    max_server_heat: float


class Simulator:
    def __init__(
        self,
        rng: random.Random,
        workload: Workload,
        search: SearchAlgorithm,
        sharding_state: ShardState,
        server_model: ServerModel,
        n_keys: int,
    ):
        self.rng = rng
        self.workload = workload
        self.search = search
        self.state = sharding_state
        self.server_model = server_model
        self.n_keys = n_keys
        self.latencies: List[float] = []

    def step(self) -> float:
        q = self.workload.next()
        plan = self.search.plan(q)
        routed = self.state.route_workunits(plan)

        total_latency = 0.0
        for sid, work in routed.items():
            total_latency += self.server_model.process(self.state.servers[sid], work)
            self.state.record_server_work(sid, work)

        for sid in self.state.servers:
            self.server_model.tick(self.state.servers[sid])

        self.latencies.append(total_latency)
        return total_latency

    def rebalance_greedy(self, churn_budget: float) -> None:
        greedy_slicer_balance(self.state, churn_budget=churn_budget)

    def rebalance_ilp_gurobi(
        self,
        key_sizes: List[float],
        migration_K: int,
        time_limit_s: float = 10.0,
        mip_gap: float = 0.01,
        sample_queries: int = 500,
        verbose: bool = False,
    ) -> None:
        """
        ILP is key-level. We:
          1) read current key->server allocation from current slices,
          2) estimate per-key freq_i using sampled query plans (search-driven),
          3) solve ILP in Gurobi,
          4) rebuild state as 1-key slices (simple, consistent).
        """
        server_ids = sorted(self.state.servers.keys())
        sid_to_j = {sid: j for j, sid in enumerate(server_ids)}
        j_to_sid = {j: sid for sid, j in sid_to_j.items()}
        n_servers = len(server_ids)

        # current alloc per key (as server index 0..n-1)
        cur_alloc_j = [0] * self.n_keys
        for k in range(self.n_keys):
            sl = self.state.key_to_slice[k]
            sid = self.state.slice_to_server[sl]
            cur_alloc_j[k] = sid_to_j[sid]

        R_j = [self.state.servers[j_to_sid[j]].capacity for j in range(n_servers)]

        # freq_i: search-driven estimate
        freq_i = [0.0] * self.n_keys
        for _ in range(sample_queries):
            q = Query(qid=-1, payload=(self.rng.random(),))
            for wu in self.search.plan(q):
                freq_i[wu.obj_id] += wu.cost

        new_alloc_j = ilp_rebalance_keys_gurobi(
            n_keys=self.n_keys,
            n_servers=n_servers,
            r_i=key_sizes,
            R_j=R_j,
            current_alloc=cur_alloc_j,
            freq_i=freq_i,
            K=migration_K,
            time_limit_s=time_limit_s,
            mip_gap=mip_gap,
            verbose=verbose,
        )
        new_alloc_sid = [j_to_sid[j] for j in new_alloc_j]
        rebuild_as_single_key_slices(self.state, self.n_keys, key_sizes, new_alloc_sid)

    def run(
        self,
        steps: int,
        rebalance_every: int,
        policy: str,
        *,
        churn_budget: float,
        key_sizes: List[float],
        migration_K: int,
        gurobi_time_limit_s: float = 10.0,
        gurobi_mip_gap: float = 0.01,
    ) -> SimStats:
        if rebalance_every <= 0:
            rebalance_every = steps + 1

        for t in range(1, steps + 1):
            self.step()
            if t % rebalance_every == 0:
                if policy == "greedy":
                    self.rebalance_greedy(churn_budget=churn_budget)
                elif policy == "ilp":
                    self.rebalance_ilp_gurobi(
                        key_sizes=key_sizes,
                        migration_K=migration_K,
                        time_limit_s=gurobi_time_limit_s,
                        mip_gap=gurobi_mip_gap,
                    )
                else:
                    raise ValueError(f"Unknown policy: {policy}")

        return self.summarize()

    def summarize(self) -> SimStats:
        if not self.latencies:
            return SimStats(0.0, 0.0, 0.0, 0, 0.0)

        xs = sorted(self.latencies)
        avg = sum(xs) / len(xs)
        p95 = xs[int(0.95 * (len(xs) - 1))]

        imb = self.state.imbalance()
        hot = self.state.hottest_server()
        max_heat = max(self.state.servers[sid].heat for sid in self.state.servers)

        return SimStats(avg_latency=avg, p95_latency=p95, imbalance=imb, hottest_server=hot, max_server_heat=max_heat)
