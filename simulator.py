# simulator.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import random

from server import ServerModel
from search import SearchAlgorithm, Query, WorkUnit
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


def _diff_state(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    before_slices = {s["slice_id"]: s for s in before.get("slices", [])}
    after_slices = {s["slice_id"]: s for s in after.get("slices", [])}
    before_ids = set(before_slices.keys())
    after_ids = set(after_slices.keys())

    added = [after_slices[sid] for sid in sorted(after_ids - before_ids)]
    removed = [before_slices[sid] for sid in sorted(before_ids - after_ids)]

    moved = []
    for sid in sorted(before_ids & after_ids):
        if before_slices[sid].get("server") != after_slices[sid].get("server"):
            moved.append({
                "slice_id": sid,
                "from": before_slices[sid].get("server"),
                "to": after_slices[sid].get("server"),
            })

    key_moves = []
    before_keys = before.get("key_to_server", [])
    after_keys = after.get("key_to_server", [])
    for k, (b, a) in enumerate(zip(before_keys, after_keys)):
        if b != a:
            key_moves.append({"key": k, "from": b, "to": a})

    cluster_transitions = []
    before_cluster_slices = before.get("cluster_to_slices", [])
    after_cluster_slices = after.get("cluster_to_slices", [])
    before_cluster_servers = before.get("cluster_to_servers", [])
    after_cluster_servers = after.get("cluster_to_servers", [])
    before_cluster_sizes = before.get("cluster_shard_sizes", [])
    after_cluster_sizes = after.get("cluster_shard_sizes", [])
    n_clusters = min(
        len(before_cluster_slices),
        len(after_cluster_slices),
        len(before_cluster_servers),
        len(after_cluster_servers),
        len(before_cluster_sizes),
        len(after_cluster_sizes),
    )
    for k in range(n_clusters):
        if (
            before_cluster_slices[k] != after_cluster_slices[k]
            or before_cluster_servers[k] != after_cluster_servers[k]
            or before_cluster_sizes[k] != after_cluster_sizes[k]
        ):
            cluster_transitions.append({
                "cluster": k,
                "slices_before": before_cluster_slices[k],
                "slices_after": after_cluster_slices[k],
                "servers_before": before_cluster_servers[k],
                "servers_after": after_cluster_servers[k],
                "shard_sizes_before": before_cluster_sizes[k],
                "shard_sizes_after": after_cluster_sizes[k],
            })

    return {
        "slice_added": added,
        "slice_removed": removed,
        "slice_moved": moved,
        "key_migrations": key_moves,
        "cluster_transitions": cluster_transitions,
    }


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

    def rebalance_greedy(
        self,
        churn_budget: float,
        *,
        action_log: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        greedy_slicer_balance(self.state, churn_budget=churn_budget, action_log=action_log)

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


    def run_timeseries(
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
        burn_in: int = 0,
        fanout_var_window: int = 200,
        trace_jsonl: Optional[str] = None,
        trace_state_every: int = 1,
        reset_history_on_rebalance: bool = False,
        requests_mean: int = 1,
        requests_std: int = 0,
        requests_min: int = 1,
        requests_max: Optional[int] = None,
        lock_payload_per_step: bool = True,
        split_hot_slices: bool = True,
        aggregate_requests: bool = True,
        fanout_request_samples: int = 128,
        growth_on_rebalance: bool = True,
        growth_popular_fraction: float = 0.02,
        growth_popular_k: int = 0,
        growth_frac: float = 0.05,
        growth_add: float = 0.0,
        growth_new_shards_max: int = 3,
        include_state_in_rows: bool = False,
        print_current_tick: bool = False,
        tick_print_every: int = 1,
    ) -> List[dict]:
        """Run ONE simulation and return per-timestep metrics (for CSV export).

        Columns include:
          - t
          - latency
          - fanout (distinct servers touched in this timestep; avg per-request if simulated individually)
          - fanout_req_* (per-request fan-out summary for this timestep; sampled if aggregate mode)
          - fanout_var (rolling variance over last `fanout_var_window` timesteps, after burn-in)
          - imbalance (post-rebalance Slicer: max(util)/mean(util))
          - imbalance_pre (pre-rebalance)
          - hottest_server
          - rebalance (0/1)
          - n_requests (requests per timestep)
          - burst metadata (burst_id, burst_start, burst_mode, hot_mass, hot_probe_ratio, hot_set)
          - per-server instantaneous work: work_s<sid>
          - per-server keyspace size: keyspace_s<sid>
          - rebalance/growth action payloads and transition diffs (JSON)
          - optional full state snapshots in each row when `include_state_in_rows=True`
        """
        if rebalance_every <= 0:
            rebalance_every = steps + 1

        fanouts: List[float] = []
        rows: List[dict] = []

        server_ids = sorted(self.state.servers.keys())
        trace_f = None

        def emit(event: Dict[str, Any]) -> None:
            if trace_f is None:
                return
            trace_f.write(json.dumps(event))
            trace_f.write("\n")

        try:
            if trace_jsonl:
                trace_f = open(trace_jsonl, "w", encoding="utf-8")
                emit({
                    "type": "state",
                    "t": 0,
                    "reason": "init",
                    "state": self.state.snapshot(self.n_keys),
                })

            for t in range(1, steps + 1):
                if print_current_tick:
                    every = max(1, int(tick_print_every))
                    if t == 1 or t == steps or (t % every == 0):
                        print(f"[tick] {t}/{steps}", flush=True)

                if requests_std > 0:
                    n_requests = int(round(self.rng.gauss(requests_mean, requests_std)))
                else:
                    n_requests = int(round(requests_mean))
                n_requests = max(requests_min, n_requests)
                if requests_max is not None:
                    n_requests = min(n_requests, requests_max)

                queries = self.workload.next_batch(n_requests, lock_payload=lock_payload_per_step)
                q0 = queries[0]
                meta: Dict[str, Any] = {}
                if q0.payload and isinstance(q0.payload[0], dict):
                    meta = q0.payload[0]

                hot_list = [int(x) for x in meta.get("hot", []) if isinstance(x, (int, float))]
                hot_weights = []
                if isinstance(meta.get("hot_weights"), list):
                    for v in meta.get("hot_weights", []):
                        try:
                            hot_weights.append(float(v))
                        except (TypeError, ValueError):
                            hot_weights.append(0.0)
                hot_set = set(hot_list)
                hot_mass = float(meta.get("hot_mass", 0.0)) if meta else 0.0
                burst_id = int(meta.get("burst_id", 0)) if meta else 0
                burst_start = int(meta.get("burst_start", 0)) if meta else 0
                burst_mode = str(meta.get("burst_mode", "")) if meta else ""
                burst_len = int(meta.get("burst_len", 0)) if meta else 0
                burst_remaining = int(meta.get("burst_remaining", 0)) if meta else 0
                dist_mode = str(meta.get("dist_mode", "zipf")) if meta else "zipf"
                dist_switch = int(meta.get("dist_switch", 0)) if meta else 0
                dist_id = int(meta.get("dist_id", 0)) if meta else 0

                use_aggregate = (
                    aggregate_requests
                    and hasattr(self.search, "plan_counts")
                    and n_requests > 1
                )

                # execute + record (batch)
                total_latency = 0.0
                work_this_step = {sid: 0.0 for sid in server_ids}
                total_fanout = 0
                total_probes = 0
                hot_hits = 0
                plan_counts: Dict[int, int] = {}
                request_fanouts: List[float] = []
                fanout_server_cache: Dict[int, set[int]] = {}

                def per_request_fanout_from_plan(plan: List[WorkUnit]) -> float:
                    touched: set[int] = set()
                    for wu in plan:
                        key_id = int(wu.obj_id)
                        servers_for_key = fanout_server_cache.get(key_id)
                        if servers_for_key is None:
                            servers_for_key = {
                                int(self.state.slice_to_server[slid])
                                for slid in self.state.key_to_slices.get(key_id, [])
                            }
                            fanout_server_cache[key_id] = servers_for_key
                        touched.update(servers_for_key)
                    return float(len(touched))

                if use_aggregate:
                    sample_n = max(1, min(int(n_requests), int(max(1, fanout_request_samples))))
                    if sample_n > 0 and hasattr(self.search, "plan"):
                        search_rng = getattr(self.search, "rng", None)
                        saved_state = None
                        if search_rng is not None and hasattr(search_rng, "getstate"):
                            saved_state = search_rng.getstate()
                        try:
                            for _ in range(sample_n):
                                sample_plan = self.search.plan(q0)
                                request_fanouts.append(per_request_fanout_from_plan(sample_plan))
                        finally:
                            if saved_state is not None and hasattr(search_rng, "setstate"):
                                search_rng.setstate(saved_state)

                    counts = self.search.plan_counts(n_requests, meta)
                    plan_counts = dict(counts)
                    total_probes = sum(counts.values())
                    if hot_set:
                        hot_hits = sum(counts.get(k, 0) for k in hot_set)

                    list_cost = getattr(self.search, "list_cost", None)
                    work_units: List[WorkUnit] = []
                    for k, cnt in counts.items():
                        if cnt <= 0:
                            continue
                        cost = (list_cost[k] if list_cost is not None else 1.0) * cnt
                        work_units.append(WorkUnit(obj_id=k, cost=cost))

                    routed = self.state.route_workunits(work_units)
                    total_fanout = len(routed)
                    for sid, work in routed.items():
                        total_latency += self.server_model.process(self.state.servers[sid], work)
                        work_this_step[sid] = work
                else:
                    for q in queries:
                        plan = self.search.plan(q)
                        total_probes += len(plan)
                        if hot_set:
                            hot_hits += sum(1 for wu in plan if wu.obj_id in hot_set)
                        for wu in plan:
                            plan_counts[wu.obj_id] = plan_counts.get(wu.obj_id, 0) + 1

                        routed = self.state.route_workunits(plan)
                        total_fanout += len(routed)
                        request_fanouts.append(per_request_fanout_from_plan(plan))
                        for sid, work in routed.items():
                            total_latency += self.server_model.process(self.state.servers[sid], work)
                            work_this_step[sid] += work

                for sid in server_ids:
                    self.state.record_server_work(sid, work_this_step[sid])
                total_work = sum(work_this_step.values())

                for sid in server_ids:
                    self.server_model.tick(self.state.servers[sid])

                self.latencies.append(total_latency)

                if use_aggregate:
                    fanout = float(total_fanout)
                else:
                    fanout = (total_fanout / n_requests) if n_requests > 0 else 0.0
                fanouts.append(fanout)

                fanout_req_samples = len(request_fanouts)
                if request_fanouts:
                    fanout_req_mean = sum(request_fanouts) / fanout_req_samples
                    fanout_req_min = min(request_fanouts)
                    fanout_req_max = max(request_fanouts)
                    fanout_req_var = (
                        sum((x - fanout_req_mean) ** 2 for x in request_fanouts) / fanout_req_samples
                    )
                else:
                    fanout_req_mean = fanout
                    fanout_req_min = fanout
                    fanout_req_max = fanout
                    fanout_req_var = 0.0

                hot_probe_ratio = (hot_hits / total_probes) if total_probes > 0 else 0.0

                imbalance_pre = self.state.imbalance()

                slice_loads: Dict[int, float] = {slid: 0.0 for slid in self.state.slices.keys()}
                cluster_loads: Dict[int, float] = {}
                list_cost = getattr(self.search, "list_cost", None)
                for k, cnt in plan_counts.items():
                    if cnt <= 0:
                        continue
                    cost = (list_cost[k] if list_cost is not None else 1.0) * cnt
                    cluster_loads[k] = float(cost)
                    for sl, part in self.state.distribute_key_cost_to_slices(k, cost).items():
                        slice_loads[sl] = slice_loads.get(sl, 0.0) + float(part)

                self.state.update_slice_loads(slice_loads)

                hot_slice_id = None
                cold_slice_id = None
                hot_slice_load = 0.0
                cold_slice_load = 0.0
                if slice_loads:
                    hot_slice_id = max(slice_loads, key=slice_loads.get)
                    cold_slice_id = min(slice_loads, key=slice_loads.get)
                    hot_slice_load = slice_loads[hot_slice_id]
                    cold_slice_load = slice_loads[cold_slice_id]

                state_for_step = None
                if trace_f is not None and trace_state_every > 0 and t % trace_state_every == 0:
                    state_for_step = self.state.snapshot(self.n_keys)

                if trace_f is not None:
                    event = {
                        "type": "step",
                        "t": t,
                        "qid": q0.qid,
                        "n_requests": n_requests,
                        "burst_id": burst_id,
                        "burst_start": burst_start,
                        "burst_mode": burst_mode,
                        "burst_len": burst_len,
                        "burst_remaining": burst_remaining,
                        "dist_mode": dist_mode,
                        "dist_switch": dist_switch,
                        "dist_id": dist_id,
                        "hot_mass": hot_mass,
                        "hot_set": hot_list,
                        "hot_weights": hot_weights,
                        "hot_probe_ratio": hot_probe_ratio,
                        "plan_counts": plan_counts,
                        "total_probes": total_probes,
                        "cluster_loads": cluster_loads,
                        "slice_loads": slice_loads,
                        "fanout": fanout,
                        "fanout_req_mean": fanout_req_mean,
                        "fanout_req_var": fanout_req_var,
                        "fanout_req_min": fanout_req_min,
                        "fanout_req_max": fanout_req_max,
                        "fanout_req_samples": fanout_req_samples,
                        "imbalance_pre": imbalance_pre,
                        "server_work": work_this_step,
                        "server_heat": {sid: self.state.servers[sid].heat for sid in server_ids},
                    }
                    if state_for_step is not None:
                        event["state"] = state_for_step
                    emit(event)

                did_rebalance = 0
                growth_event = 0
                growth_added_total = 0.0
                growth_slices = 0
                growth_servers: List[int] = []
                growth_clusters: List[int] = []
                growth_actions: List[Dict[str, Any]] = []
                rebalance_actions: List[Dict[str, Any]] = []
                rebalance_diff: Dict[str, Any] = {}
                rebalance_key_migrations: List[Dict[str, Any]] = []
                rebalance_cluster_transitions: List[Dict[str, Any]] = []
                rebalance_slice_added: List[Dict[str, Any]] = []
                rebalance_slice_removed: List[Dict[str, Any]] = []
                rebalance_slice_moved: List[Dict[str, Any]] = []

                if t % rebalance_every == 0:
                    did_rebalance = 1
                    state_before = state_for_step or self.state.snapshot(self.n_keys)

                    # Step 1.5: add new shards for the most popular clusters.
                    if growth_on_rebalance and server_ids:
                        ranked_clusters = sorted(
                            ((int(k), float(v)) for k, v in cluster_loads.items() if float(v) > 0.0),
                            key=lambda kv: (-kv[1], kv[0]),
                        )
                        if growth_popular_k and growth_popular_k > 0:
                            k = min(len(ranked_clusters), int(growth_popular_k))
                        else:
                            k = max(1, int(round(len(ranked_clusters) * max(0.0, growth_popular_fraction))))
                            k = min(len(ranked_clusters), k)
                        growth_clusters = [cluster for cluster, _load in ranked_clusters[:k]]
                        if growth_clusters and (growth_frac > 0.0 or growth_add > 0.0):
                            growth_actions, growth_added_total = self.state.add_popular_data_round_robin(
                                growth_clusters,
                                cluster_loads,
                                growth_frac=growth_frac,
                                growth_add=growth_add,
                                max_new_shards_per_cluster=growth_new_shards_max,
                            )
                            growth_slices = len(growth_actions)
                            growth_event = 1 if growth_actions else 0
                            growth_servers = sorted({int(act.get("server", -1)) for act in growth_actions if "server" in act})

                            # update key_sizes for consistency
                            for act in growth_actions:
                                lo = int(act.get("lo", 0))
                                hi = int(act.get("hi", 0))
                                delta = float(act.get("delta", 0.0))
                                if delta > 0 and hi > lo:
                                    per_key = delta / (hi - lo)
                                    for kidx in range(lo, hi):
                                        key_sizes[kidx] += per_key

                            if trace_f is not None and state_before is not None:
                                emit({
                                    "type": "growth",
                                    "t": t,
                                    "mode": "popular_round_robin",
                                    "clusters": growth_clusters,
                                    "servers": growth_servers,
                                    "actions": growth_actions,
                                    "total_added": growth_added_total,
                                })

                    actions: List[Dict[str, Any]] = []

                    if split_hot_slices and hot_list:
                        actions.extend(self.state.split_hot_slices(hot_list))

                    if policy == "greedy":
                        self.rebalance_greedy(churn_budget=churn_budget, action_log=actions)
                    elif policy == "ilp":
                        self.rebalance_ilp_gurobi(
                            key_sizes=key_sizes,
                            migration_K=migration_K,
                            time_limit_s=gurobi_time_limit_s,
                            mip_gap=gurobi_mip_gap,
                        )
                    else:
                        raise ValueError(f"Unknown policy: {policy}")

                    state_after = self.state.snapshot(self.n_keys)
                    diff = _diff_state(state_before, state_after)
                    rebalance_actions = list(actions)
                    rebalance_diff = diff
                    rebalance_key_migrations = diff.get("key_migrations", [])
                    rebalance_cluster_transitions = diff.get("cluster_transitions", [])
                    rebalance_slice_added = diff.get("slice_added", [])
                    rebalance_slice_removed = diff.get("slice_removed", [])
                    rebalance_slice_moved = diff.get("slice_moved", [])

                    if trace_f is not None:
                        emit({
                            "type": "rebalance",
                            "t": t,
                            "policy": policy,
                            "actions": actions,
                            "state_before": state_before,
                            "state_after": state_after,
                            "diff": diff,
                            "cluster_migrations": diff.get("key_migrations", []),
                            "cluster_transitions": diff.get("cluster_transitions", []),
                        })
                    if reset_history_on_rebalance:
                        self.state.reset_work_history()

                # rolling fanout variance (population variance) after burn-in
                fanout_var = 0.0
                if t > burn_in and len(fanouts) >= 2:
                    w = fanouts[max(0, len(fanouts) - fanout_var_window):]
                    mu = sum(w) / len(w)
                    fanout_var = sum((x - mu) ** 2 for x in w) / len(w)

                keyspace_this_step = {sid: 0.0 for sid in server_ids}
                slice_count_this_step = {sid: 0 for sid in server_ids}
                for sl in self.state.slices.values():
                    sid = self.state.slice_to_server[sl.sid]
                    keyspace_this_step[sid] += sl.size
                    slice_count_this_step[sid] += 1

                server_heat_map = {sid: float(self.state.servers[sid].heat) for sid in server_ids}
                server_util_map = {int(sid): float(v) for sid, v in self.state._all_current_utils().items()}

                mean_work = (total_work / len(server_ids)) if server_ids else 0.0
                hottest_work_server = max(work_this_step, key=work_this_step.get) if server_ids else -1
                coldest_work_server = min(work_this_step, key=work_this_step.get) if server_ids else -1
                hottest_work = work_this_step.get(hottest_work_server, 0.0)
                coldest_work = work_this_step.get(coldest_work_server, 0.0)

                row = {
                    "t": t,
                    "latency": total_latency,
                    "fanout": fanout,
                    "fanout_req_mean": fanout_req_mean,
                    "fanout_req_var": fanout_req_var,
                    "fanout_req_min": fanout_req_min,
                    "fanout_req_max": fanout_req_max,
                    "fanout_req_samples": fanout_req_samples,
                    "fanout_var": fanout_var,
                    "imbalance": self.state.imbalance(),
                    "imbalance_pre": imbalance_pre,
                    "hottest_server": self.state.hottest_server(),
                    "coldest_server": self.state.coldest_server(),
                    "hottest_work_server": hottest_work_server,
                    "coldest_work_server": coldest_work_server,
                    "hottest_work": hottest_work,
                    "coldest_work": coldest_work,
                    "mean_work": mean_work,
                    "rebalance": did_rebalance,
                    "growth_event": growth_event,
                    "growth_added_total": growth_added_total,
                    "growth_slices": growth_slices,
                    "growth_servers": json.dumps(growth_servers),
                    "growth_clusters": json.dumps(growth_clusters),
                    "growth_actions": json.dumps(growth_actions),
                    "burn_in": int(t <= burn_in),
                    "n_requests": n_requests,
                    "work_units": total_probes,
                    "workload": total_work,
                    "dist_mode": dist_mode,
                    "dist_switch": dist_switch,
                    "dist_id": dist_id,
                    "burst_id": burst_id,
                    "burst_start": burst_start,
                    "burst_mode": burst_mode,
                    "burst_len": burst_len,
                    "burst_remaining": burst_remaining,
                    "hot_mass": hot_mass,
                    "hot_probe_ratio": hot_probe_ratio,
                    "hot_set": json.dumps(hot_list),
                    "hot_weights": json.dumps(hot_weights),
                    "cluster_loads": json.dumps(cluster_loads),
                    "hottest_slice": hot_slice_id,
                    "coldest_slice": cold_slice_id,
                    "hottest_slice_load": hot_slice_load,
                    "coldest_slice_load": cold_slice_load,
                    "slice_loads": json.dumps(slice_loads),
                    "n_slices_total": len(self.state.slices),
                    "n_clusters_total": self.n_keys,
                    "server_work_map": json.dumps(work_this_step),
                    "server_heat_map": json.dumps(server_heat_map),
                    "server_util_map": json.dumps(server_util_map),
                    "server_keyspace_map": json.dumps(keyspace_this_step),
                    "server_slice_counts": json.dumps(slice_count_this_step),
                    "rebalance_actions": json.dumps(rebalance_actions),
                    "rebalance_diff": json.dumps(rebalance_diff),
                    "rebalance_key_migrations": json.dumps(rebalance_key_migrations),
                    "rebalance_cluster_transitions": json.dumps(rebalance_cluster_transitions),
                    "rebalance_slice_added": json.dumps(rebalance_slice_added),
                    "rebalance_slice_removed": json.dumps(rebalance_slice_removed),
                    "rebalance_slice_moved": json.dumps(rebalance_slice_moved),
                    "rebalance_key_migration_count": len(rebalance_key_migrations),
                    "rebalance_cluster_transition_count": len(rebalance_cluster_transitions),
                    "rebalance_slice_added_count": len(rebalance_slice_added),
                    "rebalance_slice_removed_count": len(rebalance_slice_removed),
                    "rebalance_slice_moved_count": len(rebalance_slice_moved),
                }
                if include_state_in_rows:
                    state_snapshot = self.state.snapshot(self.n_keys)
                    row["state_key_to_slice"] = json.dumps(state_snapshot.get("key_to_slice", []))
                    row["state_key_to_server"] = json.dumps(state_snapshot.get("key_to_server", []))
                    row["state_cluster_to_slices"] = json.dumps(state_snapshot.get("cluster_to_slices", []))
                    row["state_cluster_to_servers"] = json.dumps(state_snapshot.get("cluster_to_servers", []))
                    row["state_cluster_shard_sizes"] = json.dumps(state_snapshot.get("cluster_shard_sizes", []))
                    row["state_slices"] = json.dumps(state_snapshot.get("slices", []))
                for sid in server_ids:
                    row[f"work_s{sid}"] = work_this_step[sid]
                    row[f"keyspace_s{sid}"] = keyspace_this_step[sid]
                    row[f"heat_s{sid}"] = server_heat_map[sid]
                    row[f"util_s{sid}"] = server_util_map.get(sid, 0.0)
                    row[f"slicecount_s{sid}"] = slice_count_this_step[sid]
                rows.append(row)

            return rows
        finally:
            if trace_f is not None:
                trace_f.close()

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
