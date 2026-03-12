
# sharding.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import random

from server import Server
from search import WorkUnit


# -----------------------------
# Slices (Slicer-style contiguous key ranges)
# -----------------------------

@dataclass
class Slice:
    sid: int
    lo: int
    hi: int
    size: float  # churn weight (bytes / keyspace)
    load: float = 0.0  # recent load (set each timestep)
#also

class ShardState:
    """
    Maintains slice placement, routes WorkUnits (key ids) to servers,
    and supports reversible ops: reassign/split/merge/undo.
    """
    def __init__(
        self,
        servers: List[Server],
        slices: List[Slice],
        slice_to_server: Dict[int, int],
        window: int = 200,
        rng: Optional[random.Random] = None,
    ):
        self.servers: Dict[int, Server] = {sv.sid: sv for sv in servers}
        self.slices: Dict[int, Slice] = {sl.sid: sl for sl in slices}
        self.slice_to_server: Dict[int, int] = dict(slice_to_server)
        self._rng = rng or random.Random(0)

        self._op_stack: List[Tuple[str, Any]] = []
        self._window = max(1, window)
        self._work_history: Dict[int, List[float]] = {sid: [] for sid in self.servers}
        self._rr_server_cursor = 0

        self._rebuild_keymap()

    # ---- interface used by greedy_slicer_balance ----

    def slices_on(self, server_id: int) -> List[int]:
        return [slid for slid, sid in self.slice_to_server.items() if sid == server_id]

    def adjacent_cold_pairs(self, server_id: int) -> List[Tuple[int, int]]:
        sl_ids = self.slices_on(server_id)
        sls = sorted((self.slices[i] for i in sl_ids), key=lambda s: (s.lo, s.hi))
        pairs: List[Tuple[int, int]] = []
        for a, b in zip(sls, sls[1:]):
            if (a.lo == b.lo and a.hi == b.hi) or a.hi == b.lo:
                pairs.append((a.sid, b.sid))
        return pairs

    def keyspace(self, slice_id: int) -> float:
        return float(self.slices[slice_id].size)

    def imbalance(self) -> float:
        """Load imbalance per Slicer: max(load)/mean(load).
        Prefer current slice-load estimates (recomputed each timestep), and fall back
        to recent observed server work if slice loads are unavailable.
        """
        utils_map = self._all_current_utils()
        utils = [utils_map[sid] for sid in self.servers]
        if not utils:
            return 0.0
        m = sum(utils) / len(utils)
        if m <= 1e-12:
            return 0.0
        return max(utils) / m

    def hottest_server(self) -> int:
        utils = self._all_current_utils()
        return max(self.servers, key=lambda sid: utils[sid])

    def coldest_server(self) -> int:
        utils = self._all_current_utils()
        return min(self.servers, key=lambda sid: utils[sid])

    def apply_reassign(self, slice_id: int, dst_server: int) -> None:
        prev = self.slice_to_server[slice_id]
        self._op_stack.append(("reassign", (slice_id, prev)))
        self.slice_to_server[slice_id] = dst_server
        self._rebuild_keymap()

    def apply_split(self, slice_id: int) -> None:
        sl = self.slices[slice_id]
        new_id1 = self._new_slice_id()
        new_id2 = self._new_slice_id(new_id1 + 1)

        if sl.hi - sl.lo <= 1:
            if sl.size <= 1e-12:
                self._op_stack.append(("noop", None))
                return
            # Slicer-like shard splits do not need to be equal; keep deterministic randomness.
            ratio = 0.3 + 0.4 * self._rng.random()
            size1 = sl.size * ratio
            size2 = sl.size - size1
            s1 = Slice(sid=new_id1, lo=sl.lo, hi=sl.hi, size=size1)
            s2 = Slice(sid=new_id2, lo=sl.lo, hi=sl.hi, size=size2)
        else:
            mid = (sl.lo + sl.hi) // 2
            if mid == sl.lo or mid == sl.hi:
                self._op_stack.append(("noop", None))
                return
            size1 = sl.size / 2.0
            size2 = sl.size - size1
            s1 = Slice(sid=new_id1, lo=sl.lo, hi=mid, size=size1)
            s2 = Slice(sid=new_id2, lo=mid, hi=sl.hi, size=size2)
        # Preserve load estimate across split so greedy moves can evaluate benefit immediately.
        total_sz = max(1e-12, size1 + size2)
        s1.load = sl.load * (size1 / total_sz)
        s2.load = sl.load - s1.load
        server = self.slice_to_server[slice_id]

        self._op_stack.append(("split", (slice_id, sl, (new_id1, new_id2), server)))

        del self.slices[slice_id]
        del self.slice_to_server[slice_id]
        self.slices[new_id1] = s1
        self.slices[new_id2] = s2
        self.slice_to_server[new_id1] = server
        self.slice_to_server[new_id2] = server
        self._rebuild_keymap()

    def instantaneous_imbalance_from_tick(self, tick_work: Dict[int, float]) -> float:
        """
        Compute imbalance using only the current tick's work
        (no rolling window).
        """
        utils = []
        for sid, server in self.servers.items():
            cap = float(server.capacity)
            load = tick_work.get(sid, 0.0)
            if cap <= 1e-12:
                utils.append(0.0)
            else:
                utils.append(load / cap)

        if not utils:
            return 0.0

        m = sum(utils) / len(utils)
        if m <= 1e-12:
            return 0.0

        return max(utils) / m

    def apply_merge(self, slice_a: int, slice_b: int) -> None:
        if slice_a not in self.slices or slice_b not in self.slices:
            self._op_stack.append(("noop", None))
            return

        sa = self.slices[slice_a]
        sb = self.slices[slice_b]
        server_a = self.slice_to_server[slice_a]
        server_b = self.slice_to_server[slice_b]
        if server_a != server_b:
            self._op_stack.append(("noop", None))
            return

        same_cluster_shards = (sa.lo == sb.lo and sa.hi == sb.hi)
        if not (same_cluster_shards or sa.hi == sb.lo or sb.hi == sa.lo):
            self._op_stack.append(("noop", None))
            return

        if same_cluster_shards:
            lo = sa.lo
            hi = sa.hi
        else:
            lo = min(sa.lo, sb.lo)
            hi = max(sa.hi, sb.hi)
        merged_size = sa.size + sb.size
        new_id = self._new_slice_id()
        merged = Slice(sid=new_id, lo=lo, hi=hi, size=merged_size, load=sa.load + sb.load)

        self._op_stack.append(("merge", (new_id, merged, (slice_a, sa), (slice_b, sb), server_a)))

        del self.slices[slice_a]
        del self.slices[slice_b]
        del self.slice_to_server[slice_a]
        del self.slice_to_server[slice_b]
        self.slices[new_id] = merged
        self.slice_to_server[new_id] = server_a
        self._rebuild_keymap()

    def undo_last(self) -> None:
        if not self._op_stack:
            return
        kind, payload = self._op_stack.pop()

        if kind == "noop":
            return

        if kind == "reassign":
            slice_id, prev_server = payload
            self.slice_to_server[slice_id] = prev_server
            self._rebuild_keymap()
            return

        if kind == "split":
            old_id, old_slice, (n1, n2), old_server = payload
            if n1 in self.slices:
                del self.slices[n1]
                del self.slice_to_server[n1]
            if n2 in self.slices:
                del self.slices[n2]
                del self.slice_to_server[n2]
            self.slices[old_id] = old_slice
            self.slice_to_server[old_id] = old_server
            self._rebuild_keymap()
            return

        if kind == "merge":
            new_id, _merged, (a_id, a_slice), (b_id, b_slice), server = payload
            if new_id in self.slices:
                del self.slices[new_id]
                del self.slice_to_server[new_id]
            self.slices[a_id] = a_slice
            self.slices[b_id] = b_slice
            self.slice_to_server[a_id] = server
            self.slice_to_server[b_id] = server
            self._rebuild_keymap()
            return

        raise RuntimeError(f"Unknown op kind: {kind}")

    # ---- routing / accounting ----

    def route_workunits(self, work_units: List[WorkUnit]) -> Dict[int, float]:
        """Map WorkUnits (key ids) -> server work sums."""
        out: Dict[int, float] = {}

        for wu in work_units:
            key_id = wu.obj_id
            cost = wu.cost

            slice_loads = self.distribute_key_cost_to_slices(key_id, cost)
            if not slice_loads:
                raise KeyError(f"Key {key_id} not covered by any slice.")

            for sl, load in slice_loads.items():
                sid = self.slice_to_server[sl]
                out[sid] = out.get(sid, 0.0) + load

            # ---------------------------------------------------
            # NEW: hotspot bias toward primary server
            # ---------------------------------------------------
            if hasattr(self, "key_primary_server"):
                primary = self.key_primary_server.get(key_id)

                if primary is not None:
                    out[primary] = out.get(primary, 0.0) + cost * 0.35

        return out

    def distribute_key_cost_to_slices(self, key_id: int, cost: float) -> Dict[int, float]:
        slice_ids = list(self.key_to_slices.get(key_id, []))
        if not slice_ids:
            return {}
        if len(slice_ids) == 1:
            return {slice_ids[0]: float(cost)}

        weights = []
        for sl_id in slice_ids:
            sl = self.slices[sl_id]
            span = max(1, sl.hi - sl.lo)
            weights.append(max(1e-12, sl.size / span))
        total_w = sum(weights)
        if total_w <= 1e-12:
            share = float(cost) / len(slice_ids)
            return {sl_id: share for sl_id in slice_ids}
        return {
            sl_id: float(cost) * (w / total_w)
            for sl_id, w in zip(slice_ids, weights)
        }

    def record_server_work(self, server_id: int, work: float) -> None:
        hist = self._work_history[server_id]
        hist.append(work)
        if len(hist) > self._window:
            del hist[: len(hist) - self._window]

    def reset_work_history(self) -> None:
        for sid in self._work_history:
            self._work_history[sid].clear()

    def update_slice_loads(self, slice_loads: Dict[int, float]) -> None:
        for sl in self.slices.values():
            sl.load = float(slice_loads.get(sl.sid, 0.0))

    def grow_slices_on_servers(
        self,
        server_ids: List[int],
        *,
        growth_frac: float = 0.0,
        growth_add: float = 0.0,
    ) -> Tuple[List[Dict[str, Any]], float]:
        actions: List[Dict[str, Any]] = []
        total_added = 0.0
        for sid in server_ids:
            for sl_id in self.slices_on(sid):
                sl = self.slices.get(sl_id)
                if sl is None:
                    continue
                before = sl.size
                delta = before * float(growth_frac) + float(growth_add)
                if delta <= 0:
                    continue
                sl.size = before + delta
                total_added += delta
                actions.append({
                    "kind": "grow",
                    "slice_id": sl_id,
                    "server": sid,
                    "lo": sl.lo,
                    "hi": sl.hi,
                    "cluster": (sl.lo if sl.hi - sl.lo == 1 else None),
                    "size_before": before,
                    "size_after": sl.size,
                    "delta": delta,
                })
        return actions, total_added

    def add_popular_data_round_robin(
        self,
        popular_clusters: List[int],
        cluster_loads: Dict[int, float],
        *,
        growth_frac: float = 0.0,
        growth_add: float = 0.0,
        max_new_shards_per_cluster: int = 3,
    ) -> Tuple[List[Dict[str, Any]], float]:
        actions: List[Dict[str, Any]] = []
        server_ids = sorted(self.servers.keys())
        if not server_ids:
            return actions, 0.0

        targets = [int(k) for k in popular_clusters if int(k) in self.key_to_slices]
        if not targets:
            return actions, 0.0

        positive = [max(0.0, float(cluster_loads.get(k, 0.0))) for k in targets]
        total_pop_load = sum(positive)
        total_added = 0.0
        max_new_shards_per_cluster = max(1, int(max_new_shards_per_cluster))

        for idx, cluster in enumerate(targets):
            shard_ids = list(self.key_to_slices.get(cluster, []))
            if not shard_ids:
                continue
            current_size = sum(self.slices[slid].size for slid in shard_ids)
            cluster_load = max(0.0, float(cluster_loads.get(cluster, 0.0)))
            if total_pop_load > 1e-12:
                load_share = cluster_load / total_pop_load
            else:
                load_share = 1.0 / len(targets)

            # Heavier hot clusters grow disproportionately.
            skew = 0.5 + load_share * len(targets)
            total_growth = current_size * float(growth_frac) * skew + float(growth_add) * max(1.0, skew)
            if total_growth <= 0.0:
                continue

            n_new_shards = 1
            if max_new_shards_per_cluster > 1:
                n_new_shards += int(round((max_new_shards_per_cluster - 1) * min(1.0, load_share * len(targets))))
                n_new_shards = min(max_new_shards_per_cluster, n_new_shards)

            part_weights = [0.2 + self._rng.random() for _ in range(n_new_shards)]
            total_w = sum(part_weights)
            part_sizes = [total_growth * (w / total_w) for w in part_weights]
            part_sizes[-1] += total_growth - sum(part_sizes)

            for part_idx, part_size in enumerate(part_sizes):
                if part_size <= 0.0:
                    continue
                new_sid = self._new_slice_id()
                server_sid = server_ids[self._rr_server_cursor % len(server_ids)]
                self._rr_server_cursor += 1
                self.slices[new_sid] = Slice(sid=new_sid, lo=cluster, hi=cluster + 1, size=part_size, load=0.0)
                self.slice_to_server[new_sid] = server_sid
                total_added += part_size
                actions.append({
                    "kind": "add_hot_data",
                    "cluster": cluster,
                    "cluster_load": cluster_load,
                    "cluster_load_share": load_share,
                    "slice_id": new_sid,
                    "server": server_sid,
                    "lo": cluster,
                    "hi": cluster + 1,
                    "delta": part_size,
                    "part_index": part_idx,
                    "parts_for_cluster": n_new_shards,
                    "cluster_size_before": current_size,
                    "cluster_size_after": current_size + total_growth,
                })

        if actions:
            self._rebuild_keymap()
        return actions, total_added

    def split_hot_slices(self, hot_keys: List[int]) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        seen: set[int] = set()
        for k in hot_keys:
            candidates = list(self.key_to_slices.get(k, []))
            if not candidates:
                continue
            sl_id = max(candidates, key=lambda sid: (self.slices[sid].load, self.slices[sid].size, -sid))
            if sl_id in seen:
                continue
            seen.add(sl_id)
            sl = self.slices.get(sl_id)
            if sl is None:
                continue

            before_ids = set(self.slices.keys())
            before = {"slice_id": sl.sid, "lo": sl.lo, "hi": sl.hi, "size": sl.size}
            server = self.slice_to_server[sl_id]

            self.apply_split(sl_id)

            new_ids = sorted(set(self.slices.keys()) - before_ids)
            new_slices = [
                {
                    "slice_id": sid,
                    "lo": self.slices[sid].lo,
                    "hi": self.slices[sid].hi,
                    "size": self.slices[sid].size,
                    "server": self.slice_to_server[sid],
                }
                for sid in new_ids
            ]
            actions.append({
                "kind": "split_hot",
                "key": k,
                "parent_slice": before,
                "server": server,
                "new_slices": new_slices,
            })

        return actions

    def invariant_check(self, n_keys: int) -> None:
        for k in range(n_keys):
            if k not in self.key_to_slices or not self.key_to_slices[k]:
                raise AssertionError(f"Key {k} not covered by any slice.")
        for sid in self.servers:
            total = sum(self.slices[slid].size for slid in self.slices_on(sid))
            if total > self.servers[sid].capacity + 1e-9:
                raise AssertionError(f"Server {sid} over capacity: {total} > {self.servers[sid].capacity}")

    def snapshot(self, n_keys: int) -> Dict[str, Any]:
        slices = []
        for sl in sorted(self.slices.values(), key=lambda s: s.sid):
            slices.append({
                "slice_id": sl.sid,
                "lo": sl.lo,
                "hi": sl.hi,
                "cluster": (sl.lo if sl.hi - sl.lo == 1 else None),
                "size": sl.size,
                "load": sl.load,
                "server": self.slice_to_server[sl.sid],
            })
        key_to_slice = [self.key_to_slice[k] for k in range(n_keys)]
        key_to_server = [self.slice_to_server[slid] for slid in key_to_slice]
        cluster_to_slices = [list(self.key_to_slices.get(k, [])) for k in range(n_keys)]
        cluster_to_servers = [[self.slice_to_server[slid] for slid in slids] for slids in cluster_to_slices]
        cluster_shard_sizes = [[self.slices[slid].size for slid in slids] for slids in cluster_to_slices]
        return {
            "slices": slices,
            "key_to_slice": key_to_slice,
            "key_to_server": key_to_server,
            "cluster_to_slices": cluster_to_slices,
            "cluster_to_servers": cluster_to_servers,
            "cluster_shard_sizes": cluster_shard_sizes,
        }

    # ---- helpers ----

    def _recent_load(self, server_id: int) -> float:
        hist = self._work_history[server_id]
        return float(sum(hist)) if hist else 0.0

    def _slice_estimated_loads(self) -> Tuple[Dict[int, float], float]:
        loads = {sid: 0.0 for sid in self.servers}
        total = 0.0
        for sl_id, sl in self.slices.items():
            v = float(sl.load)
            if v <= 0.0:
                continue
            sid = self.slice_to_server[sl_id]
            loads[sid] = loads.get(sid, 0.0) + v
            total += v
        return loads, total

    def _current_load(self, server_id: int) -> float:
        slice_loads, total = self._slice_estimated_loads()
        if total > 1e-12:
            return slice_loads.get(server_id, 0.0)
        return self._recent_load(server_id)

    def _current_util(self, server_id: int) -> float:
        cap = float(self.servers[server_id].capacity)
        if cap <= 1e-12:
            return 0.0
        return self._current_load(server_id) / cap

    def _all_current_utils(self) -> Dict[int, float]:
        slice_loads, total = self._slice_estimated_loads()
        out: Dict[int, float] = {}
        if total > 1e-12:
            for sid, server in self.servers.items():
                cap = float(server.capacity)
                out[sid] = 0.0 if cap <= 1e-12 else slice_loads.get(sid, 0.0) / cap
            return out
        for sid, server in self.servers.items():
            cap = float(server.capacity)
            out[sid] = 0.0 if cap <= 1e-12 else self._recent_load(sid) / cap
        return out

    def _recent_util(self, server_id: int) -> float:
        cap = float(self.servers[server_id].capacity)
        if cap <= 1e-12:
            return 0.0
        return self._recent_load(server_id) / cap


    def _new_slice_id(self, start: Optional[int] = None) -> int:
        if start is None:
            start = max(self.slices.keys(), default=0) + 1
        x = start
        while x in self.slices:
            x += 1
        return x

    def _rebuild_keymap(self) -> None:
        self.key_to_slice: Dict[int, int] = {}
        self.key_to_slices: Dict[int, List[int]] = {}
        for sl in self.slices.values():
            for k in range(sl.lo, sl.hi):
                self.key_to_slices.setdefault(k, []).append(sl.sid)
        for k, slids in self.key_to_slices.items():
            slids.sort(key=lambda sid: (-self.slices[sid].size, sid))
            self.key_to_slice[k] = slids[0]


# -----------------------------
# Policy A: Greedy Slicer weighted-move
# -----------------------------

def greedy_slicer_balance(
    state: ShardState,
    churn_budget: float,
    *,
    action_log: Optional[List[Dict[str, Any]]] = None,
) -> ShardState:
    """Greedy weighted-move rebalance.

    In the current simplified load model, only reassigning a shard between servers
    changes per-server load immediately. Hot-shard splitting is handled separately
    before this function runs, and same-server merges do not change imbalance.
    """
    churn_used = 0.0

    while churn_used < churn_budget:
        hot = state.hottest_server()
        cold = state.coldest_server()
        if hot == cold:
            break

        best_move = None
        best_weight = 0.0
        current_imbalance = state.imbalance()
        if current_imbalance <= 0.0:
            break

        server_loads, _total_slice_load = state._slice_estimated_loads()
        utils = state._all_current_utils()
        n_servers = len(state.servers)
        total_util = sum(utils.values())
        hot_cap = max(1e-12, float(state.servers[hot].capacity))
        cold_cap = max(1e-12, float(state.servers[cold].capacity))
        other_max_util = 0.0
        for sid, util in utils.items():
            if sid == hot or sid == cold:
                continue
            if util > other_max_util:
                other_max_util = util

        # Candidate moves touching the hottest server: shard reassignments only.
        for sl in state.slices_on(hot):
            churn = state.keyspace(sl)
            if churn <= 0.0:
                continue
            moved_load = max(0.0, float(state.slices[sl].load))
            if moved_load <= 0.0:
                continue
            #FIXME figure out why there is a spike, problem due to logging logic, maybe code is to complex
            hot_after_util = max(0.0, server_loads.get(hot, 0.0) - moved_load) / hot_cap
            cold_after_util = (server_loads.get(cold, 0.0) + moved_load) / cold_cap
            total_util_after = total_util - (moved_load / hot_cap) + (moved_load / cold_cap)
            mean_util_after = total_util_after / n_servers if n_servers > 0 else 0.0
            if mean_util_after <= 1e-12:
                continue

            new_max_util = max(other_max_util, hot_after_util, cold_after_util)
            new_imbalance = new_max_util / mean_util_after
            benefit = current_imbalance - new_imbalance
            if benefit <= 0.0:
                continue

            w = benefit / churn
            if w > best_weight:
                best_weight = w
                best_move = ("reassign", sl, churn)

        if best_move is None or best_weight <= 0:
            break

        kind, payload, churn = best_move
        if churn_used + churn > churn_budget:
            break

        before_imbalance = current_imbalance

        action: Dict[str, Any] = {
            "kind": kind,
            "churn": churn,
            "hot_server": hot,
            "cold_server": cold,
        }

        src_server = state.slice_to_server[payload]
        moved_slice = state.slices[payload]
        state.apply_reassign(payload, cold)
        action.update({
            "slice_id": payload,
            "lo": moved_slice.lo,
            "hi": moved_slice.hi,
            "cluster": (moved_slice.lo if moved_slice.hi - moved_slice.lo == 1 else None),
            "from": src_server,
            "to": cold,
        })

        churn_used += churn
        after_imbalance = state.imbalance()
        action.update({
            "imbalance_before": before_imbalance,
            "imbalance_after": after_imbalance,
        })
        if action_log is not None:
            action_log.append(action)

    return state


# -----------------------------
# Policy B: ILP/LP (your formulation) using Gurobi
# -----------------------------

def ilp_rebalance_keys_gurobi(
    n_keys: int,
    n_servers: int,
    r_i: List[float],
    R_j: List[float],
    current_alloc: List[int],  # server index 0..n_servers-1
    freq_i: List[float],       # workload weight per key (includes cost)
    K: int,
    *,
    time_limit_s: float = 10.0,
    mip_gap: float = 0.01,
    verbose: bool = False,
) -> List[int]:
    """
    Minimize Z
    s.t. sum_j x_ij = 1
         sum_i x_ij r_i <= R_j
         sum_i x_ij freq_i <= Z
         y_i >= 1 - x_i,cur(i)
         sum_i y_i <= K
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise RuntimeError(
            "gurobipy is required to use Gurobi.\n"
            "Ensure Gurobi is installed + licensed and gurobipy is importable.\n"
            f"Import error: {e}"
        )

    if len(r_i) != n_keys or len(freq_i) != n_keys or len(current_alloc) != n_keys:
        raise ValueError("r_i, freq_i, current_alloc must have length n_keys")
    if len(R_j) != n_servers:
        raise ValueError("R_j must have length n_servers")

    m = gp.Model("slicer_ilp")
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.TimeLimit = float(time_limit_s)
    m.Params.MIPGap = float(mip_gap)

    x = m.addVars(n_keys, n_servers, vtype=GRB.BINARY, name="x")
    y = m.addVars(n_keys, vtype=GRB.BINARY, name="y")
    Z = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="Z")

    m.setObjective(Z, GRB.MINIMIZE)

    # assignment
    for i in range(n_keys):
        m.addConstr(gp.quicksum(x[i, j] for j in range(n_servers)) == 1, name=f"assign_{i}")

    # capacity + load bound
    for j in range(n_servers):
        m.addConstr(gp.quicksum(x[i, j] * r_i[i] for i in range(n_keys)) <= R_j[j], name=f"cap_{j}")
        m.addConstr(gp.quicksum(x[i, j] * freq_i[i] for i in range(n_keys)) <= Z, name=f"load_{j}")

    # migration budget
    #FIXME figure out whats going on w the files and
    for i in range(n_keys):
        j0 = current_alloc[i]
        if not (0 <= j0 < n_servers):
            raise ValueError(f"current_alloc[{i}]={j0} out of range")
        m.addConstr(y[i] >= 1 - x[i, j0], name=f"mig_{i}")

    m.addConstr(gp.quicksum(y[i] for i in range(n_keys)) <= K, name="mig_budget")

    m.optimize()

    if m.SolCount == 0:
        raise RuntimeError("Gurobi found no feasible solution (SolCount=0)")

    new_alloc: List[int] = []
    for i in range(n_keys):
        chosen = None
        for j in range(n_servers):
            if x[i, j].X > 0.5:
                chosen = j
                break
        if chosen is None:
            chosen = max(range(n_servers), key=lambda jj: x[i, jj].X)
        new_alloc.append(chosen)
    return new_alloc


def rebuild_as_single_key_slices(
    state: ShardState,
    n_keys: int,
    key_sizes: List[float],
    key_to_server_sid: List[int],  # length n_keys, server SID per key
) -> None:
    slices: List[Slice] = []
    slice_to_server: Dict[int, int] = {}
    for k in range(n_keys):
        slid = k + 1
        slices.append(Slice(sid=slid, lo=k, hi=k + 1, size=key_sizes[k]))
        slice_to_server[slid] = key_to_server_sid[k]

    state.slices = {sl.sid: sl for sl in slices}
    state.slice_to_server = slice_to_server
    state._op_stack.clear()
    state._rebuild_keymap()


def build_initial_state(
    rng: random.Random,
    servers: List[Server],
    n_keys: int,
    key_sizes: List[float],
    initial_slices: int,
    window: int = 200,
    *,
    chunk_size: Optional[int] = None,
    cluster_shards_min: int = 1,
    cluster_shards_max: int = 1,
) -> ShardState:
    """Build an initial slice assignment.

    Two modes:
      0) cluster_shards_max > 1 (or min > 1): each logical cluster (key) starts with
         multiple same-key shards of unequal size; all shards for a key are initially
         placed on the same server, so later rebalancing can move shards (not whole clusters).
      1) chunk_size provided: deterministic contiguous slices of size=chunk_size (last may be smaller),
         assigned round-robin to servers. This matches the "split into chunks of 5 and round-robin" idea.
      2) otherwise: random cuts into `initial_slices` slices (original behavior).
    """
    cluster_shards_min = max(1, int(cluster_shards_min))
    cluster_shards_max = max(cluster_shards_min, int(cluster_shards_max))

    if cluster_shards_max > 1 or cluster_shards_min > 1:
        slices = []
        slice_to_server: Dict[int, int] = {}
        sid = 1

        for k in range(n_keys):
            n_parts = rng.randint(cluster_shards_min, cluster_shards_max)
            n_parts = max(1, n_parts)

            if n_parts == 1:
                shard_sizes = [float(key_sizes[k])]
            else:
                # Unequal partitions (Dirichlet-like via random positive weights).
                w = [0.2 + rng.random() for _ in range(n_parts)]
                tot_w = sum(w)
                shard_sizes = [float(key_sizes[k]) * (x / tot_w) for x in w]
                # Fix roundoff to preserve exact total size.
                shard_sizes[-1] += float(key_sizes[k]) - sum(shard_sizes)

            # Start all shards for a cluster on the same server so later moves are shard-level.
            server_sid = servers[k % len(servers)].sid
            for sz in shard_sizes:
                slices.append(Slice(sid=sid, lo=k, hi=k + 1, size=float(sz)))
                slice_to_server[sid] = server_sid
                sid += 1
    elif chunk_size is not None and chunk_size > 0:
        slices: List[Slice] = []
        sid = 1
        for lo in range(0, n_keys, chunk_size):
            hi = min(n_keys, lo + chunk_size)
            size = sum(key_sizes[lo:hi])
            slices.append(Slice(sid=sid, lo=lo, hi=hi, size=size))
            sid += 1
        slice_to_server = {sl.sid: servers[(sl.sid - 1) % len(servers)].sid for sl in slices}
    else:
        initial_slices = max(1, min(initial_slices, n_keys))
        cuts = sorted(rng.sample(range(1, n_keys), k=initial_slices - 1))
        cuts = [0] + cuts + [n_keys]

        slices = []
        for idx in range(len(cuts) - 1):
            lo, hi = cuts[idx], cuts[idx + 1]
            size = sum(key_sizes[lo:hi])
            slices.append(Slice(sid=idx + 1, lo=lo, hi=hi, size=size))

        slice_to_server = {sl.sid: servers[(sl.sid - 1) % len(servers)].sid for sl in slices}
    # ---------------------------------------------------
    # NEW: assign a primary server for each key
    # ---------------------------------------------------
    # create the state object first
    state = ShardState(
        servers = servers,
        slices = slices,
        slice_to_server = slice_to_server,
        window = window,
        rng = rng,
    )

    # ---------------------------------------------------
    # NEW: assign a primary server for each key
    # ---------------------------------------------------
    state.key_primary_server = {}

    for k in range(n_keys):
        key_slices = state.key_to_slices.get(k, [])
        if not key_slices:
            continue

        # choose first slice's server as primary
        primary_slice = key_slices[0]
        primary_server = state.slice_to_server[primary_slice]

        state.key_primary_server[k] = primary_server

    return state