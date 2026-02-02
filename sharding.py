
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
    ):
        self.servers: Dict[int, Server] = {sv.sid: sv for sv in servers}
        self.slices: Dict[int, Slice] = {sl.sid: sl for sl in slices}
        self.slice_to_server: Dict[int, int] = dict(slice_to_server)

        self._op_stack: List[Tuple[str, Any]] = []
        self._window = max(1, window)
        self._work_history: Dict[int, List[float]] = {sid: [] for sid in self.servers}

        self._rebuild_keymap()

    # ---- interface used by greedy_slicer_balance ----

    def slices_on(self, server_id: int) -> List[int]:
        return [slid for slid, sid in self.slice_to_server.items() if sid == server_id]

    def adjacent_cold_pairs(self, server_id: int) -> List[Tuple[int, int]]:
        sl_ids = self.slices_on(server_id)
        sls = sorted((self.slices[i] for i in sl_ids), key=lambda s: (s.lo, s.hi))
        pairs: List[Tuple[int, int]] = []
        for a, b in zip(sls, sls[1:]):
            if a.hi == b.lo:
                pairs.append((a.sid, b.sid))
        return pairs

    def keyspace(self, slice_id: int) -> float:
        return float(self.slices[slice_id].size)

    def imbalance(self) -> float:
        loads = [self._recent_load(sid) for sid in self.servers]
        return (max(loads) - min(loads)) if loads else 0.0

    def hottest_server(self) -> int:
        return max(self.servers, key=lambda sid: self._recent_load(sid))

    def coldest_server(self) -> int:
        return min(self.servers, key=lambda sid: self._recent_load(sid))

    def apply_reassign(self, slice_id: int, dst_server: int) -> None:
        prev = self.slice_to_server[slice_id]
        self._op_stack.append(("reassign", (slice_id, prev)))
        self.slice_to_server[slice_id] = dst_server
        self._rebuild_keymap()

    def apply_split(self, slice_id: int) -> None:
        sl = self.slices[slice_id]
        if sl.hi - sl.lo <= 1:
            self._op_stack.append(("noop", None))
            return

        mid = (sl.lo + sl.hi) // 2
        if mid == sl.lo or mid == sl.hi:
            self._op_stack.append(("noop", None))
            return

        new_id1 = self._new_slice_id()
        new_id2 = self._new_slice_id(new_id1)
        size1 = sl.size / 2.0
        size2 = sl.size - size1

        s1 = Slice(sid=new_id1, lo=sl.lo, hi=mid, size=size1)
        s2 = Slice(sid=new_id2, lo=mid, hi=sl.hi, size=size2)
        server = self.slice_to_server[slice_id]

        self._op_stack.append(("split", (slice_id, sl, (new_id1, new_id2), server)))

        del self.slices[slice_id]
        del self.slice_to_server[slice_id]
        self.slices[new_id1] = s1
        self.slices[new_id2] = s2
        self.slice_to_server[new_id1] = server
        self.slice_to_server[new_id2] = server
        self._rebuild_keymap()

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

        if not (sa.hi == sb.lo or sb.hi == sa.lo):
            self._op_stack.append(("noop", None))
            return

        lo = min(sa.lo, sb.lo)
        hi = max(sa.hi, sb.hi)
        merged_size = sa.size + sb.size
        new_id = self._new_slice_id()
        merged = Slice(sid=new_id, lo=lo, hi=hi, size=merged_size)

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
            sl = self.key_to_slice.get(wu.obj_id)
            if sl is None:
                raise KeyError(f"Key {wu.obj_id} not covered by any slice.")
            sid = self.slice_to_server[sl]
            out[sid] = out.get(sid, 0.0) + wu.cost
        return out

    def record_server_work(self, server_id: int, work: float) -> None:
        hist = self._work_history[server_id]
        hist.append(work)
        if len(hist) > self._window:
            del hist[: len(hist) - self._window]

    def invariant_check(self, n_keys: int) -> None:
        for k in range(n_keys):
            if k not in self.key_to_slice:
                raise AssertionError(f"Key {k} not covered by any slice.")
        for sid in self.servers:
            total = sum(self.slices[slid].size for slid in self.slices_on(sid))
            if total > self.servers[sid].capacity + 1e-9:
                raise AssertionError(f"Server {sid} over capacity: {total} > {self.servers[sid].capacity}")

    # ---- helpers ----

    def _recent_load(self, server_id: int) -> float:
        hist = self._work_history[server_id]
        return float(sum(hist)) if hist else 0.0

    def _new_slice_id(self, start: Optional[int] = None) -> int:
        if start is None:
            start = max(self.slices.keys(), default=0) + 1
        x = start
        while x in self.slices:
            x += 1
        return x

    def _rebuild_keymap(self) -> None:
        self.key_to_slice: Dict[int, int] = {}
        for sl in self.slices.values():
            for k in range(sl.lo, sl.hi):
                self.key_to_slice[k] = sl.sid


# -----------------------------
# Policy A: Greedy Slicer weighted-move
# -----------------------------

def greedy_slicer_balance(state: ShardState, churn_budget: float) -> ShardState:
    churn_used = 0.0

    while churn_used < churn_budget:
        hot = state.hottest_server()
        cold = state.coldest_server()

        best_move = None
        best_weight = 0.0

        # candidate moves touching hottest server
        for sl in state.slices_on(hot):
            # REASSIGN
            before = state.imbalance()
            churn = state.keyspace(sl)

            state.apply_reassign(sl, cold)
            benefit = before - state.imbalance()
            state.undo_last()

            if churn > 0:
                w = benefit / churn
                if w > best_weight:
                    best_weight = w
                    best_move = ("reassign", sl, churn)

            # SPLIT
            before = state.imbalance()
            churn = state.keyspace(sl)

            state.apply_split(sl)
            benefit = before - state.imbalance()
            state.undo_last()

            if churn > 0:
                w = benefit / churn
                if w > best_weight:
                    best_weight = w
                    best_move = ("split", sl, churn)

        # merge cold adjacent slices
        for sl1, sl2 in state.adjacent_cold_pairs(cold):
            before = state.imbalance()
            churn = state.keyspace(sl1) + state.keyspace(sl2)

            state.apply_merge(sl1, sl2)
            benefit = before - state.imbalance()
            state.undo_last()

            if churn > 0:
                w = benefit / churn
                if w > best_weight:
                    best_weight = w
                    best_move = ("merge", (sl1, sl2), churn)

        if best_move is None or best_weight <= 0:
            break

        kind, payload, churn = best_move
        if churn_used + churn > churn_budget:
            break

        if kind == "reassign":
            state.apply_reassign(payload, cold)
        elif kind == "split":
            state.apply_split(payload)
        elif kind == "merge":
            a, b = payload
            state.apply_merge(a, b)

        churn_used += churn

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
) -> ShardState:
    initial_slices = max(1, min(initial_slices, n_keys))
    cuts = sorted(rng.sample(range(1, n_keys), k=initial_slices - 1))
    cuts = [0] + cuts + [n_keys]

    slices: List[Slice] = []
    for idx in range(len(cuts) - 1):
        lo, hi = cuts[idx], cuts[idx + 1]
        size = sum(key_sizes[lo:hi])
        slices.append(Slice(sid=idx + 1, lo=lo, hi=hi, size=size))

    slice_to_server = {sl.sid: (sl.sid - 1) % len(servers) for sl in slices}
    return ShardState(servers=servers, slices=slices, slice_to_server=slice_to_server, window=window)
