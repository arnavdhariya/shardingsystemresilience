# search.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence
import random


@dataclass(frozen=True)
class WorkUnit:
    """Logical work unit. For IVF: obj_id=list_id, cost=scan/compute cost."""
    obj_id: int #ivf cluster
    cost: float #cost is the computation cost


@dataclass(frozen=True)
class Query:
    qid: int
    payload: tuple  # contains hot clusters/distribution/burst info to create hotspots



class SearchAlgorithm(Protocol):
    def plan(self, query: Query) -> List[WorkUnit]:
        ...


class IVFSearch(SearchAlgorithm):
    """IVF planner: returns nprobe lists per query.

    Default behavior: weighted draws without replacement (approx).
    If allow_repeat=True, probes are drawn with replacement to allow
    small hot sets to dominate (useful for 80/20 bursty workloads).

    If query.payload includes a dict in payload[0] with keys:
      - "hot": iterable[int] hot list_ids
      - "hot_mass": float in [0,1], fraction of probes to draw from the hot pool
    then we draw ~hot_mass*nprobe probes from the hot pool and the rest from cold.
    This lets fan-out change during spike intervals.
    """

    def __init__(
        self,
        rng: random.Random,
        n_lists: int, #number of clusters
        nprobe: int, #lists searched per query
        list_cost: Optional[List[float]] = None, #cost to scan a cluster
        list_weights: Optional[List[float]] = None, #prob cluster chosen
        allow_repeat: bool = False,
    ):
        self.rng = rng
        self.n_lists = n_lists
        self.nprobe = nprobe
        self.allow_repeat = bool(allow_repeat)

        self.list_cost = list_cost or [1.0] * n_lists
        if len(self.list_cost) != n_lists:
            raise ValueError("list_cost length must equal n_lists")

        if list_weights is None:
            list_weights = [1.0] * n_lists
        if len(list_weights) != n_lists:
            raise ValueError("list_weights length must equal n_lists")

        tot = sum(list_weights)
        self.prob = [w / tot for w in list_weights]

    def plan(self, query: Query) -> List[WorkUnit]:
        nprobe = self.nprobe

        if hasattr(self, "probe_jitter"):
            jitter = self.rng.randint(-self.probe_jitter, self.probe_jitter)
            nprobe = max(1, self.nprobe + jitter)
        if self.allow_repeat:
            target = nprobe
        else:
            target = min(nprobe, self.n_lists)

        hot_cfg = None
        if query.payload and isinstance(query.payload[0], dict) and query.payload[0]:
            hot_cfg = query.payload[0]

        if hot_cfg is None:
            if self.allow_repeat:
                chosen = [self._weighted_draw() for _ in range(target)]
                return [WorkUnit(obj_id=i, cost=self.list_cost[i]) for i in chosen]

            chosen_set: set[int] = set()
            attempts = 0
            while len(chosen_set) < target and attempts < target * 50:
                chosen_set.add(self._weighted_draw())
                attempts += 1
            if len(chosen_set) < target:
                remaining = [i for i in range(self.n_lists) if i not in chosen_set]
                self.rng.shuffle(remaining)
                for i in remaining[: target - len(chosen_set)]:
                    chosen_set.add(i)
            return [WorkUnit(obj_id=i, cost=self.list_cost[i]) for i in chosen_set]

        if hot_cfg is not None:
            hot = list(dict.fromkeys(int(x) for x in hot_cfg.get("hot", [])))
            hot = [x for x in hot if 0 <= x < self.n_lists]
            hot_weights = self._normalized_item_weights(hot, hot_cfg.get("hot_weights"))
            hot_mass = float(hot_cfg.get("hot_mass", 0.0))
            hot_mass = max(0.0, min(1.0, hot_mass))
            n_hot = int(round(hot_mass * target))
            n_hot = max(0, min(n_hot, target))
            hot_set = set(hot)
            cold_pool = [i for i in range(self.n_lists) if i not in hot_set]
            if not hot:
                n_hot = 0
            n_cold = max(0, target - n_hot)

            dist_mode = str(hot_cfg.get("dist_mode", "zipf"))

            if self.allow_repeat:
                chosen: List[int] = []
                if n_hot > 0:
                    for _ in range(n_hot):
                        chosen.append(self._weighted_draw_from_items(hot, hot_weights))
                pool = cold_pool if cold_pool else hot or list(range(self.n_lists))
                for _ in range(n_cold):
                    if dist_mode == "zipf":
                        chosen.append(self._weighted_draw_from_pool(pool))
                    else:
                        chosen.append(pool[self.rng.randrange(len(pool))])
                return [WorkUnit(obj_id=i, cost=self.list_cost[i]) for i in chosen]

            n_hot = min(n_hot, len(hot))
            chosen_set: set[int] = set()

            if n_hot > 0:
                chosen_set.update(self._weighted_sample_without_replacement(hot, hot_weights, n_hot))

            # Fill remaining from cold pool (uniform or zipf, no repeats)
            pool = cold_pool if cold_pool else [i for i in range(self.n_lists) if i not in chosen_set]
            if dist_mode == "zipf":
                attempts = 0
                while len(chosen_set) < target and attempts < target * 50 and pool:
                    i = self._weighted_draw_from_pool(pool)
                    if i not in chosen_set:
                        chosen_set.add(i)
                    attempts += 1
            else:
                self.rng.shuffle(pool)
                for i in pool[: max(0, target - len(chosen_set))]:
                    chosen_set.add(i)

            if len(chosen_set) < target:
                remaining = [i for i in range(self.n_lists) if i not in chosen_set]
                self.rng.shuffle(remaining)
                for i in remaining[: target - len(chosen_set)]:
                    chosen_set.add(i)

        return [WorkUnit(obj_id=i, cost=self.list_cost[i]) for i in chosen_set]

    def plan_counts(self, n_requests: int, hot_cfg: Optional[dict] = None) -> dict[int, int]:
        total_probes = max(0, int(n_requests)) * int(self.nprobe)
        if total_probes <= 0:
            return {}

        hot: List[int] = []
        hot_mass = 0.0
        dist_mode = "zipf"
        if hot_cfg:
            hot = list(dict.fromkeys(int(x) for x in hot_cfg.get("hot", [])))
            hot = [x for x in hot if 0 <= x < self.n_lists]
            hot_weights = self._normalized_item_weights(hot, hot_cfg.get("hot_weights"))
            hot_mass = float(hot_cfg.get("hot_mass", 0.0))
            hot_mass = max(0.0, min(1.0, hot_mass))
            dist_mode = str(hot_cfg.get("dist_mode", "zipf"))
        else:
            hot_weights = []

        n_hot = int(round(hot_mass * total_probes))
        n_hot = max(0, min(n_hot, total_probes))
        if not hot:
            n_hot = 0

        hot_set = set(hot)
        cold_pool = [i for i in range(self.n_lists) if i not in hot_set]
        if not cold_pool:
            n_hot = total_probes

        n_cold = max(0, total_probes - n_hot)
        counts: dict[int, int] = {}

        if hot_cfg is None:
            return self._weighted_counts(total_probes)

        self._weighted_counts_for_items(counts, hot, hot_weights, n_hot)
        pool = cold_pool if cold_pool else hot or list(range(self.n_lists))
        if dist_mode == "zipf":
            self._weighted_counts_from_pool(counts, pool, n_cold)
        else:
            self._even_split(counts, pool, n_cold)
        return counts

    def _even_split(self, counts: dict[int, int], items: List[int], n: int) -> None:
        if n <= 0 or not items:
            return
        base = n // len(items)
        rem = n % len(items)
        for i in items:
            counts[i] = counts.get(i, 0) + base
        if rem > 0:
            pool = list(items)
            self.rng.shuffle(pool)
            for i in pool[:rem]:
                counts[i] = counts.get(i, 0) + 1

    def _weighted_counts_for_items(
        self,
        counts: dict[int, int],
        items: Sequence[int],
        weights: Sequence[float],
        n: int,
    ) -> None:
        if n <= 0 or not items:
            return
        if len(weights) != len(items):
            self._even_split(counts, list(items), n)
            return
        total_w = sum(max(0.0, float(w)) for w in weights)
        if total_w <= 1e-12:
            self._even_split(counts, list(items), n)
            return

        normalized = [max(0.0, float(w)) / total_w for w in weights]
        expected = [w * n for w in normalized]
        base = [int(x) for x in expected]
        rem = n - sum(base)
        for idx, b in enumerate(base):
            if b:
                item = int(items[idx])
                counts[item] = counts.get(item, 0) + b
        if rem > 0:
            frac = [expected[i] - base[i] for i in range(len(items))]
            idxs = list(range(len(items)))
            self.rng.shuffle(idxs)
            idxs.sort(key=lambda i: frac[i], reverse=True)
            for i in idxs[:rem]:
                item = int(items[i])
                counts[item] = counts.get(item, 0) + 1

    def _uniform_counts(self, total_probes: int) -> dict[int, int]:
        counts: dict[int, int] = {}
        items = list(range(self.n_lists))
        self._even_split(counts, items, total_probes)
        return counts

    def _weighted_counts(self, total_probes: int) -> dict[int, int]:
        counts: dict[int, int] = {}
        if total_probes <= 0:
            return counts
        expected = [p * total_probes for p in self.prob]
        base = [int(x) for x in expected]
        rem = total_probes - sum(base)
        for i, b in enumerate(base):
            if b:
                counts[i] = b
        if rem > 0:
            frac = [expected[i] - base[i] for i in range(self.n_lists)]
            idxs = list(range(self.n_lists))
            self.rng.shuffle(idxs)
            idxs.sort(key=lambda i: frac[i], reverse=True)
            for i in idxs[:rem]:
                counts[i] = counts.get(i, 0) + 1
        return counts

    def _weighted_counts_from_pool(self, counts: dict[int, int], pool: List[int], total_probes: int) -> None:
        if total_probes <= 0 or not pool:
            return
        weights = [self.prob[i] for i in pool]
        total_w = sum(weights)
        if total_w <= 1e-12:
            self._even_split(counts, pool, total_probes)
            return
        expected = [w / total_w * total_probes for w in weights]
        base = [int(x) for x in expected]
        rem = total_probes - sum(base)
        for idx, b in enumerate(base):
            if b:
                k = pool[idx]
                counts[k] = counts.get(k, 0) + b
        if rem > 0:
            frac = [expected[i] - base[i] for i in range(len(pool))]
            idxs = list(range(len(pool)))
            self.rng.shuffle(idxs)
            idxs.sort(key=lambda i: frac[i], reverse=True)
            for i in idxs[:rem]:
                k = pool[i]
                counts[k] = counts.get(k, 0) + 1

    def _weighted_draw_from_pool(self, pool: List[int]) -> int:
        if not pool:
            return self._weighted_draw()
        total = sum(self.prob[i] for i in pool)
        if total <= 1e-12:
            return pool[self.rng.randrange(len(pool))]
        r = self.rng.random() * total
        acc = 0.0
        for i in pool:
            acc += self.prob[i]
            if r <= acc:
                return i
        return pool[-1]

    def _weighted_draw_from_items(self, items: Sequence[int], weights: Sequence[float]) -> int:
        if not items:
            return self._weighted_draw()
        if len(weights) != len(items):
            return items[self.rng.randrange(len(items))]
        total = sum(max(0.0, float(w)) for w in weights)
        if total <= 1e-12:
            return items[self.rng.randrange(len(items))]
        r = self.rng.random() * total
        acc = 0.0
        for item, weight in zip(items, weights):
            acc += max(0.0, float(weight))
            if r <= acc:
                return int(item)
        return int(items[-1])

    def _weighted_sample_without_replacement(
        self,
        items: Sequence[int],
        weights: Sequence[float],
        k: int,
    ) -> List[int]:
        pool = list(items)
        ws = list(weights)
        out: List[int] = []
        k = max(0, min(int(k), len(pool)))
        while pool and len(out) < k:
            picked = self._weighted_draw_from_items(pool, ws)
            idx = pool.index(picked)
            out.append(int(picked))
            del pool[idx]
            if idx < len(ws):
                del ws[idx]
        return out

    def _normalized_item_weights(self, items: Sequence[int], raw_weights: object) -> List[float]:
        if not items:
            return []
        if not isinstance(raw_weights, (list, tuple)):
            return [1.0 / len(items)] * len(items)

        cleaned: List[float] = []
        for idx in range(len(items)):
            if idx < len(raw_weights):
                try:
                    cleaned.append(max(0.0, float(raw_weights[idx])))
                except (TypeError, ValueError):
                    cleaned.append(0.0)
            else:
                cleaned.append(0.0)

        total = sum(cleaned)
        if total <= 1e-12:
            return [1.0 / len(items)] * len(items)
        return [w / total for w in cleaned]

    def _weighted_draw(self) -> int:
        r = self.rng.random()
        acc = 0.0
        for i, p in enumerate(self.prob):
            acc += p
            if r <= acc:
                return i
        return self.n_lists - 1
