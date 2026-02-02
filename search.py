# search.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Protocol
import random


@dataclass(frozen=True)
class WorkUnit:
    """Logical work unit. For IVF: obj_id=list_id, cost=scan/compute cost."""
    obj_id: int
    cost: float


@dataclass(frozen=True)
class Query:
    qid: int
    payload: tuple[float, ...]  # keep generic


class SearchAlgorithm(Protocol):
    def plan(self, query: Query) -> List[WorkUnit]:
        ...


class IVFSearch(SearchAlgorithm):
    """
    IVF planner: returns nprobe lists per query (weighted, approx w/o replacement).
    """
    def __init__(
        self,
        rng: random.Random,
        n_lists: int,
        nprobe: int,
        list_cost: Optional[List[float]] = None,
        list_weights: Optional[List[float]] = None,
    ):
        self.rng = rng
        self.n_lists = n_lists
        self.nprobe = nprobe

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
        target = min(self.nprobe, self.n_lists)
        chosen: set[int] = set()
        attempts = 0

        while len(chosen) < target and attempts < target * 20:
            chosen.add(self._weighted_draw())
            attempts += 1

        if len(chosen) < target:
            remaining = [i for i in range(self.n_lists) if i not in chosen]
            self.rng.shuffle(remaining)
            for i in remaining[: target - len(chosen)]:
                chosen.add(i)

        return [WorkUnit(obj_id=i, cost=self.list_cost[i]) for i in chosen]

    def _weighted_draw(self) -> int:
        r = self.rng.random()
        acc = 0.0
        for i, p in enumerate(self.prob):
            acc += p
            if r <= acc:
                return i
        return self.n_lists - 1
