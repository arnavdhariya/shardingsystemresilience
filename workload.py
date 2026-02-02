# workload.py
from __future__ import annotations
from typing import Protocol
import random
from search import Query


class QueryDistribution(Protocol):
    def sample(self, rng: random.Random, qid: int) -> Query:
        ...


class RandomVectorQueryDist:
    """Just generates a random payload; search ignores payload in this toy scaffold."""
    def __init__(self, dim: int = 4):
        self.dim = dim

    def sample(self, rng: random.Random, qid: int) -> Query:
        payload = tuple(rng.random() for _ in range(self.dim))
        return Query(qid=qid, payload=payload)


class Workload:
    def __init__(self, rng: random.Random, dist: QueryDistribution):
        self.rng = rng
        self.dist = dist
        self._qid = 0

    def next(self) -> Query:
        self._qid += 1
        return self.dist.sample(self.rng, self._qid)


def zipf_weights(n: int, s: float, rng: random.Random) -> list[float]:
    """
    Returns shuffled Zipf-ish weights normalized to sum=1.
    Higher s => more skew => more hotspots.
    """
    w = [1.0 / ((i + 1) ** s) for i in range(n)]
    rng.shuffle(w)
    tot = sum(w)
    return [x / tot for x in w]
