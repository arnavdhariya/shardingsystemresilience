# workload.py
from __future__ import annotations
from typing import Optional, Protocol
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



class ClusterSpikeQueryDist:
    """Generates queries with a time-varying "hot set" of logical clusters (keys).

    Idea:
      - There are `n_lists` logical clusters (0..n_lists-1).
      - At any time, `n_hot` clusters receive `hot_mass` fraction of attention.
      - Every `spike_every` timesteps, we pick a new hot set (optionally with a spike_duration window).
      - We encode the hot set into Query.payload as a dict so the search planner can change fan-out.

    This matches: "1-2 clusters out of n become hot, causing fan-out spikes".
    """
    def __init__(
        self,
        n_lists: int,
        n_hot: int = 2,
        hot_mass: float = 0.8,
        spike_every: int = 200,
        spike_duration: int = 200,
    ):
        self.n_lists = n_lists
        self.n_hot = max(1, int(n_hot))
        self.hot_mass = max(0.0, min(1.0, float(hot_mass)))
        self.spike_every = max(1, int(spike_every))
        self.spike_duration = max(1, int(spike_duration))

    def sample(self, rng: random.Random, qid: int) -> Query:
        # Pick a (possibly changing) hot set based on qid as a proxy for time.
        epoch = (qid - 1) // self.spike_every
        rng2 = random.Random((epoch + 1) * 9973)  # deterministic hot-set per epoch
        hot = list(range(self.n_lists))
        rng2.shuffle(hot)
        hot = hot[: min(self.n_hot, self.n_lists)]

        # Only "activate" the hotspot during the first spike_duration steps of each epoch.
        in_spike = ((qid - 1) % self.spike_every) < self.spike_duration
        hot_weights = [1.0 / len(hot)] * len(hot) if hot else []
        payload = ({"hot": hot, "hot_weights": hot_weights, "hot_mass": self.hot_mass} if in_spike else {},)
        return Query(qid=qid, payload=payload)



class BurstyHotsetQueryDist:
    """Bursty, regime-switching hotset generator with 80/20-style splits.

    We generate bursts of variable length. Each burst pins a hot set of clusters
    and uses hot_mass for the fraction of probes drawn from that hot set.
    Burst lengths are sampled from a (clipped) normal distribution by default,
    which yields irregular, non-sinusoidal changes in the active hot set.

    Optionally, an alternate mode can be sampled with probability `alt_prob`,
    allowing a bimodal workload regime (e.g., different hot-set size or hot mass).
    """
    def __init__(
        self,
        n_lists: int,
        n_hot: int = 2,
        hot_mass: float = 0.8,
        *,
        burst_mean: int = 200,
        burst_std: int = 50,
        burst_min: int = 20,
        burst_max: Optional[int] = 800,
        alt_prob: float = 0.0,
        alt_n_hot: Optional[int] = None,
        alt_hot_mass: Optional[float] = None,
        dist_switch_prob: float = 0.1,
        dist_start_mode: str = "zipf",
        hot_skew: float = 1.2,
        fixed_hotset: bool = False,
        fixed_hot_ids: Optional[list[int]] = None,
    ):
        self.n_lists = n_lists
        self.n_hot = max(1, int(n_hot))
        self.hot_mass = max(0.0, min(1.0, float(hot_mass)))

        self.burst_mean = max(1, int(burst_mean))
        self.burst_std = max(0, int(burst_std))
        self.burst_min = max(1, int(burst_min))
        self.burst_max = int(burst_max) if burst_max is not None else None

        self.alt_prob = max(0.0, min(1.0, float(alt_prob)))
        self.alt_n_hot = int(alt_n_hot) if alt_n_hot is not None else None
        self.alt_hot_mass = (
            max(0.0, min(1.0, float(alt_hot_mass)))
            if alt_hot_mass is not None
            else None
        )
        self.dist_switch_prob = max(0.0, min(1.0, float(dist_switch_prob)))
        self._dist_mode = dist_start_mode if dist_start_mode in ("zipf", "uniform") else "zipf"
        self._dist_id = 0
        self.hot_skew = max(0.0, float(hot_skew))
        self.fixed_hotset = bool(fixed_hotset or fixed_hot_ids is not None)

        self._fixed_hot: list[int] = []
        if fixed_hot_ids is not None:
            seen: set[int] = set()
            for x in fixed_hot_ids:
                try:
                    k = int(x)
                except (TypeError, ValueError):
                    continue
                if 0 <= k < self.n_lists and k not in seen:
                    self._fixed_hot.append(k)
                    seen.add(k)

        self._remaining = 0
        self._burst_id = 0
        self._burst_len = 0
        self._mode = "primary"
        self._hot: list[int] = []
        self._hot_weights: list[float] = []
        self._hot_mass = self.hot_mass

    def _sample_burst_len(self, rng: random.Random) -> int:
        if self.burst_std > 0:
            v = int(round(rng.gauss(self.burst_mean, self.burst_std)))
        else:
            v = int(round(self.burst_mean))
        v = max(self.burst_min, v)
        if self.burst_max is not None:
            v = min(self.burst_max, v)
        return max(1, v)

    def _start_new_burst(self, rng: random.Random) -> None:
        self._burst_id += 1
        self._burst_len = self._sample_burst_len(rng)
        self._remaining = self._burst_len

        use_alt = self.alt_prob > 0.0 and rng.random() < self.alt_prob
        if use_alt:
            self._mode = "alt"
            n_hot = self.alt_n_hot if self.alt_n_hot is not None else self.n_hot
            self._hot_mass = self.alt_hot_mass if self.alt_hot_mass is not None else self.hot_mass
        else:
            self._mode = "primary"
            n_hot = self.n_hot
            self._hot_mass = self.hot_mass

        target_hot = min(max(1, int(n_hot)), self.n_lists)
        if self.fixed_hotset:
            if not self._fixed_hot:
                hot = list(range(self.n_lists))
                rng.shuffle(hot)
                self._fixed_hot = hot[:target_hot]
            if len(self._fixed_hot) < target_hot:
                existing = set(self._fixed_hot)
                extension = [k for k in range(self.n_lists) if k not in existing]
                self._fixed_hot.extend(extension[: target_hot - len(self._fixed_hot)])
            self._hot = list(self._fixed_hot[:target_hot])
        else:
            hot = list(range(self.n_lists))
            rng.shuffle(hot)
            self._hot = hot[:target_hot]
        if self._hot:
            base = [1.0 / ((i + 1) ** self.hot_skew) for i in range(len(self._hot))]
            tot = sum(base)
            self._hot_weights = [x / tot for x in base]
        else:
            self._hot_weights = []

    def sample(self, rng: random.Random, qid: int) -> Query:
        burst_start = 0
        if self._remaining <= 0:
            self._start_new_burst(rng)
            burst_start = 1

        dist_switch = 0
        if self.dist_switch_prob > 0.0 and rng.random() < self.dist_switch_prob:
            self._dist_mode = "uniform" if self._dist_mode != "uniform" else "zipf"
            self._dist_id += 1
            dist_switch = 1

        payload = ({
            "hot": self._hot,
            "hot_weights": self._hot_weights,
            "hot_mass": self._hot_mass,
            "burst_id": self._burst_id,
            "burst_start": burst_start,
            "burst_mode": self._mode,
            "burst_len": self._burst_len,
            "burst_remaining": self._remaining,
            "dist_mode": self._dist_mode,
            "dist_switch": dist_switch,
            "dist_id": self._dist_id,
        },)

        self._remaining -= 1
        return Query(qid=qid, payload=payload)


class Workload:
    def __init__(self, rng: random.Random, dist: QueryDistribution):
        self.rng = rng
        self.dist = dist
        self._qid = 0

    def next(self) -> Query:
        self._qid += 1
        return self.dist.sample(self.rng, self._qid)

    def next_batch(self, n: int, *, lock_payload: bool = True) -> list[Query]:
        n = max(1, int(n))
        q0 = self.next()
        if n == 1:
            return [q0]

        if lock_payload:
            payload = q0.payload
            qs = [q0]
            for _ in range(n - 1):
                self._qid += 1
                qs.append(Query(qid=self._qid, payload=payload))
            return qs

        qs = [q0]
        for _ in range(n - 1):
            qs.append(self.next())
        return qs


def zipf_weights(n: int, s: float, rng: random.Random) -> list[float]:
    """
    Returns shuffled Zipf-ish weights normalized to sum=1.
    Higher s => more skew => more hotspots.
    """
    w = [1.0 / ((i + 1) ** s) for i in range(n)]
    rng.shuffle(w)
    tot = sum(w)
    return [x / tot for x in w]
