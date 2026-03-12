# server.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Server:
    sid: int
    capacity: float
    heat: float = 0.0
    cumulative_work: float = 0.0
    window_work: float = 0.0


class ServerModel(Protocol):
    def process(self, server: Server, work: float) -> float:
        """Return latency contribution from this server for this work."""
        ...

    def tick(self, server: Server) -> None:
        """Per-timestep update (e.g., heat decay)."""
        ...


class SimpleHeatModel:
    """
    Minimal model:
      latency = work * (1 + alpha * heat)
      heat += beta * work
      heat *= (1 - decay) each tick
    """
    def __init__(self, alpha: float = 0.0005, beta: float = 0.01, decay: float = 0.02):
        self.alpha = alpha
        self.beta = beta
        self.decay = decay

    def process(self, server: Server, work: float) -> float:
        latency = work * (1.0 + self.alpha * server.heat)
        server.heat += self.beta * work
        server.cumulative_work += work
        server.window_work += work
        return latency

    def tick(self, server: Server) -> None:
        server.heat *= (1.0 - self.decay)
