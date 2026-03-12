# Sharding System Resilience

This repository is a Python simulation of bursty sharded workloads with Slicer-style shard movement.

The current model supports:
- Bursty 80/20 query routing to hot clusters.
- Two cold-tail modes: `uniform` and `zipf`.
- Clusters that contain multiple unequal-sized shards.
- Rebalancing that moves shards, splits hot shards, and can merge adjacent cold shards.
- Step 1.5 data additions that add new shards to the most popular clusters, placed round-robin across servers.
- Detailed CSV and JSONL trace output so you can recompute imbalance, fan-out, workload, and cluster transitions yourself.

Main entry point:
- `python3 main.py`

Batch sweep helper:
- `python3 collect_data.py`

Full-fidelity 2000-step runner (zipf + uniform, narrative TXT log + full JSONL trace):
- `python3 collect_data_trace.py`
  - writes human-readable timestep logs as plain text with full-sentence explanations and explicit imbalance calculations.
  - also writes detailed JSONL trace for machine-parsable transitions.

Paper-style plotting helper:
- `python3 plot_paper_figures.py`

Detailed codebase and method documentation:
- `CODEBASE.md`
