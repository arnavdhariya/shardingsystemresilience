# Codebase Documentation

## Overview

This repository simulates a sharded service where:
- logical clusters (`keys`) receive bursty query load,
- each cluster can be split into multiple physical shards (`slices`),
- shards are placed on servers,
- periodic rebalancing moves shards to reduce load imbalance,
- optional Step 1.5 growth adds new shards to the most popular clusters before rebalancing.

The core metric is Slicer-style imbalance:
- `imbalance = max(server_load) / mean(server_load)`
- `1.0` means perfectly balanced.

The simulator writes:
- CSV time series for plotting and metric verification.
- JSONL traces with per-step state, growth actions, rebalance actions, and before/after cluster transitions.

## Runtime Flow

1. `main.py` parses CLI arguments and builds the simulation objects.
2. `workload.py` generates query metadata for each timestep.
3. `search.py` converts query metadata into per-cluster probe counts/work units.
4. `sharding.py` routes cluster work across shards and servers.
5. `simulator.py` applies server cost, logs metrics, performs growth, then rebalances.
6. CSV and JSONL files are written for later analysis.

## Module Reference

### `main.py`

`main()`
- Parses all CLI arguments.
- Derives request defaults when they are omitted:
  - `requests_mean = n_servers * requests_per_server`
  - `requests_std = 20% of requests_mean`
  - `requests_min = requests_mean - 3 * requests_std` (lower bounded at `1`)
  - `requests_max = requests_mean + 3 * requests_std`
- Creates servers, workload generator, search planner, sharding state, and simulator.
- Runs the time-series simulation.
- Writes the output CSV.

Important CLI defaults:
- `seed=7`
- `policy=greedy`
- `steps=2000`
- `rebalance_every=50`
- `n_servers=100`
- `n_keys=100`
- `nprobe=20`
- `n_hot=2`
- `hot_mass=0.8`
- `hot_skew=1.2`
- `server_capacity=1e9`
- `cluster_shards_min=2`
- `cluster_shards_max=5`
- `requests_per_server=10000`
- `growth_on_rebalance=True`
- `growth_popular_fraction=0.02`
- `growth_frac=0.05`
- `growth_add=0.0`
- `growth_new_shards_max=3`

Output CSV columns:
- Core timing and balancing:
  - `t`, `latency`, `fanout`, `fanout_req_mean`, `fanout_req_var`, `fanout_req_min`, `fanout_req_max`, `fanout_req_samples`, `fanout_var`, `imbalance`, `imbalance_pre`
- Server identity:
  - `hottest_server`, `coldest_server`, `hottest_work_server`, `coldest_work_server`
- Server work summary:
  - `hottest_work`, `coldest_work`, `mean_work`
- Rebalance / growth flags:
  - `rebalance`, `growth_event`, `growth_added_total`, `growth_slices`, `growth_servers`, `growth_clusters`
- Workload shape:
  - `n_requests`, `work_units`, `workload`, `dist_mode`, `dist_switch`, `dist_id`
- Burst metadata:
  - `burst_id`, `burst_start`, `burst_mode`, `burst_len`, `burst_remaining`
- Hot-cluster metadata:
  - `hot_mass`, `hot_probe_ratio`, `hot_set`, `hot_weights`
- Cluster / slice load detail:
  - `cluster_loads`, `hottest_slice`, `coldest_slice`, `hottest_slice_load`, `coldest_slice_load`, `slice_loads`
- Per-server columns:
  - `work_s<sid>` and `keyspace_s<sid>` for every server

### `server.py`

`Server`
- Dataclass representing one server.
- Fields:
  - `sid`: server id
  - `capacity`: capacity used for imbalance normalization
  - `heat`: current accumulated heat
  - `cumulative_work`: all-time processed work
  - `window_work`: running processed work accumulator

`ServerModel`
- Protocol documenting the interface used by the simulator.
- `process(server, work)`: apply work and return latency contribution.
- `tick(server)`: timestep update.

`SimpleHeatModel`
- Minimal latency/heat model.

`SimpleHeatModel.__init__(alpha, beta, decay)`
- Stores the latency and heat coefficients.

`SimpleHeatModel.process(server, work)`
- Returns `work * (1 + alpha * heat)`.
- Increases `heat`, `cumulative_work`, and `window_work`.

`SimpleHeatModel.tick(server)`
- Applies heat decay each timestep.

### `search.py`

`WorkUnit`
- Dataclass: one planned unit of logical work.
- Fields:
  - `obj_id`: cluster/list id
  - `cost`: work amount for that cluster

`Query`
- Dataclass representing one logical query.

`SearchAlgorithm`
- Protocol for planners.

`IVFSearch`
- Planner that converts query metadata into cluster probes.
- Supports repeated probes (`allow_repeat=True`) so small hot sets can dominate.
- In both `uniform` and `zipf` modes, the hot mass still applies; only the cold tail changes.

`IVFSearch.__init__(...)`
- Stores planner shape (`n_lists`, `nprobe`) plus per-cluster cost and base Zipf weights.

`IVFSearch.plan(query)`
- Produces explicit `WorkUnit` objects for one query.
- If hot metadata is present:
  - routes `hot_mass * nprobe` toward the hot set,
  - uses `hot_weights` to skew load inside the hot set,
  - routes the rest through the cold pool using `uniform` or `zipf`.
- If no hot metadata is present:
  - falls back to the base weighted distribution.

`IVFSearch.plan_counts(n_requests, hot_cfg=None)`
- Fast aggregated version of `plan()`.
- Returns `{cluster_id: probe_count}` for an entire timestep.
- Used when the simulator batches many requests per timestep.

Helper methods:
- `_even_split(counts, items, n)`: equal integer apportionment.
- `_weighted_counts_for_items(counts, items, weights, n)`: weighted integer apportionment.
- `_uniform_counts(total_probes)`: equal allocation across all clusters.
- `_weighted_counts(total_probes)`: weighted allocation using global planner probabilities.
- `_weighted_counts_from_pool(counts, pool, total_probes)`: weighted cold-tail allocation from a subset.
- `_weighted_draw_from_pool(pool)`: single weighted draw using global probabilities.
- `_weighted_draw_from_items(items, weights)`: single weighted draw from explicit item weights.
- `_weighted_sample_without_replacement(items, weights, k)`: weighted no-repeat sampling.
- `_normalized_item_weights(items, raw_weights)`: validates and normalizes hot-set weights.
- `_weighted_draw()`: single weighted draw from the full global distribution.

### `workload.py`

`QueryDistribution`
- Protocol for query generators.

`RandomVectorQueryDist`
- Simple baseline distribution that emits payloads the planner mostly ignores.

`RandomVectorQueryDist.__init__(dim=4)`
- Stores vector payload width.

`RandomVectorQueryDist.sample(rng, qid)`
- Emits a random tuple payload.

`ClusterSpikeQueryDist`
- Deterministic, epoch-based hot-set generator.
- Useful as a simpler baseline than the bursty generator.

`ClusterSpikeQueryDist.__init__(...)`
- Stores spike timing, hot-set size, and hot mass.

`ClusterSpikeQueryDist.sample(rng, qid)`
- Selects a hot set per epoch and emits it only during the spike window.
- Includes equal `hot_weights`.

`BurstyHotsetQueryDist`
- Main workload generator.
- Creates non-sinusoidal bursts of variable length.
- Each burst defines:
  - a hot set,
  - `hot_mass`,
  - `hot_weights` (unequal popularity inside the hot set),
  - a current cold-tail mode (`uniform` or `zipf`).

`BurstyHotsetQueryDist.__init__(...)`
- Stores burst length parameters, alternate-mode options, distribution switching probability, and `hot_skew`.

`BurstyHotsetQueryDist._sample_burst_len(rng)`
- Samples a clipped normal burst length.

`BurstyHotsetQueryDist._start_new_burst(rng)`
- Starts a new burst, picks the hot clusters, and assigns Zipf-like `hot_weights`.

`BurstyHotsetQueryDist.sample(rng, qid)`
- Emits the current timestep’s hot-set metadata.

`Workload`
- Wrapper that increments query ids and delegates to a distribution.

`Workload.__init__(rng, dist)`
- Stores the RNG and distribution.

`Workload.next()`
- Returns one query.

`Workload.next_batch(n, lock_payload=True)`
- Returns a batch of queries.
- If `lock_payload=True`, all queries in the timestep share the same hot-set metadata.

`zipf_weights(n, s, rng)`
- Builds shuffled Zipf-like global weights for cluster popularity.

### `sharding.py`

`Slice`
- Dataclass for a physical shard.
- Fields:
  - `sid`: slice id
  - `lo`, `hi`: covered key range (`[lo, hi)`)
  - `size`: keyspace / churn weight
  - `load`: current timestep slice load estimate

`ShardState`
- Owns the current slice placement and all mutation logic.

`ShardState.__init__(servers, slices, slice_to_server, window=200, rng=None)`
- Stores servers, slices, placement, rolling history, and the round-robin server cursor used for Step 1.5 additions.

Placement / lookup methods:
- `slices_on(server_id)`: slice ids currently on a server.
- `adjacent_cold_pairs(server_id)`: merge candidates on a server.
- `keyspace(slice_id)`: size of one slice.
- `_new_slice_id(start=None)`: next unused slice id.
- `_rebuild_keymap()`: rebuilds:
  - `key_to_slice`: primary slice per cluster
  - `key_to_slices`: all slices per cluster

Load / imbalance methods:
- `imbalance()`: Slicer metric using current `slice.load` values if available.
- `hottest_server()`: server with highest current utilization.
- `coldest_server()`: server with lowest current utilization.
- `instantaneous_imbalance_from_tick(tick_work)`: one-timestep imbalance from raw server work.
- `record_server_work(server_id, work)`: append to rolling history.
- `reset_work_history()`: clear history after rebalance if desired.
- `update_slice_loads(slice_loads)`: overwrite each slice’s current load.
- `_recent_load(server_id)`: rolling-history server load.
- `_slice_estimated_loads()`: current server loads implied by `slice.load`.
- `_current_load(server_id)`: current load, preferring `slice.load`.
- `_current_util(server_id)`: current load divided by capacity.
- `_all_current_utils()`: current utilization map for all servers.
- `_recent_util(server_id)`: history-based utilization fallback.

Mutation methods:
- `apply_reassign(slice_id, dst_server)`: move one slice to another server.
- `apply_split(slice_id)`: split one slice into two:
  - range slices split by midpoint,
  - single-cluster slices split into two unequal same-cluster shards.
- `apply_merge(slice_a, slice_b)`: merge adjacent range slices or same-cluster shards on one server.
- `undo_last()`: reverse the last temporary mutation used during greedy search.

Routing / growth methods:
- `route_workunits(work_units)`: routes logical work to servers by first splitting load across all shards of a cluster.
- `distribute_key_cost_to_slices(key_id, cost)`: size-weighted load split across all shards of a cluster.
- `grow_slices_on_servers(...)`: legacy helper that increases existing slice sizes on selected servers.
- `add_popular_data_round_robin(popular_clusters, cluster_loads, ...)`: current Step 1.5 helper:
  - selects popular clusters,
  - computes disproportionate growth from cluster popularity,
  - creates new same-cluster shards,
  - places new shards round-robin across servers.
- `split_hot_slices(hot_keys)`: pre-rebalance split of the hottest shard for each hot cluster.

Validation / snapshot methods:
- `invariant_check(n_keys)`: ensures all clusters are covered and no server exceeds capacity.
- `snapshot(n_keys)`: returns full cluster/slice placement state for tracing.

Top-level algorithms:
- `greedy_slicer_balance(state, churn_budget, action_log=None)`
  - Weighted-move greedy rebalance.
  - Considers moves touching the hottest server:
    - reassign,
    - split,
    - merge on the coldest server.
  - Chooses the best `benefit / churn` move until budget is exhausted.
- `ilp_rebalance_keys_gurobi(...)`
  - Optional exact key-level rebalance using Gurobi.
- `rebuild_as_single_key_slices(state, n_keys, key_sizes, key_to_server_sid)`
  - Rebuilds placement as one slice per cluster.
- `build_initial_state(...)`
  - Builds the initial slice layout.
  - Current preferred mode is multi-shard-per-cluster:
    - each cluster starts with `cluster_shards_min..cluster_shards_max` unequal shards,
    - all initial shards for one cluster start on the same server,
    - later rebalancing can move only part of that cluster.

### `simulator.py`

`SimStats`
- Summary dataclass for non-time-series runs.

`_diff_state(before, after)`
- Computes:
  - added/removed/moved slices,
  - primary cluster migrations,
  - full cluster transitions (`slices_before/after`, `servers_before/after`, `shard_sizes_before/after`).

`Simulator`
- Coordinates the workload, planner, sharding state, and server model.

`Simulator.__init__(...)`
- Stores all components plus collected latencies.

`Simulator.step()`
- One simple single-query step.

`Simulator.rebalance_greedy(churn_budget, action_log=None)`
- Runs the greedy rebalancer.

`Simulator.rebalance_ilp_gurobi(...)`
- Estimates per-cluster load, solves the ILP, and rebuilds placement.

`Simulator.run_timeseries(...)`
- Main experimental driver.
- Per timestep it:
  - samples the request count,
  - generates a batch,
  - aggregates planner counts,
  - computes `cluster_loads`,
  - computes `slice_loads`,
  - records server work,
  - emits step traces,
  - optionally runs Step 1.5 growth for the most popular clusters,
  - optionally splits hot shards,
  - rebalances,
  - emits rebalance traces,
  - writes one CSV row.

Trace event types:
- `state`: full placement snapshot
- `step`: per-timestep workload, plan, and server metrics
- `growth`: Step 1.5 additions before rebalance
- `rebalance`: action list plus before/after state and diff

`Simulator.run(...)`
- Shorter non-CSV runner that only returns `SimStats`.

`Simulator.summarize()`
- Computes average latency, p95 latency, final imbalance, hottest server, and max heat.

### `collect_data.py`

Purpose:
- Batch-sweep helper for repeated ablation runs.

Fixed parameters in this file:
- `OUTDIR`
- `POLICY`
- `REPLICATES`
- `BASE_SEED`
- `N_KEYS`
- `NPROBE`
- `ZIPF_S`
- `SERVER_CAPACITY`
- `INITIAL_SLICES`
- `CHURN_BUDGET`
- `MIGRATION_K`
- `GUROBI_TIME_LIMIT`
- `GUROBI_MIP_GAP`
- `SERVER_COUNTS`
- `REBALANCE_EVERY`
- `TOTAL_STEPS`
- `BASE_SERVERS`
- `BASE_REBALANCE`
- `BASE_STEPS`

Functions:
- `ensure_dir(d)`: create output directory.
- `run_sim(...)`: run one ablation configuration and return summary stats.
- `write_csv(path, rows)`: write rows to a CSV file.
- `sweep(name, configs)`: execute one sweep across configs and replicates.
- `main()`: runs all configured sweeps.

### `collect_data_trace.py`

Purpose:
- Canonical full-fidelity runner for long traces.
- Runs a continuous 2000-step simulation for both `zipf` and `uniform` cold-tail modes.
- Logs full per-timestep CSV plus full JSONL trace snapshots/actions.

Why this runner exists:
- The old step-by-step loop pattern can break periodic rebalancing semantics.
- This runner uses one continuous `run_timeseries()` call per scenario so `rebalance_every` works correctly.

Key components:
- `RunnerConfig`: fixed parameters for the 2000-step run.
- `RunnerConfig.request_volume()`: derives request volume from `n_servers * requests_per_server`.
- `_build_sim(...)`: builds one simulator instance for one workload mode.
- `_write_rows_csv(...)`: writes all metrics columns, including per-request fan-out stats.
- `run_scenario(...)`: executes one full scenario (`zipf` or `uniform`).
- `main()`: runs both scenarios and writes `run_config_full_2000.json`.

### `plot_paper_figures.py`

Purpose:
- Build paper-style comparison figures from one Zipf dataset and one Uniform dataset.

Functions:
- `_ensure_mplconfig()`: points Matplotlib and font caches at writable repo-local paths.
- `set_vldb_style()`: applies publication-style rcParams inspired by database systems papers.
- `_parse_json_cell(value, default)`: safe JSON decode for CSV cells.
- `load_timeseries(path)`: reads a CSV and derives plotting series, including the 80/20 hot-vs-cold split.
  - Prefers per-request fan-out metrics when present.
- `_rebalance_times(data)`: timesteps where rebalancing occurred.
- `_add_rebalance_lines(ax, rebalance_steps, label)`: draws dashed rebalance markers.
- `_save_figure(fig, base_path)`: writes both PNG and PDF.
- `plot_workload_imbalance(zipf, uniform, outdir)`: request volume, hot/cold workload split, and imbalance figure.
- `plot_fanout(zipf, uniform, outdir)`: fan-out plus rolling variance figure.
- `_rolling_candles(values, window)`: rolling min/max and mean ± std for candlesticks.
- `_draw_candles(...)`: renders one variance-aware candlestick panel.
- `plot_fanout_candles(zipf, uniform, outdir, window)`: candlestick figure for fan-out variation.
- `main()`: CLI wrapper.

## Current Growth Logic (Step 1.5)

Current behavior when `growth_on_rebalance=True`:

1. Compute actual per-cluster load for the timestep (`cluster_loads`).
2. Rank clusters by descending load.
3. Choose the top `growth_popular_k` clusters, or the top `growth_popular_fraction` if `growth_popular_k=0`.
4. For each selected cluster:
   - compute disproportionate growth from current cluster size and load share,
   - create `1..growth_new_shards_max` new shards (more popular clusters tend to get more shards),
   - split the added size unevenly across those new shards,
   - place the new shards on servers round-robin.
5. Update `key_sizes` so later balancing remains consistent.
6. Run hot-shard splitting and rebalancing on the updated state.

This means cluster size can diverge over time:
- a popular cluster at `t=10` can receive new shards,
- by `t=100`, repeatedly popular clusters become materially larger and more fragmented than cold clusters.

## How To Recompute Metrics Yourself

From the CSV:

- Work imbalance from actual server work:
  - `max(work_s*) / mean(work_s*)`
- Pre-rebalance imbalance:
  - compare against `imbalance_pre`
- Post-rebalance predicted imbalance:
  - compare against `imbalance`
- Total timestep workload:
  - `workload`
- Per-cluster workload:
  - parse `cluster_loads` JSON
- Per-slice workload:
  - parse `slice_loads` JSON
- Fan-out trend:
  - `fanout`
- Growth schedule:
  - `growth_event`, `growth_slices`, `growth_clusters`, `growth_added_total`

From the JSONL trace:

- `growth.actions` shows exactly which new shard was created, for which cluster, on which server, and by how much.
- `rebalance.actions` shows each split/reassign/merge action.
- `rebalance.cluster_transitions` shows the cluster-level before/after shard layout.

## Practical Entry Points

For standard time-series data:
- `python3 main.py`

For a controlled run with trace:
- `python3 main.py --trace-jsonl trace.jsonl --out-csv timeseries.csv`

For canonical 2000-step full trace runs:
- `python3 collect_data_trace.py`

For parameter sweeps:
- `python3 collect_data.py`

For paper-style comparison plots:
- `python3 plot_paper_figures.py`
