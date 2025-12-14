# Sharding System Resilience Simulation

## Overview

This project provides a lightweight, pure Python simulation framework for exploring sharding policies in similarity search systems. It allows you to experiment with different sharding schemes, workload patterns, and basic Approximate Nearest Neighbor (ANN) functionalities in a controlled environment, without relying on external ANN libraries like Faiss for the core index logic.

The goal is to provide an abstract and extensible platform to understand how sharding strategies impact data distribution and query performance under various operational scenarios (inserts, deletes, searches, merges, and splits).

## Features

*   **Pure Python Simulated ANN Index:** A basic, brute-force ANN index implementation that stores vectors and performs similarity searches. This allows for clear understanding of the underlying sharding logic without external dependencies.
*   **Namespace-based Sharding Policy:** Inspired by systems like Pinecone, this policy organizes vectors into logical "namespaces," each acting as an independent shard.
*   **Workload Generation:** A configurable workload generator that produces a sequence of insert, delete, and search operations, allowing for custom workload distributions.
*   **Dynamic Shard Management:**
    *   **Insert & Delete:** Operations to add and remove vectors within specific namespaces.
    *   **Merge:** Combines two existing namespaces into a new, consolidated namespace.
    *   **Split:** Divides an existing namespace into two new namespaces.
*   **Modular Design:** The separation of `Workload`, `ShardingPolicy`, and `Server` classes facilitates easy modification and experimentation with different components.

## Getting Started

Follow these steps to set up and run the simulation:

### Prerequisites

*   Python 3.x
*   `numpy` (for vector operations)

### Setup

1.  **Navigate to the project directory:**
    ```bash
    cd shardingsystemresilience
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install numpy
    ```

### Running the Simulation

Execute the `server.py` script to start the simulation:

```bash
venv/bin/python3 server.py
```

You will see output detailing the operations performed, the state of the namespaces, and demonstrations of merge and split operations.

## Project Structure

*   `server.py`: The main entry point of the simulation. It initializes the workload and sharding policy, processes operations, and prints statistics.
*   `sharding_policy.py`: Defines the `ShardingPolicy` class, which manages the creation, training, insertion, deletion, searching, merging, and splitting of simulated ANN indexes across different namespaces. It includes the `SimulatedANNIndex` class, a pure Python replacement for external ANN libraries.
*   `workload.py`: Contains the `Workload` class, responsible for generating a sequence of operations (insert, delete, search) with configurable ratios and vector dimensions.

## Extensibility and Future Work

This simulation framework is designed to be easily extensible. Here are some ideas for further exploration:

*   **Different Sharding Schemes:** Implement other sharding policies (e.g., consistent hashing, range-based sharding).
*   **Advanced ANN Algorithms:** Integrate different simulated ANN algorithms (e.g., k-d trees, LSH) into the `SimulatedANNIndex` or create new index types.
*   **More Complex Workloads:** Introduce skewed access patterns, temporal locality, or varying query loads in the `Workload` generator.
*   **Cost Model Integration:** Add a cost model to quantify the resource consumption (e.g., CPU, memory, network I/O) of different operations and policies.
*   **Visualization:** Develop tools to visualize shard distribution, workload patterns, and performance metrics.
*   **Automated Scaling Decisions:** Implement logic within the `ShardingPolicy` to trigger merges or splits based on predefined thresholds (e.g., shard size, query load).