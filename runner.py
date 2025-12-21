import numpy as np
import time
from workload import Workload
from sharding_policy import ShardingPolicy

def run_test(config):
    """Runs a single test configuration and reports results."""
    print(f"--- Running Test: {config['name']} ---")
    
    # Initialize workload and sharding policy
    workload = Workload(
        num_operations=config.get('num_operations', 1000),
        insert_ratio=config.get('insert_ratio', 0.5),
        delete_ratio=config.get('delete_ratio', 0.1),
        search_ratio=config.get('search_ratio', 0.4),
        num_vectors=config.get('num_vectors', 1000),
        vector_dim=config.get('vector_dim', 128),
        num_namespaces=config.get('num_namespaces', 10)
    )
    sharding_policy = ShardingPolicy(vector_dim=workload.vector_dim)

    # 1. Generate workload
    start_time = time.time()
    operations = workload.generate_workload()
    generation_time = time.time() - start_time
    print(f"Workload generation time: {generation_time:.4f} seconds")

    # 2. Create and train indexes
    training_data = {}
    for op in operations:
        if op['type'] == 'insert':
            namespace = op['namespace']
            if namespace not in training_data:
                training_data[namespace] = []
            training_data[namespace].append(op['vector'])

    start_time = time.time()
    for namespace, data in training_data.items():
        if data:
            sharding_policy.create_shard(namespace)
            sharding_policy.train(namespace, np.array(data))
    training_time = time.time() - start_time
    print(f"Index creation and training time: {training_time:.4f} seconds")

    # 3. Process operations
    insert_count = 0
    delete_count = 0
    search_count = 0
    
    op_timings = {'insert': [], 'delete': [], 'search': []}

    for op in operations:
        op_start_time = time.time()
        if op['type'] == 'insert':
            sharding_policy.insert(op['vector_id'], op['vector'], op['namespace'])
            insert_count += 1
        elif op['type'] == 'delete':
            sharding_policy.delete(op['vector_id'], op['namespace'])
            delete_count += 1
        elif op['type'] == 'search':
            sharding_policy.search(op['query_vector'], op['k'], op['namespace'])
            search_count += 1
        op_timings[op['type']].append(time.time() - op_start_time)

    total_op_time = sum(sum(times) for times in op_timings.values())
    
    # --- Report Results ---
    print("\n[Results]")
    print(f"Total operations processed: {len(operations)}")
    print(f"  - Inserts: {insert_count}")
    print(f"  - Deletes: {delete_count}")
    print(f"  - Searches: {search_count}")
    
    print(f"\nTotal operation processing time: {total_op_time:.4f} seconds")
    for op_type, times in op_timings.items():
        if times:
            avg_time = sum(times) / len(times)
            print(f"  - Average '{op_type}' time: {avg_time * 1000:.4f} ms")

    print("\nFinal Shard Sizes:")
    for namespace in sorted(sharding_policy.shards.keys()):
        shard = sharding_policy.shards[namespace]
        print(f"  - Namespace {namespace}: {shard.ntotal} vectors")
    
    print(f"--- Test Complete: {config['name']} ---\n")


def run_compound_attack_test(config):
    """Runs the compound attack workload and reports skewed results."""
    print(f"--- Running Test: {config['name']} ---")

    hot_namespace = config.get('hot_namespace', 0)
    
    # Initialize workload and sharding policy
    workload = Workload(
        num_operations=config.get('num_operations', 15000),
        num_vectors=config.get('num_vectors', 1000),
        vector_dim=config.get('vector_dim', 128),
        num_namespaces=config.get('num_namespaces', 10)
    )
    sharding_policy = ShardingPolicy(vector_dim=workload.vector_dim)

    # 1. Generate compound attack workload
    start_time = time.time()
    operations, popular_vector_id = workload.generate_compound_attack_workload(hot_namespace=hot_namespace)
    generation_time = time.time() - start_time
    print(f"Compound attack workload generation time: {generation_time:.4f} seconds")

    # 2. Create and train indexes
    training_data = {}
    for op in operations:
        if op['type'] == 'insert':
            namespace = op['namespace']
            if namespace not in training_data:
                training_data[namespace] = []
            training_data[namespace].append(op['vector'])

    for namespace, data in training_data.items():
        if data:
            sharding_policy.create_shard(namespace)
            sharding_policy.train(namespace, np.array(data))

    # 3. Process operations and time hot vs. cold searches
    op_timings = {'insert': [], 'hot_search': [], 'cold_search': []}
    hot_searches = 0
    cold_searches = 0
    
    for op in operations:
        op_start_time = time.time()
        if op['type'] == 'insert':
            sharding_policy.insert(op['vector_id'], op['vector'], op['namespace'])
            op_timings['insert'].append(time.time() - op_start_time)
        elif op['type'] == 'search':
            results = sharding_policy.search(op['query_vector'], op['k'], op['namespace'])
            duration = time.time() - op_start_time
            
            # A "hot" search is any search directed at the hot_namespace
            if op['namespace'] == hot_namespace:
                op_timings['hot_search'].append(duration)
                hot_searches += 1
            else:
                op_timings['cold_search'].append(duration)
                cold_searches += 1

    # --- Report Results ---
    print("\n[Adversarial Results]")
    print(f"Targeted Hot Namespace: {hot_namespace}")
    print(f"Hot Searches (to bloated shard): {hot_searches}")
    print(f"Cold Searches (to small shards): {cold_searches}")

    print("\nPerformance Breakdown:")
    for op_type, times in op_timings.items():
        if times:
            avg_time = sum(times) / len(times)
            print(f"  - Average '{op_type}' time: {avg_time * 1000:.4f} ms")

    print("\nFinal Shard Sizes:")
    for namespace in sorted(sharding_policy.shards.keys()):
        shard = sharding_policy.shards[namespace]
        size_info = f"{shard.ntotal} vectors"
        if namespace == hot_namespace:
            size_info += " (HOT SHARD)"
        print(f"  - Namespace {namespace}: {size_info}")
        
    print(f"--- Test Complete: {config['name']} ---\n")


def main():
    """Defines and runs a series of test workloads."""
    
    adversarial_test = {
        "name": "Adversarial Attack: Compound Hotspot",
        "num_operations": 15000, # Includes ~10k inserts for bloating
        "num_vectors": 5000,
        "vector_dim": 128,
        "num_namespaces": 5, # Fewer cold shards to make the distinction clear
        "hot_namespace": 0
    }
        
    run_compound_attack_test(adversarial_test)

if __name__ == "__main__":
    main()


