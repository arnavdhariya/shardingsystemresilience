from workload import Workload
from sharding_policy import ShardingPolicy
import numpy as np

def main():
    # Initialize workload and sharding policy
    workload = Workload(num_operations=1000, num_vectors=1000, vector_dim=128, num_namespaces=10)
    sharding_policy = ShardingPolicy(vector_dim=128)

    # Generate workload
    operations = workload.generate_workload()

    # Collect training data
    training_data = {}
    for op in operations:
        if op['type'] == 'insert':
            namespace = op['namespace']
            if namespace not in training_data:
                training_data[namespace] = []
            training_data[namespace].append(op['vector'])

    # Create and train indexes
    for namespace, data in training_data.items():
        if data:
            sharding_policy.create_shard(namespace)
            sharding_policy.train(namespace, np.array(data))


    # Process operations
    insert_count = 0
    delete_count = 0
    search_count = 0

    for op in operations:
        if op['type'] == 'insert':
            sharding_policy.insert(op['vector_id'], op['vector'], op['namespace'])
            insert_count += 1
        elif op['type'] == 'delete':
            sharding_policy.delete(op['vector_id'], op['namespace'])
            delete_count += 1
        elif op['type'] == 'search':
            sharding_policy.search(op['query_vector'], op['k'], op['namespace'])
            search_count += 1

    # Print statistics
    print(f"Simulation complete.")
    print(f"Total operations: {len(operations)}")
    print(f"Inserts: {insert_count}")
    print(f"Deletes: {delete_count}")
    print(f"Searches: {search_count}")

    for namespace, shard in sharding_policy.shards.items():
        print(f"Namespace {namespace} size: {shard.ntotal}")

    # Demonstrate merge and split
    sharding_policy.merge(0, 1, 10)
    sharding_policy.split(2, 11, 12)

    print("\nAfter merge and split:")
    for namespace, shard in sharding_policy.shards.items():
        print(f"Namespace {namespace} size: {shard.ntotal}")


if __name__ == "__main__":
    main()
