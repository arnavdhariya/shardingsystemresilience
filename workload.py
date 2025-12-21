import numpy as np

class Workload:
    def __init__(self, num_operations=1000, insert_ratio=0.5, delete_ratio=0.1, search_ratio=0.4, num_vectors=10000, vector_dim=128, num_namespaces=10):
        self.num_operations = num_operations
        self.insert_ratio = insert_ratio
        self.delete_ratio = delete_ratio
        self.search_ratio = search_ratio
        self.num_vectors = num_vectors
        self.vector_dim = vector_dim
        self.num_namespaces = num_namespaces

    def generate_workload(self):
        operations = []
        for _ in range(self.num_operations):
            op_type = np.random.choice(['insert', 'delete', 'search'], p=[self.insert_ratio, self.delete_ratio, self.search_ratio])
            if op_type == 'insert':
                vector_id = np.random.randint(0, self.num_vectors)
                vector = np.random.rand(self.vector_dim).astype('float32')
                namespace = np.random.randint(0, self.num_namespaces)
                operations.append({'type': 'insert', 'vector_id': vector_id, 'vector': vector, 'namespace': namespace})
            elif op_type == 'delete':
                vector_id = np.random.randint(0, self.num_vectors)
                namespace = np.random.randint(0, self.num_namespaces)
                operations.append({'type': 'delete', 'vector_id': vector_id, 'namespace': namespace})
            elif op_type == 'search':
                query_vector = np.random.rand(self.vector_dim).astype('float32')
                k = 10
                namespace = np.random.randint(0, self.num_namespaces)
                operations.append({'type': 'search', 'query_vector': query_vector, 'k': k, 'namespace': namespace})
    def generate_adversarial_workload(self, hot_namespace=0, skew_ratio=0.9):
        """Generates a skewed workload to simulate a 'hot shard' scenario."""
        operations = []
        
        # 1. Define the popular vector and plant it in the hot namespace
        popular_vector = np.random.rand(self.vector_dim).astype('float32')
        popular_vector_id = self.num_vectors + 1 # Ensure a unique ID
        operations.append({
            'type': 'insert',
            'vector_id': popular_vector_id,
            'vector': popular_vector,
            'namespace': hot_namespace
        })

        # 2. Populate other namespaces with some noise
        for i in range(self.num_namespaces * 10): # Add some data to other shards
            if self.num_namespaces > 1:
                ns = np.random.randint(0, self.num_namespaces)
                if ns == hot_namespace: continue
            else:
                ns = 0
            vector_id = np.random.randint(0, self.num_vectors)
            vector = np.random.rand(self.vector_dim).astype('float32')
            operations.append({'type': 'insert', 'vector_id': vector_id, 'vector': vector, 'namespace': ns})

        # 3. Generate the skewed workload
        for i in range(self.num_operations - len(operations)):
            # 90% of the time, search for the popular vector in the hot namespace
            if np.random.rand() < skew_ratio:
                operations.append({
                    'type': 'search',
                    'query_vector': popular_vector,
                    'k': 10,
                    'namespace': hot_namespace
                })
            else:
                # 10% of the time, do a random operation (background noise)
                op_type = np.random.choice(['insert', 'search'])
                ns = np.random.randint(0, self.num_namespaces)
                if op_type == 'search':
                    operations.append({
                        'type': 'search',
                        'query_vector': np.random.rand(self.vector_dim).astype('float32'),
                        'k': 10,
                        'namespace': ns
                    })
                else: # insert
                    operations.append({
                        'type': 'insert',
                        'vector_id': np.random.randint(0, self.num_vectors),
                        'vector': np.random.rand(self.vector_dim).astype('float32'),
                        'namespace': ns
                    })
    def generate_compound_attack_workload(self, hot_namespace=0, hot_shard_bloat_count=10000, skew_ratio=0.9):
        """Generates a severe hotspot by concentrating data and queries."""
        operations = []
        
        # 1. Bloat the hot shard with filler data
        for _ in range(hot_shard_bloat_count):
            operations.append({
                'type': 'insert',
                'vector_id': np.random.randint(0, self.num_vectors * 2),
                'vector': np.random.rand(self.vector_dim).astype('float32'),
                'namespace': hot_namespace
            })

        # 2. Add a small amount of data to other "cold" shards
        if self.num_namespaces > 1:
            for i in range(self.num_namespaces * 5):
                ns = np.random.randint(0, self.num_namespaces)
                if ns == hot_namespace: continue
                operations.append({
                    'type': 'insert',
                    'vector_id': np.random.randint(0, self.num_vectors),
                    'vector': np.random.rand(self.vector_dim).astype('float32'),
                    'namespace': ns
                })

        # 3. Plant the popular vector in the hot shard
        popular_vector = np.random.rand(self.vector_dim).astype('float32')
        popular_vector_id = self.num_vectors + 1 # Ensure a unique ID
        operations.append({
            'type': 'insert',
            'vector_id': popular_vector_id,
            'vector': popular_vector,
            'namespace': hot_namespace
        })

        # 4. Generate the skewed query flood
        remaining_ops = self.num_operations - len(operations)
        for _ in range(remaining_ops):
            if np.random.rand() < skew_ratio:
                # Target the popular vector in the bloated shard
                operations.append({
                    'type': 'search',
                    'query_vector': popular_vector,
                    'k': 10,
                    'namespace': hot_namespace
                })
            else:
                # Send a "cold" search to a random (likely small) shard
                ns = np.random.randint(0, self.num_namespaces)
                operations.append({
                    'type': 'search',
                    'query_vector': np.random.rand(self.vector_dim).astype('float32'),
                    'k': 10,
                    'namespace': ns
                })
        
        return operations, popular_vector_id
