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
        return operations
