import numpy as np

class SimulatedANNIndex:
    """A simulated ANN index that uses brute-force search."""
    def __init__(self, vector_dim):
        self.vector_dim = vector_dim
        self.vectors = {}
        self.is_trained = True # No training needed for this simple index

    @property
    def ntotal(self):
        return len(self.vectors)

    def train(self, data):
        # No training needed for a brute-force index
        print("Skipping training for simulated index.")
        pass

    def add_with_ids(self, vectors, ids):
        for i, vector_id in enumerate(ids):
            self.vectors[vector_id] = vectors[i]

    def remove(self, vector_id):
        if vector_id in self.vectors:
            del self.vectors[vector_id]

    def reconstruct(self, vector_id):
        return self.vectors.get(vector_id)

    def search(self, query_vector, k):
        if not self.vectors:
            return np.array([[]]), np.array([[]])

        ids = list(self.vectors.keys())
        vectors = np.array(list(self.vectors.values()))
        
        # Calculate L2 distance
        distances = np.linalg.norm(vectors - query_vector, axis=1)
        
        # Get top k results
        sorted_indices = np.argsort(distances)
        top_k_indices = sorted_indices[:k]
        
        top_distances = distances[top_k_indices]
        top_ids = np.array(ids)[top_k_indices]
        
        return np.array([top_distances]), np.array([top_ids])


class ShardingPolicy:
    def __init__(self, vector_dim=128):
        self.vector_dim = vector_dim
        self.shards = {}
        self.id_map = {}

    def create_shard(self, namespace):
        if namespace not in self.shards:
            self.shards[namespace] = SimulatedANNIndex(self.vector_dim)
            self.id_map[namespace] = set()

    def train(self, namespace, data):
        if namespace in self.shards:
            self.shards[namespace].train(data)

    def insert(self, vector_id, vector, namespace):
        if namespace in self.shards:
            self.shards[namespace].add_with_ids(vector.reshape(1, -1), [vector_id])
            self.id_map[namespace].add(vector_id)

    def delete(self, vector_id, namespace):
        if namespace in self.shards and vector_id in self.id_map[namespace]:
            self.shards[namespace].remove(vector_id)
            self.id_map[namespace].remove(vector_id)

    def search(self, query_vector, k, namespace):
        if namespace in self.shards:
            D, I = self.shards[namespace].search(query_vector.reshape(1, -1), k)
            return [{'shard_id': namespace, 'distances': D, 'indices': I}]
        return []

    def _reconstruct_vectors_and_ids(self, namespace):
        vectors = []
        ids = []
        if namespace in self.shards:
            index = self.shards[namespace]
            for vector_id in self.id_map.get(namespace, set()):
                vec = index.reconstruct(vector_id)
                if vec is not None:
                    vectors.append(vec)
                    ids.append(vector_id)
        return np.array(vectors), np.array(ids)

    def merge(self, namespace_1, namespace_2, new_namespace):
        print(f"Merging namespaces {namespace_1} and {namespace_2} into {new_namespace}...")
        vectors_1, ids_1 = self._reconstruct_vectors_and_ids(namespace_1)
        vectors_2, ids_2 = self._reconstruct_vectors_and_ids(namespace_2)

        all_vectors = np.concatenate((vectors_1, vectors_2)) if vectors_1.size > 0 and vectors_2.size > 0 else (vectors_1 if vectors_1.size > 0 else vectors_2)
        all_ids = np.concatenate((ids_1, ids_2)) if ids_1.size > 0 and ids_2.size > 0 else (ids_1 if ids_1.size > 0 else ids_2)
        
        if all_vectors.size > 0:
            self.create_shard(new_namespace)
            self.train(new_namespace, all_vectors)
            self.shards[new_namespace].add_with_ids(all_vectors, all_ids)
            self.id_map[new_namespace] = set(all_ids)

        for ns in [namespace_1, namespace_2]:
            if ns in self.shards:
                del self.shards[ns]
                del self.id_map[ns]

    def split(self, namespace, new_namespace_1, new_namespace_2):
        print(f"Splitting namespace {namespace} into {new_namespace_1} and {new_namespace_2}...")
        vectors, ids = self._reconstruct_vectors_and_ids(namespace)

        if vectors.size > 0:
            midpoint = len(vectors) // 2
            vectors_1, ids_1 = vectors[:midpoint], ids[:midpoint]
            vectors_2, ids_2 = vectors[midpoint:], ids[midpoint:]

            for ns, vecs, i in [(new_namespace_1, vectors_1, ids_1), (new_namespace_2, vectors_2, ids_2)]:
                if vecs.size > 0:
                    self.create_shard(ns)
                    self.train(ns, vecs)
                    self.shards[ns].add_with_ids(vecs, i)
                    self.id_map[ns] = set(i)

        if namespace in self.shards:
            del self.shards[namespace]
            del self.id_map[namespace]