import numpy as np
import json
import pickle
import mmap
from collections import OrderedDict

class DiskANNPersist:
    def __init__(self, dim=128, R=16, record_bytes=None):
        self.D = dim
        self.R = R
        self.record_size = 4 * (dim + R) if record_bytes is None else record_bytes

    def save_index(self, filepath, graph):
        with open(filepath, 'wb') as f:
            for idx in range(len(graph.nodes)):
                vec = graph.nodes[idx].vector.astype(np.float32)
                f.write(vec.tobytes())
                neighbors = list(graph.nodes[idx].neighbors)
                neighbors += [0] * (self.R - len(neighbors))
                f.write(np.array(neighbors[:self.R], dtype=np.uint32).tobytes())

    def save_meta(self, filepath, meta_dict):
        with open(filepath, 'w') as f:
            json.dump(meta_dict, f)

    def save_pq_codes(self, filepath, pq_codes):
        pq_codes.astype(np.uint8).tofile(filepath)

    def save_pq_codebook(self, filepath, pq_model):
        with open(filepath, 'wb') as f:
            pickle.dump(pq_model, f)

    def load_meta(self, filepath):
        with open(filepath, 'r') as f:
            return json.load(f)

    def load_pq_codes(self, filepath, num_nodes, n_subvectors):
        return np.fromfile(filepath, dtype=np.uint8).reshape((num_nodes, n_subvectors))

    def load_pq_codebook(self, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class MMapNodeReader:
    def __init__(self, filepath, dim=128, R=16, cache_size=1024):
        self.D = dim
        self.R = R
        self.record_size = 4 * (dim + R)
        self.file = open(filepath, 'rb')
        self.mmap_obj = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.cache = OrderedDict()
        self.cache_size = cache_size

    def get_node(self, node_id):
        if node_id in self.cache:
            self.cache.move_to_end(node_id)
            return self.cache[node_id]
        offset = node_id * self.record_size
        self.mmap_obj.seek(offset)
        vec = np.frombuffer(self.mmap_obj.read(4 * self.D), dtype=np.float32)
        neighbors = np.frombuffer(self.mmap_obj.read(4 * self.R), dtype=np.uint32)
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)
        self.cache[node_id] = (vec, neighbors)
        return vec, neighbors

    def close(self):
        self.mmap_obj.close()
        self.file.close()
