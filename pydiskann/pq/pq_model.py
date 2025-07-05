#PQ 訓練與使用
from sklearn.cluster import KMeans
import numpy as np

class SimplePQ:
    def __init__(self, n_subvectors=2):
        self.n_subvectors = n_subvectors
        self.kmeans_list = []

    def fit(self, vectors, show_progress=False):
        d = vectors.shape[1]
        sub_dim = d // self.n_subvectors
        iterator = range(self.n_subvectors)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc='PQ 子量化器訓練')
        for i in iterator:
            subvectors = vectors[:, i*sub_dim:(i+1)*sub_dim]
            kmeans = KMeans(n_clusters=16, n_init=1, max_iter=50, random_state=42).fit(subvectors)
            self.kmeans_list.append(kmeans)

    def encode(self, vectors):
        d = vectors.shape[1]
        sub_dim = d // self.n_subvectors
        codes = []
        for i in range(self.n_subvectors):
            subvectors = vectors[:, i*sub_dim:(i+1)*sub_dim]
            codes.append(self.kmeans_list[i].predict(subvectors))
        return np.stack(codes, axis=1)

    def decode(self, codes):
        vectors = []
        for i, kmeans in enumerate(self.kmeans_list):
            centers = kmeans.cluster_centers_
            vectors.append(centers[codes[:, i]])
        return np.hstack(vectors)

    def distance(self, code1, code2):
        return np.sum(code1 != code2)
