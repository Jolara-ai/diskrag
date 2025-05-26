import faiss
import numpy as np

class FastPQ:
    def __init__(self, n_subvectors=8):
        self.n_subvectors = n_subvectors
        self.centroids = []  # 不儲存 kmeans，僅儲存 centers

    def fit(self, vectors):
        d = vectors.shape[1]
        sub_dim = d // self.n_subvectors
        self.centroids = []
        for i in range(self.n_subvectors):
            subv = vectors[:, i*sub_dim:(i+1)*sub_dim].astype(np.float32)
            kmeans = faiss.Kmeans(d=sub_dim, k=16, niter=20, verbose=False)
            kmeans.train(subv)
            self.centroids.append(kmeans.centroids.copy())  # copy numpy array

    def encode(self, vectors):
        d = vectors.shape[1]
        sub_dim = d // self.n_subvectors
        codes = []
        for i in range(self.n_subvectors):
            subv = vectors[:, i*sub_dim:(i+1)*sub_dim].astype(np.float32)
            cent = self.centroids[i]
            dists = np.linalg.norm(subv[:, np.newaxis, :] - cent[np.newaxis, :, :], axis=2)
            codes.append(np.argmin(dists, axis=1))
        return np.stack(codes, axis=1)

    def decode(self, codes):
        parts = [self.centroids[i][codes[:, i]] for i in range(self.n_subvectors)]
        return np.hstack(parts)

    def distance(self, code1, code2):
        return np.sum(code1 != code2)
