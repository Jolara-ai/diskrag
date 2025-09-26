import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, List, Optional

class DiskANNPQ:
    """
    DiskANN 論文標準的 Product Quantization 實現 (v3 - 移除標準化)
    
    核心修正:
    - 移除了內部數據標準化 (Standardization) 功能，以確保 PQ 近似距離
      與原始空間的 L2 距離在尺度上具有可比性，這是解決 recall=0 問題的關鍵。
    
    核心設計原則:
    - n_centroids 固定為 256: 每個子向量編碼恰好用 1 byte (uint8) 表示
    - n_subvectors 作為主要調節參數: 平衡精度和效率
    - 向量化計算: 利用 NumPy 廣播機制提升性能
    - 動態參數調整: 根據數據規模自適應調整訓練參數
    """
    
    def __init__(self, n_subvectors: int = 8, n_centroids: int = 256):
        if n_centroids != 256:
            print(f"⚠️  警告: n_centroids 已從 {n_centroids} 調整為 256 (最佳實踐)")
            n_centroids = 256
            
        self.n_subvectors = n_subvectors
        self.n_centroids = n_centroids
        self.kmeans_list = []
        self.sub_dim = 0
        self.is_fitted = False
        
    def _get_adaptive_kmeans_params(self, n_samples: int) -> dict:
        """根據數據規模動態調整 KMeans 參數"""
        if n_samples < 10000:
            return {'n_init': 10, 'max_iter': 300}
        elif n_samples < 100000:
            return {'n_init': 5, 'max_iter': 300}
        else:
            return {'n_init': 3, 'max_iter': 200}
    
    def fit(self, vectors: np.ndarray, show_progress: bool = False) -> None:
        """
        訓練 PQ 模型
        
        Args:
            vectors: 訓練向量 [N, D] (應為 float32)
            show_progress: 是否顯示進度條
        """
        n_vectors, d = vectors.shape
        
        if d % self.n_subvectors != 0:
            raise ValueError(f"向量維度 {d} 必須能被子向量數量 {self.n_subvectors} 整除")
            
        self.sub_dim = d // self.n_subvectors
        
        if n_vectors < self.n_centroids:
            raise ValueError(f"訓練數據量 {n_vectors} 不足，至少需要 {self.n_centroids} 個向量")
        
        print(f"訓練 PQ 模型: {self.n_subvectors}×{self.n_centroids} (子向量×聚類中心)")
        
        iterator = range(self.n_subvectors)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc='PQ 子量化器訓練')
            except ImportError:
                print("提示: 安裝 tqdm 以顯示進度條")
            
        self.kmeans_list = []
        for i in iterator:
            start_idx = i * self.sub_dim
            end_idx = (i + 1) * self.sub_dim
            subvectors = vectors[:, start_idx:end_idx]
            
            kmeans_params = self._get_adaptive_kmeans_params(n_vectors)
            kmeans = KMeans(
                n_clusters=self.n_centroids,
                n_init=kmeans_params['n_init'],
                max_iter=kmeans_params['max_iter'],
                random_state=42 + i,
                algorithm='lloyd',
                init='k-means++'
            )
            kmeans.fit(subvectors)
            self.kmeans_list.append(kmeans)
            
        self.is_fitted = True

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        將向量編碼為 PQ 碼
        
        Args:
            vectors: 待編碼向量 [N, D]
            
        Returns:
            PQ 碼 [N, n_subvectors] (uint8 類型)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未訓練，請先調用 fit() 方法")
            
        n_vectors, d = vectors.shape
        codes = np.zeros((n_vectors, self.n_subvectors), dtype=np.uint8)
        
        for i in range(self.n_subvectors):
            start_idx = i * self.sub_dim
            end_idx = (i + 1) * self.sub_dim
            subvectors = vectors[:, start_idx:end_idx]
            codes[:, i] = self.kmeans_list[i].predict(subvectors)
            
        return codes
        
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        將 PQ 碼解碼為近似向量
        
        Args:
            codes: PQ 碼 [N, n_subvectors]
            
        Returns:
            解碼向量 [N, D]
        """
        if not self.is_fitted:
            raise ValueError("模型尚未訓練")
            
        n_vectors = codes.shape[0]
        d = self.sub_dim * self.n_subvectors
        decoded = np.zeros((n_vectors, d), dtype=np.float32)
        
        for i in range(self.n_subvectors):
            start_idx = i * self.sub_dim
            end_idx = (i + 1) * self.sub_dim
            centroids = self.kmeans_list[i].cluster_centers_
            decoded[:, start_idx:end_idx] = centroids[codes[:, i]]
            
        return decoded
        
    def compute_distance_table(self, query_vector: np.ndarray) -> np.ndarray:
        """
        計算查詢向量與所有聚類中心的距離表
        
        Args:
            query_vector: 查詢向量 [D]
            
        Returns:
            距離表 [n_subvectors, n_centroids] (平方距離)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未訓練")

        distance_table = np.zeros((self.n_subvectors, self.n_centroids), dtype=np.float32)
        
        for i in range(self.n_subvectors):
            start_idx = i * self.sub_dim
            end_idx = (i + 1) * self.sub_dim
            query_sub = query_vector[start_idx:end_idx]
            
            centroids = self.kmeans_list[i].cluster_centers_
            diff = centroids - query_sub[np.newaxis, :]
            distance_table[i, :] = np.sum(diff * diff, axis=1)
                
        return distance_table
        
    def asymmetric_distance_sq(self, codes: np.ndarray, distance_table: np.ndarray) -> np.ndarray:
        """使用預計算的距離表快速計算 ADC 平方距離"""
        n_vectors = codes.shape[0]
        distances_sq = np.zeros(n_vectors, dtype=np.float32)
        
        for i in range(self.n_subvectors):
            distances_sq += distance_table[i, codes[:, i]]
            
        return distances_sq
        
    def asymmetric_distance(self, codes: np.ndarray, distance_table: np.ndarray) -> np.ndarray:
        """使用預計算的距離表快速計算 ADC 距離"""
        distances_sq = self.asymmetric_distance_sq(codes, distance_table)
        return np.sqrt(distances_sq)
        
    def estimate_selectivity(self, vectors: np.ndarray, sample_size: int = 1000) -> float:
        """估算PQ選擇性（碼本區分度）"""
        sample_size = min(sample_size, len(vectors))
        sample_indices = np.random.choice(len(vectors), sample_size, replace=False)
        sample_vectors = vectors[sample_indices]
        
        codes = self.encode(sample_vectors)
        
        # 計算每個子向量中使用的唯一中心點數量
        total_unique_centroids = 0
        for i in range(self.n_subvectors):
            total_unique_centroids += len(np.unique(codes[:, i]))
            
        # 選擇性 = 平均每個子向量使用的中心點比例
        selectivity = total_unique_centroids / (self.n_subvectors * self.n_centroids)
        return selectivity

# 向後兼容的別名
FastPQ = DiskANNPQ