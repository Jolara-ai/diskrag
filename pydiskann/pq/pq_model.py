import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, List, Optional

class DiskANNPQSearch:
    """
    使用 PQ 編碼進行 DiskANN 搜索的輔助類
    """
    
    def __init__(self, pq_model, pq_codes: np.ndarray):
        """
        初始化 PQ 搜索器
        
        Args:
            pq_model: 已訓練的 PQ 模型
            pq_codes: 所有向量的 PQ 編碼 [N, n_subvectors]
        """
        self.pq_model = pq_model
        self.pq_codes = pq_codes
        self.distance_table = None
        
    def prepare_query(self, query_vector: np.ndarray) -> None:
        """
        為查詢向量準備距離表
        
        Args:
            query_vector: 查詢向量 [D]
        """
        self.distance_table = self.pq_model.compute_distance_table(query_vector)
        
    def fast_distance(self, node_ids: List[int]) -> np.ndarray:
        """
        快速計算查詢向量與指定節點的近似距離
        
        Args:
            node_ids: 節點 ID 列表
            
        Returns:
            近似距離數組
        """
        if self.distance_table is None:
            raise ValueError("請先調用 prepare_query() 方法")
            
        codes = self.pq_codes[node_ids]
        return self.pq_model.asymmetric_distance(codes, self.distance_table)
        
    def filter_candidates(self, node_ids: List[int], threshold: float) -> List[int]:
        """
        使用 PQ 距離過濾候選節點
        
        Args:
            node_ids: 候選節點 ID 列表
            threshold: 距離閾值
            
        Returns:
            過濾後的節點 ID 列表
        """
        distances = self.fast_distance(node_ids)
        return [node_ids[i] for i in range(len(node_ids)) if distances[i] <= threshold]


# 向後兼容的別名
SimplePQ = DiskANNPQSearch
