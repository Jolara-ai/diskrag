import numpy as np
import heapq
import random
from tqdm import tqdm
from typing import Optional, List, Tuple


class Node:
    def __init__(self, idx, vector, pq_code=None, is_deleted=False):
        self.idx = idx
        self.vector = vector
        self.pq_code = pq_code  # 新增：PQ 編碼
        self.neighbors = set()
        self.is_deleted = is_deleted

class VamanaGraphWithPQ:
    def __init__(self, R, pq_model=None, distance_metric='l2'):
        self.R = R
        self.nodes = {}
        self.pq_model = pq_model  # PQ 模型
        self.medoid_idx = None
        self.use_pq_for_search = False  # 是否在搜索時使用 PQ
        self._distance_table_cache = {}  # 緩存距離表
        self.distance_metric = distance_metric
        
    def set_pq_model(self, pq_model):
        """設置 PQ 模型並重新編碼所有向量"""
        self.pq_model = pq_model
        if pq_model and pq_model.is_fitted:
            print("重新編碼所有向量...")
            vectors = np.array([node.vector for node in self.nodes.values()])
            if len(vectors) > 0:
                pq_codes = pq_model.encode(vectors)
                for i, (idx, node) in enumerate(self.nodes.items()):
                    node.pq_code = pq_codes[i]
            print(f"完成 {len(vectors)} 個向量的 PQ 編碼")

    def add_node(self, idx, vector, pq_code=None):
        """添加節點，支持 PQ 編碼"""
        if pq_code is None and self.pq_model and self.pq_model.is_fitted:
            pq_code = self.pq_model.encode(vector.reshape(1, -1))[0]
        self.nodes[idx] = Node(idx, vector, pq_code)

    def add_edge(self, from_idx, to_idx):
        if from_idx != to_idx and from_idx in self.nodes and to_idx in self.nodes:
            self.nodes[from_idx].neighbors.add(to_idx)
            
    def enable_pq_search(self, enable=True):
        """啟用/禁用 PQ 加速搜索"""
        if enable and (not self.pq_model or not self.pq_model.is_fitted):
            raise ValueError("PQ 模型未訓練，無法啟用 PQ 搜索")
        self.use_pq_for_search = enable
        if enable:
            print("已啟用 PQ 加速搜索")
        else:
            print("已禁用 PQ 加速搜索，使用精確距離計算")

    def insert_node(self, idx, vector, pq_code=None, L_insert=None):
        """
        插入一個新節點到圖中。
        Args:
            idx: 新節點的索引。
            vector: 新節點的向量。
            pq_code: 新節點的 PQ 編碼（如果可用）。
            L_insert: 插入時用於搜尋鄰居的候選集大小。如果為 None，則使用圖的 R 值。
        """
        if idx in self.nodes:
            if self.nodes[idx].is_deleted:
                self.nodes[idx].is_deleted = False
                self.nodes[idx].vector = vector
                self.nodes[idx].pq_code = pq_code
                self.nodes[idx].neighbors.clear() # Clear old neighbors for re-pruning
                print(f"節點 {idx} 已重新啟用。")
            else:
                raise ValueError(f"節點 {idx} 已存在。")
        else:
            if pq_code is None and self.pq_model and self.pq_model.is_fitted:
                pq_code = self.pq_model.encode(vector.reshape(1, -1))[0]
            new_node = Node(idx, vector, pq_code)
            self.nodes[idx] = new_node

        if len(self.nodes) == 1:
            self.medoid_idx = idx
            return

        # Find initial neighbors for the new node using greedy search
        start_node_for_search = self.medoid_idx
        if start_node_for_search is None or self.nodes[start_node_for_search].is_deleted:
            for node_id, node_obj in self.nodes.items():
                if not node_obj.is_deleted and node_id != idx:
                    start_node_for_search = node_id
                    break
            else:
                self.medoid_idx = idx
                return

        L_val = L_insert if L_insert is not None else self.R * 2
        
        # Perform greedy search to find candidate neighbors for the new node
        candidates_for_new_node = greedy_search_cython(self, start_node_for_search, new_node.vector, L_val, compute_query_distance)
        
        # Robust prune the new node's neighbors
        robust_prune_cython(self, idx, set(candidates_for_new_node), 1.0, self.R, compute_distance)
        
        # Ensure bidirectional connections for the new node's neighbors
        for neighbor_idx in list(self.nodes[idx].neighbors):
            if neighbor_idx in self.nodes and not self.nodes[neighbor_idx].is_deleted:
                self.add_edge(neighbor_idx, idx)
                # Re-prune neighbor's connections if it now exceeds R (simplified, full re-prune is complex)
                if len(self.nodes[neighbor_idx].neighbors) > self.R:
                    # For simplicity, we just prune the neighbor's list, not a full robust_prune
                    # A full robust_prune here would be computationally expensive for every insertion
                    # This is a known trade-off in dynamic HNSW/Vamana
                    pass # Placeholder for more complex re-pruning if needed

    def delete_node(self, idx):
        """
        標記節點為已刪除。
        Args:
            idx: 要刪除的節點索引。
        """
        if idx not in self.nodes:
            raise ValueError(f"節點 {idx} 不存在。")
        self.nodes[idx].is_deleted = True
        print(f"節點 {idx} 已標記為刪除。")

    def consolidate_index(self, R=None, L=None, alpha=None, pq_model=None, distance_metric=None, show_progress=False):
        """
        通過從所有活動節點重建圖來合併索引。
        這將物理移除已刪除的節點並重新優化圖結構。
        Args:
            R: 新圖的最大出度。如果為 None，則使用當前圖的 R。
            L: 新圖搜尋時的候選集大小。如果為 None，則使用當前圖的 R * 2。
            alpha: 新圖剪枝參數。如果為 None，則使用 1.0。
            pq_model: 已訓練的 PQ 模型。如果為 None，則使用當前圖的 pq_model。
            distance_metric: 距離度量。如果為 None，則使用當前圖的 distance_metric。
            show_progress: 是否顯示進度。
        """
        print("開始合併索引...")

        # 收集所有活動節點的向量和索引
        active_nodes_data = []
        active_node_map = {} # 舊索引到新索引的映射
        new_idx_counter = 0
        for old_idx, node in sorted(self.nodes.items()): # 排序以確保重建時的確定性
            if not node.is_deleted:
                active_nodes_data.append(node.vector)
                active_node_map[old_idx] = new_idx_counter
                new_idx_counter += 1
        
        if not active_nodes_data:
            print("沒有活動節點可供合併。索引已清空。")
            self.nodes = {}
            self.medoid_idx = None
            self._distance_table_cache.clear()
            return

        # 準備用於重建的參數
        rebuild_R = R if R is not None else self.R
        rebuild_L = L if L is not None else self.R * 2 # 默認使用 R*2
        rebuild_alpha = alpha if alpha is not None else 1.0
        rebuild_pq_model = pq_model if pq_model is not None else self.pq_model
        rebuild_distance_metric = distance_metric if distance_metric is not None else self.distance_metric

        # 使用 build_vamana_with_pq 函數重建圖
        # 注意：build_vamana_with_pq 期望從 0 開始的連續索引
        # 我們需要一個臨時的映射來處理這個問題，或者直接傳遞向量並讓它生成新索引
        
        # 為了簡化，我們將直接傳遞向量，並讓 build_vamana_with_pq 創建一個新圖
        # 然後我們將其狀態複製過來
        
        # 創建一個臨時的 VamanaGraphWithPQ 實例來構建新圖
        temp_graph = build_vamana_with_pq(
            points=active_nodes_data,
            pq_model=rebuild_pq_model,
            R=rebuild_R,
            L=rebuild_L,
            alpha=rebuild_alpha,
            use_pq_in_build=False, # 合併時通常不使用 PQ 構建
            show_progress=show_progress,
            distance_metric=rebuild_distance_metric
        )

        # 將新圖的狀態複製到當前實例
        self.nodes = {}
        for new_idx, temp_node in temp_graph.nodes.items():
            # 找到原始節點的索引
            original_idx = None
            for old_idx, mapped_new_idx in active_node_map.items():
                if mapped_new_idx == new_idx:
                    original_idx = old_idx
                    break
            
            if original_idx is not None:
                # 創建一個新的 Node 對象，使用原始索引
                node_obj = Node(original_idx, temp_node.vector, temp_node.pq_code, is_deleted=False)
                node_obj.neighbors = set() # 清空鄰居，稍後重新添加
                self.nodes[original_idx] = node_obj
        
        # 重新添加鄰居，並將新圖的鄰居映射回原始索引
        for new_idx, temp_node in temp_graph.nodes.items():
            original_idx = None
            for old_idx, mapped_new_idx in active_node_map.items():
                if mapped_new_idx == new_idx:
                    original_idx = old_idx
                    break
            
            if original_idx is not None:
                for temp_neighbor_idx in temp_node.neighbors:
                    original_neighbor_idx = None
                    for old_neighbor_idx, mapped_new_neighbor_idx in active_node_map.items():
                        if mapped_new_neighbor_idx == temp_neighbor_idx:
                            original_neighbor_idx = old_neighbor_idx
                            break
                    if original_neighbor_idx is not None:
                        self.nodes[original_idx].neighbors.add(original_neighbor_idx)

        # 更新 medoid
        if temp_graph.medoid_idx is not None:
            self.medoid_idx = None
            for old_idx, mapped_new_idx in active_node_map.items():
                if mapped_new_idx == temp_graph.medoid_idx:
                    self.medoid_idx = old_idx
                    break
        else:
            self.medoid_idx = None # 如果新圖沒有 medoid

        self._distance_table_cache.clear() # 清空緩存

        print(f"索引合併完成。剩餘 {len(self.nodes)} 個活動節點。")



class VamanaGraph:
    """向後兼容的原始 VamanaGraph 類"""
    def __init__(self, R):
        self.R = R
        self.nodes = {}

    def add_node(self, idx, vector):
        self.nodes[idx] = Node(idx, vector)

    def add_edge(self, from_idx, to_idx):
        if from_idx != to_idx:
            self.nodes[from_idx].neighbors.add(to_idx)

from .cython_utils import l2_distance_fast_cython, pq_distance_fast_cython, cosine_similarity_cython, greedy_search_cython, robust_prune_cython, generate_initial_neighbors_cython, compute_approximate_medoid_cython, build_vamana_index_cython

def l2_distance_fast(x, y):
    """快速 L2 平方距離計算"""
    return l2_distance_fast_cython(x, y)

def pq_distance_fast(pq_model, code1, code2):
    """快速 PQ 距離計算（平方距離，與 l2_distance_fast 一致）"""
    if not pq_model or not pq_model.is_fitted:
        raise ValueError("PQ 模型未初始化")
    return pq_distance_fast_cython(pq_model, code1, code2)

def compute_distance(graph, idx1, idx2, query_vector=None, distance_metric='l2'):
    """
    統一的距離計算函數
    - 構建時使用精確距離
    - 搜索時可選 PQ 距離或精確距離
    """
    node1 = graph.nodes[idx1]
    node2 = graph.nodes[idx2]
    
    # 如果有查詢向量且啟用了 PQ 搜索，使用 ADC
    if (query_vector is not None and 
        hasattr(graph, 'use_pq_for_search') and
        graph.use_pq_for_search and 
        graph.pq_model and 
        graph.pq_model.is_fitted and 
        node2.pq_code is not None):
        
        # 使用 ADC (Asymmetric Distance Computation)
        distance_table = graph.pq_model.compute_distance_table(query_vector)
        return graph.pq_model.asymmetric_distance_sq(
            node2.pq_code.reshape(1, -1), distance_table
        )[0]
    
    # 如果兩個節點都有 PQ 編碼且啟用了 PQ，使用 PQ 距離
    elif (hasattr(graph, 'use_pq_for_search') and
          graph.use_pq_for_search and 
          graph.pq_model and 
          graph.pq_model.is_fitted and 
          node1.pq_code is not None and 
          node2.pq_code is not None):
        
        return pq_distance_fast(graph.pq_model, node1.pq_code, node2.pq_code)
    
    # 否則使用精確距離
    else:
        if distance_metric == 'l2':
            return l2_distance_fast(node1.vector, node2.vector)
        elif distance_metric == 'cosine':
            return cosine_similarity_cython(node1.vector, node2.vector)
        else:
            raise ValueError(f"不支持的距離度量: {distance_metric}")

def compute_query_distance(graph, query_vector, node_idx, distance_metric='l2'):
    """計算查詢向量與節點的距離"""
    node = graph.nodes[node_idx]
    
    # 如果啟用 PQ 且節點有 PQ 編碼，使用 ADC
    if (hasattr(graph, 'use_pq_for_search') and
        graph.use_pq_for_search and 
        graph.pq_model and 
        graph.pq_model.is_fitted and 
        node.pq_code is not None):
        
        # 緩存距離表以提高性能
        query_id = id(query_vector)
        if query_id not in graph._distance_table_cache:
            graph._distance_table_cache[query_id] = graph.pq_model.compute_distance_table(query_vector)
        
        distance_table = graph._distance_table_cache[query_id]
        return graph.pq_model.asymmetric_distance_sq(
            node.pq_code.reshape(1, -1), distance_table
        )[0]
    
    # 否則使用精確距離
    else:
        if distance_metric == 'l2':
            return l2_distance_fast(node.vector, query_vector)
        elif distance_metric == 'cosine':
            return cosine_similarity_cython(node.vector, query_vector)
        else:
            raise ValueError(f"不支持的距離度量: {distance_metric}")

def compute_approximate_medoid(points_array, sample_size=1000, batch_size=1024):
    """
    使用採樣和向量化計算來高效地計算近似 medoid。
    """
    n_points, dim = points_array.shape
    
    if n_points <= sample_size:
        dist_sums = np.zeros(n_points)
        for i in range(n_points):
            dist_sums[i] = np.sum(np.linalg.norm(points_array - points_array[i], axis=1))
        return np.argmin(dist_sums)
    
    sample_indices = np.random.choice(n_points, sample_size, replace=False)
    sample_points = points_array[sample_indices]
    
    total_dist_sums = np.zeros(sample_size, dtype=np.float64)

    for i in range(0, n_points, batch_size):
        batch = points_array[i:i + batch_size]
        diff = batch[:, np.newaxis, :] - sample_points[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        total_dist_sums += np.sum(distances, axis=0)
        
    best_sample_local_idx = np.argmin(total_dist_sums)
    return sample_indices[best_sample_local_idx]

def greedy_search_with_pq(graph, start_idx, query_vector, L):
    """
    支持 PQ 的優化貪婪搜索
    """
    # 清空距離表緩存
    if hasattr(graph, '_distance_table_cache'):
        graph._distance_table_cache.clear()
    
    start_dist = compute_query_distance(graph, query_vector, start_idx, distance_metric=graph.distance_metric)
    
    # 如果起始節點被刪除，則尋找一個未刪除的節點作為起始點
    if graph.nodes[start_idx].is_deleted:
        for node_id, node_obj in graph.nodes.items():
            if not node_obj.is_deleted:
                start_idx = node_id
                start_dist = compute_query_distance(graph, query_vector, start_idx, distance_metric=graph.distance_metric)
                break
        else:
            return [] # 如果所有節點都被刪除，則返回空列表

    candidates = [(start_dist, start_idx)]
    results = [(-start_dist, start_idx)]
    
    while candidates:
        dist, current_idx = heapq.heappop(candidates)
        
        if dist > -results[0][0]:
            break

        for neighbor_idx in graph.nodes[current_idx].neighbors:
            if neighbor_idx not in visited and not graph.nodes[neighbor_idx].is_deleted:
                visited.add(neighbor_idx)
                neighbor_dist = compute_query_distance(graph, query_vector, neighbor_idx, distance_metric=graph.distance_metric)
                
                if len(results) < L or neighbor_dist < -results[0][0]:
                    heapq.heappush(candidates, (neighbor_dist, neighbor_idx))
                    heapq.heappush(results, (-neighbor_dist, neighbor_idx))
                    if len(results) > L:
                        heapq.heappop(results)
    
    # 清空緩存
    if hasattr(graph, '_distance_table_cache'):
        graph._distance_table_cache.clear()
    return [idx for neg_dist, idx in sorted(results, key=lambda x: -x[0])]

def robust_prune_with_pq(graph, point_idx, candidate_set, alpha, R):
    """
    支持 PQ 的剪枝算法
    注意：構建時始終使用精確距離以保證圖質量
    """
    # 構建時禁用 PQ 以確保圖質量
    original_use_pq = getattr(graph, 'use_pq_for_search', False)
    if hasattr(graph, 'use_pq_for_search'):
        graph.use_pq_for_search = False
    
    try:
        point_vector = graph.nodes[point_idx].vector
        new_neighbors = set()
        
        # 創建候選者列表並計算精確距離
        candidates_with_dist = []
        for cid in candidate_set:
            if cid in graph.nodes:  # 確保節點存在
                dist = compute_distance(graph, point_idx, cid, distance_metric=graph.distance_metric)
                candidates_with_dist.append((dist, cid))
        
        # 按距離排序
        candidates_with_dist.sort()
        
        # 進行剪枝
        for _, p_star_idx in candidates_with_dist:
            if len(new_neighbors) >= R:
                break
            new_neighbors.add(p_star_idx)
            
            p_star_vec = graph.nodes[p_star_idx].vector
            
            # 檢查並移除其他候選者
            temp_candidates = list(candidates_with_dist)
            for dist_p_prime, p_prime_idx in temp_candidates:
                if p_prime_idx in new_neighbors:
                    continue
                p_prime_vec = graph.nodes[p_prime_idx].vector
                dist_star_prime = compute_distance(graph, p_star_idx, p_prime_idx, distance_metric=graph.distance_metric)
                
                if alpha * dist_star_prime <= dist_p_prime:
                    candidates_with_dist = [(d, i) for d, i in candidates_with_dist if i != p_prime_idx]
                    
        graph.nodes[point_idx].neighbors = new_neighbors
        
    finally:
        # 恢復原始 PQ 設置
        if hasattr(graph, 'use_pq_for_search'):
            graph.use_pq_for_search = original_use_pq

def generate_initial_neighbors(n_points, R):
    """向量化生成初始鄰居矩陣"""
    # Deprecated: Use generate_initial_neighbors_cython instead
    neighbor_matrix = np.zeros((n_points, min(R, n_points-1)), dtype=np.int32)
    for idx in range(n_points):
        if n_points <= 1:
            continue
        rand_idx = np.random.choice(n_points - 1, min(R, n_points-1), replace=False)
        neighbors = [i if i < idx else i + 1 for i in rand_idx]
        neighbor_matrix[idx] = neighbors
    return neighbor_matrix

def build_vamana_with_pq(points, pq_model=None, R=16, L=32, alpha=1.2, 
                        use_pq_in_build=False, show_progress=False, distance_metric='l2'):
    """
    建立支持 PQ 的 Vamana 圖
    
    Args:
        points: 向量數據
        pq_model: 已訓練的 PQ 模型
        R: 最大出度
        L: 搜索時的候選集大小
        alpha: 剪枝參數
        use_pq_in_build: 構建時是否使用 PQ（建議 False 以保證質量）
        show_progress: 是否顯示進度
        distance_metric: 距離度量 ('l2' 或 'cosine')
    """
    n_points = len(points)
    if n_points == 0:
        return VamanaGraphWithPQ(R, pq_model, distance_metric)
        
    graph = VamanaGraphWithPQ(R, pq_model, distance_metric)
    points_array = np.array(points, dtype=np.float32)

    # 編碼所有向量（如果有 PQ 模型）
    pq_codes = None
    if pq_model and pq_model.is_fitted:
        if show_progress:
            print("使用 PQ 編碼向量...")
        pq_codes = pq_model.encode(points_array)
        if show_progress:
            print(f"完成 {len(points)} 個向量的 PQ 編碼")

    # 添加節點
    for idx, vec in enumerate(points_array):
        pq_code = pq_codes[idx] if pq_codes is not None else None
        graph.add_node(idx, vec, pq_code)

    # 設置是否在構建時使用 PQ
    graph.use_pq_for_search = use_pq_in_build

    if show_progress:
        print("初始化隨機連接 (C++ Optimized)...")
    
    # Use Cython optimized version
    initial_neighbors = generate_initial_neighbors_cython(n_points, R)
    
    for idx in tqdm(range(n_points), disable=not show_progress, desc="Init neighbors"):
        for neighbor in initial_neighbors[idx]:
            graph.add_edge(idx, neighbor)

    if show_progress:
        print("計算近似 medoid (C++ Optimized)...")
    medoid_idx = compute_approximate_medoid_cython(points_array, sample_size=min(1000, n_points))
    graph.medoid_idx = medoid_idx
    if show_progress:
        print(f"選擇的近似 medoid: {medoid_idx}")

    # Use C++ optimized 2-pass construction
    adj_list = build_vamana_index_cython(points_array, R, L, alpha, medoid_idx, show_progress)
    
    # Reconstruct graph from adjacency list
    for idx in range(n_points):
        graph.nodes[idx].neighbors = set(adj_list[idx])
    
    if show_progress:
        print(f"Vamana 圖構建完成，共 {len(graph.nodes)} 個節點")
        if pq_model:
            memory_info = pq_model.get_memory_usage()
            print(f"PQ 壓縮比: {memory_info['compression_ratio']:.1f}x")
    
    return graph

def beam_search_with_pq(graph, query_vector, start_idx=None, beam_width=5, k=3, use_pq=True):
    """
    支持 PQ 的 Beam Search
    """
    if start_idx is None:
        start_idx = graph.medoid_idx if graph.medoid_idx is not None else 0
    
    # 設置是否使用 PQ
    original_use_pq = getattr(graph, 'use_pq_for_search', False)
    if hasattr(graph, 'use_pq_for_search'):
        graph.use_pq_for_search = use_pq and graph.pq_model and graph.pq_model.is_fitted
    
    try:
        # 清空距離表緩存
        if hasattr(graph, '_distance_table_cache'):
            graph._distance_table_cache.clear()
        
        visited = {start_idx}
        beam = []
        top_k = []
        
        start_dist = compute_query_distance(graph, query_vector, start_idx, distance_metric=graph.distance_metric)
        
        # 如果起始節點被刪除，則尋找一個未刪除的節點作為起始點
        if graph.nodes[start_idx].is_deleted:
            for node_id, node_obj in graph.nodes.items():
                if not node_obj.is_deleted:
                    start_idx = node_id
                    start_dist = compute_query_distance(graph, query_vector, start_idx, distance_metric=graph.distance_metric)
                    break
            else:
                return [] # 如果所有節點都被刪除，則返回空列表

        heapq.heappush(beam, (start_dist, start_idx))
        heapq.heappush(top_k, (-start_dist, start_idx))
        
        while beam:
            dist, current_idx = heapq.heappop(beam)
            
            # 如果當前節點被刪除，則跳過
            if graph.nodes[current_idx].is_deleted:
                continue

            if dist > -top_k[0][0] and len(top_k) == k:
                break
                
            for neighbor_idx in graph.nodes[current_idx].neighbors:
                if neighbor_idx not in visited and not graph.nodes[neighbor_idx].is_deleted:
                    visited.add(neighbor_idx)
                    neighbor_dist = compute_query_distance(graph, query_vector, neighbor_idx, distance_metric=graph.distance_metric)
                    
                    if len(top_k) < k or neighbor_dist < -top_k[0][0]:
                        heapq.heappush(beam, (neighbor_dist, neighbor_idx))
                        heapq.heappush(top_k, (-neighbor_dist, neighbor_idx))
                        if len(top_k) > k:
                            heapq.heappop(top_k)
                            
            while len(beam) > beam_width:
                heapq.heappop(beam)
        
        # 過濾掉已刪除的節點，並從堆中還原距離
        final_results = [(-neg_dist, idx) for neg_dist, idx in top_k if idx >= 0 and not graph.nodes[idx].is_deleted]
        # 返回 (距離, 節點)
        return sorted([(np.sqrt(d), idx) for d, idx in final_results], key=lambda x: x[0])
            
    finally:
        # 恢復原始設置並清空緩存
        if hasattr(graph, 'use_pq_for_search'):
            graph.use_pq_for_search = original_use_pq
        if hasattr(graph, '_distance_table_cache'):
            graph._distance_table_cache.clear()

def greedy_search(graph, start_idx, query_vector, L):
    """
    優化的貪婪搜索實現（向後兼容）
    """
    # 檢查是否支持PQ
    if hasattr(graph, 'use_pq_for_search') and graph.use_pq_for_search:
        return greedy_search_with_pq(graph, start_idx, query_vector, L)
    
    # 原始實現
    visited = {start_idx}
    start_dist = np.linalg.norm(graph.nodes[start_idx].vector - query_vector)
    
    # 候選池 (距離, 節點ID)，小頂堆
    candidates = [(start_dist, start_idx)]
    # 結果集 (負距離, 節點ID)，大頂堆
    results = [(-start_dist, start_idx)]
    while candidates:
        dist, current_idx = heapq.heappop(candidates)
        
        if dist > -results[0][0]:
            break

        for neighbor_idx in graph.nodes[current_idx].neighbors:
            if neighbor_idx not in visited:
                visited.add(neighbor_idx)
                neighbor_dist = np.linalg.norm(graph.nodes[neighbor_idx].vector - query_vector)
                
                if len(results) < L or neighbor_dist < -results[0][0]:
                    heapq.heappush(candidates, (neighbor_dist, neighbor_idx))
                    heapq.heappush(results, (-neighbor_dist, neighbor_idx))
                    if len(results) > L:
                        heapq.heappop(results)
                        
    return [idx for neg_dist, idx in sorted(results, key=lambda x: -x[0])]

def robust_prune(graph, point_idx, candidate_set, alpha, R):
    """
    剪枝算法（向後兼容）
    """
    # 檢查是否支持PQ
    if hasattr(graph, 'use_pq_for_search'):
        return robust_prune_with_pq(graph, point_idx, candidate_set, alpha, R)
    
    # 原始實現
    point_vector = graph.nodes[point_idx].vector
    new_neighbors = set()
    
    # 創建候選者列表並計算距離
    candidates_with_dist = []
    for cid in candidate_set:
        dist = l2_distance_fast(point_vector, graph.nodes[cid].vector)
        candidates_with_dist.append((dist, cid))
    
    # 按距離排序
    candidates_with_dist.sort()
    
    # 進行剪枝
    for _, p_star_idx in candidates_with_dist:
        if len(new_neighbors) >= R:
            break
        new_neighbors.add(p_star_idx)
        
        p_star_vec = graph.nodes[p_star_idx].vector
        
        # 檢查並移除其他候選者
        temp_candidates = list(candidates_with_dist)
        for dist_p_prime, p_prime_idx in temp_candidates:
            if p_prime_idx in new_neighbors:
                continue
            p_prime_vec = graph.nodes[p_prime_idx].vector
            dist_star_prime = l2_distance_fast(p_star_vec, p_prime_vec)
            
            if alpha * dist_star_prime <= dist_p_prime:
                # 移除 p_prime_idx
                candidates_with_dist = [(d, i) for d, i in candidates_with_dist if i != p_prime_idx]
                
    graph.nodes[point_idx].neighbors = new_neighbors

# 為了向後兼容，保留原始函數名
def build_vamana(points, R=16, L=32, alpha=1.2, show_progress=False):
    """向後兼容的 build_vamana 函數"""
    return build_vamana_with_pq(points, None, R, L, alpha, False, show_progress)

def beam_search(graph, query_vector, start_idx, beam_width=5, k=3):
    """向後兼容的 beam_search 函數"""
    if hasattr(graph, 'use_pq_for_search'):
        return beam_search_with_pq(graph, query_vector, start_idx, beam_width, k, False)
    else:
        # 原始實現
        visited = {start_idx}
        beam = []
        top_k = []
        start_dist = l2_distance_fast(graph.nodes[start_idx].vector, query_vector)
        heapq.heappush(beam, (start_dist, start_idx))
        heapq.heappush(top_k, (-start_dist, start_idx))
        
        while beam:
            dist, current_idx = heapq.heappop(beam)
            if dist > -top_k[0][0] and len(top_k) == k:
                break
            for neighbor_idx in graph.nodes[current_idx].neighbors:
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    neighbor_dist = l2_distance_fast(graph.nodes[neighbor_idx].vector, query_vector)
                    if len(top_k) < k or neighbor_dist < -top_k[0][0]:
                        heapq.heappush(beam, (neighbor_dist, neighbor_idx))
                        heapq.heappush(top_k, (-neighbor_dist, neighbor_idx))
                        if len(top_k) > k:
                            heapq.heappop(top_k)
        
        return sorted([(np.sqrt(d), idx) for d, idx in [(-d, idx) for d, idx in top_k] if idx >= 0])

def beam_search_from_disk(reader, query_vector, start_id, beam_width=8, k=5):
    """從磁盤讀取的 Beam Search (類似 Greedy Search with L=beam_width)"""
    visited = {start_id}
    beam = [] 
    top_k = [] # This will actually hold 'beam_width' items, acting like 'results' in greedy_search
    
    vec, _ = reader.get_node(start_id)
    init_dist = np.linalg.norm(vec - query_vector)
    heapq.heappush(beam, (init_dist, start_id))
    heapq.heappush(top_k, (-init_dist, start_id)) 
    
    while beam:
        dist, current_idx = heapq.heappop(beam)
        
        # 終止條件：如果當前最近的候選節點比結果集中最差的還差，且結果集已滿
        if dist > -top_k[0][0] and len(top_k) >= beam_width:
            break
            
        _, neighbors = reader.get_node(current_idx)
        for nid in neighbors:
            if nid in visited or nid < 0:
                continue
            visited.add(nid)
            neighbor_vec, _ = reader.get_node(nid)
            neighbor_dist = np.linalg.norm(neighbor_vec - query_vector)
            
            # 如果結果集未滿，或新節點比結果集中最差的更好
            if len(top_k) < beam_width or neighbor_dist < -top_k[0][0]:
                heapq.heappush(beam, (neighbor_dist, nid))
                heapq.heappush(top_k, (-neighbor_dist, nid))
                if len(top_k) > beam_width:
                    heapq.heappop(top_k)
                    
        # Beam maintenance: keep top beam_width smallest distances in the frontier
        if len(beam) > beam_width:
            beam = heapq.nsmallest(beam_width, beam)
            heapq.heapify(beam) 
            
    # Return top k from the results
    sorted_results = sorted([(-d, idx) for d, idx in top_k if idx >= 0])
    return sorted_results[:k]
    return sorted([(-d, idx) for d, idx in top_k if idx >= 0])

def greedy_search_optimized(graph, start_idx, query_vector, L):
    """優化的貪婪搜索（保持原有實現）"""
    visited = {start_idx}
    
    start_dist = np.linalg.norm(graph.nodes[start_idx].vector - query_vector)
    
    # 候選池 (距離, 節點ID)，小頂堆
    candidates = [(start_dist, start_idx)] 
    # 結果集 (負距離, 節點ID)，大頂堆，方便維護 top L
    results = [(-start_dist, start_idx)]

    while candidates:
        dist, current_idx = heapq.heappop(candidates)
        
        # 剪枝: 如果候選池中最好的節點都比結果集中最差的還差，可以提前終止
        if dist > -results[0][0]:
            break

        for neighbor_idx in graph.nodes[current_idx].neighbors:
            if neighbor_idx not in visited:
                visited.add(neighbor_idx)
                neighbor_dist = np.linalg.norm(graph.nodes[neighbor_idx].vector - query_vector)
                
                # 如果結果集還沒滿，或者新節點比結果集中最差的要好
                if len(results) < L or neighbor_dist < -results[0][0]:
                    heapq.heappush(candidates, (neighbor_dist, neighbor_idx))
                    heapq.heappush(results, (-neighbor_dist, neighbor_idx))
                    
                    # 維護結果集大小
                    if len(results) > L:
                        heapq.heappop(results)
                        
    return [idx for neg_dist, idx in sorted(results, key=lambda x: -x[0])]