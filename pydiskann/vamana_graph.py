import numpy as np
import heapq
import random
from tqdm import tqdm
from typing import Optional, List, Tuple

class Node:
    def __init__(self, idx, vector, pq_code=None):
        self.idx = idx
        self.vector = vector
        self.pq_code = pq_code  # 新增：PQ 編碼
        self.neighbors = set()

class VamanaGraphWithPQ:
    def __init__(self, R, pq_model=None):
        self.R = R
        self.nodes = {}
        self.pq_model = pq_model  # PQ 模型
        self.medoid_idx = None
        self.use_pq_for_search = False  # 是否在搜索時使用 PQ
        self._distance_table_cache = {}  # 緩存距離表
        
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

def l2_distance_fast(x, y):
    """快速 L2 平方距離計算"""
    return np.dot(x - y, x - y)

def pq_distance_fast(pq_model, code1, code2):
    """快速 PQ 距離計算（平方距離，與 l2_distance_fast 一致）"""
    if not pq_model or not pq_model.is_fitted:
        raise ValueError("PQ 模型未初始化")
    
    total_dist_sq = 0.0
    for i in range(pq_model.n_subvectors):
        centroid1 = pq_model.kmeans_list[i].cluster_centers_[code1[i]]
        centroid2 = pq_model.kmeans_list[i].cluster_centers_[code2[i]]
        diff = centroid1 - centroid2
        total_dist_sq += np.dot(diff, diff)
    return total_dist_sq

def compute_distance(graph, idx1, idx2, query_vector=None):
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
        return l2_distance_fast(node1.vector, node2.vector)

def compute_query_distance(graph, query_vector, node_idx):
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
        return l2_distance_fast(node.vector, query_vector)

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
    
    visited = {start_idx}
    start_dist = compute_query_distance(graph, query_vector, start_idx)
    
    candidates = [(start_dist, start_idx)]
    results = [(-start_dist, start_idx)]
    
    while candidates:
        dist, current_idx = heapq.heappop(candidates)
        
        if dist > -results[0][0]:
            break

        for neighbor_idx in graph.nodes[current_idx].neighbors:
            if neighbor_idx not in visited:
                visited.add(neighbor_idx)
                neighbor_dist = compute_query_distance(graph, query_vector, neighbor_idx)
                
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
                    candidates_with_dist = [(d, i) for d, i in candidates_with_dist if i != p_prime_idx]
                    
        graph.nodes[point_idx].neighbors = new_neighbors
        
    finally:
        # 恢復原始 PQ 設置
        if hasattr(graph, 'use_pq_for_search'):
            graph.use_pq_for_search = original_use_pq

def generate_initial_neighbors(n_points, R):
    """向量化生成初始鄰居矩陣"""
    neighbor_matrix = np.zeros((n_points, min(R, n_points-1)), dtype=np.int32)
    for idx in range(n_points):
        if n_points <= 1:
            continue
        rand_idx = np.random.choice(n_points - 1, min(R, n_points-1), replace=False)
        neighbors = [i if i < idx else i + 1 for i in rand_idx]
        neighbor_matrix[idx] = neighbors
    return neighbor_matrix

def build_vamana_with_pq(points, pq_model=None, R=16, L=32, alpha=1.2, 
                        use_pq_in_build=False, show_progress=False):
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
    """
    n_points = len(points)
    if n_points == 0:
        return VamanaGraphWithPQ(R, pq_model)
        
    graph = VamanaGraphWithPQ(R, pq_model)
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
        print("初始化隨機連接...")
    
    initial_neighbors = generate_initial_neighbors(n_points, R)
    for idx in tqdm(range(n_points), disable=not show_progress, desc="Init neighbors"):
        for neighbor in initial_neighbors[idx]:
            graph.add_edge(idx, neighbor)

    if show_progress:
        print("計算近似 medoid...")
    medoid_idx = compute_approximate_medoid(points_array, sample_size=min(1000, n_points))
    graph.medoid_idx = medoid_idx
    if show_progress:
        print(f"選擇的近似 medoid: {medoid_idx}")

    # 兩階段優化
    for pass_num in range(2):
        sigma = list(range(n_points))
        random.shuffle(sigma)
        
        current_alpha = 1.0 if pass_num == 0 else alpha
        
        iterator = tqdm(sigma, desc=f'Vamana Pass {pass_num + 1}', disable=not show_progress)
        for idx in iterator:
            candidates = greedy_search_with_pq(graph, medoid_idx, graph.nodes[idx].vector, L)
            candidate_set = set(candidates) | graph.nodes[idx].neighbors
            robust_prune_with_pq(graph, idx, candidate_set, alpha=current_alpha, R=R)
            
            for neighbor_idx in list(graph.nodes[idx].neighbors):
                graph.add_edge(neighbor_idx, idx)
                if len(graph.nodes[neighbor_idx].neighbors) > R:
                    robust_prune_with_pq(graph, neighbor_idx, 
                                       graph.nodes[neighbor_idx].neighbors, 
                                       alpha=current_alpha, R=R)
    
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
        
        start_dist = compute_query_distance(graph, query_vector, start_idx)
        heapq.heappush(beam, (start_dist, start_idx))
        heapq.heappush(top_k, (-start_dist, start_idx))
        
        while beam:
            dist, current_idx = heapq.heappop(beam)
            if dist > -top_k[0][0] and len(top_k) == k:
                break
                
            for neighbor_idx in graph.nodes[current_idx].neighbors:
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    neighbor_dist = compute_query_distance(graph, query_vector, neighbor_idx)
                    
                    if len(top_k) < k or neighbor_dist < -top_k[0][0]:
                        heapq.heappush(beam, (neighbor_dist, neighbor_idx))
                        heapq.heappush(top_k, (-neighbor_dist, neighbor_idx))
                        if len(top_k) > k:
                            heapq.heappop(top_k)
                            
            while len(beam) > beam_width:
                heapq.heappop(beam)
        
        # 如果使用了 PQ，返回時需要將平方距離開根號
        if hasattr(graph, 'use_pq_for_search') and graph.use_pq_for_search:
            return sorted([(np.sqrt(d), idx) for d, idx in [(-d, idx) for d, idx in top_k] if idx >= 0])
        else:
            return sorted([(np.sqrt(d), idx) for d, idx in [(-d, idx) for d, idx in top_k] if idx >= 0])
            
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
    """從磁盤讀取的 Beam Search（保持原有實現）"""
    visited = {start_id}
    beam = [] 
    top_k = []
    vec, _ = reader.get_node(start_id)
    init_dist = np.linalg.norm(vec - query_vector)
    heapq.heappush(beam, (init_dist, start_id))
    heapq.heappush(top_k, (-init_dist, start_id)) 
    while beam:
        dist, current_idx = heapq.heappop(beam)
        if dist > -top_k[0][0] and len(top_k) == k:
            break
        _, neighbors = reader.get_node(current_idx)
        for nid in neighbors:
            if nid in visited or nid < 0:
                continue
            visited.add(nid)
            neighbor_vec, _ = reader.get_node(nid)
            neighbor_dist = np.linalg.norm(neighbor_vec - query_vector)
            if len(top_k) < k or neighbor_dist < -top_k[0][0]:
                heapq.heappush(beam, (neighbor_dist, nid))
                heapq.heappush(top_k, (-neighbor_dist, nid))
                if len(top_k) > k:
                    heapq.heappop(top_k)
        while len(beam) > beam_width:
            heapq.heappop(beam)
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