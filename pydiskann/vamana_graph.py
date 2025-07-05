import numpy as np
import heapq
import random
from tqdm import tqdm

# --- 基礎資料結構 ---

class Node:
    def __init__(self, idx, vector):
        self.idx = idx
        self.vector = vector
        self.neighbors = set()

class VamanaGraph:
    def __init__(self, R):
        self.R = R
        self.nodes = {}

    def add_node(self, idx, vector):
        self.nodes[idx] = Node(idx, vector)

    def add_edge(self, from_idx, to_idx):
        self.nodes[from_idx].neighbors.add(to_idx)

# --- 距離計算 ---

# 優化但等價的距離計算
def l2_distance_fast(x, y):
    diff = x - y
    return np.sqrt(np.dot(diff, diff))

def l2_distance(x, y):
    return np.linalg.norm(x - y)




# --- 建圖用 Greedy Search ---

def greedy_search(graph, start_idx, query_vector, L):
    visited = set()
    candidates = []
    heapq.heappush(candidates, (l2_distance_fast(graph.nodes[start_idx].vector, query_vector), start_idx))
    result = []

    while candidates and len(result) < L:
        dist, current_idx = heapq.heappop(candidates)
        if current_idx in visited:
            continue
        visited.add(current_idx)
        result.append(current_idx)

        for neighbor_idx in graph.nodes[current_idx].neighbors:
            if neighbor_idx not in visited:
                neighbor_dist = l2_distance_fast(graph.nodes[neighbor_idx].vector, query_vector)
                heapq.heappush(candidates, (neighbor_dist, neighbor_idx))

    return result, visited

# --- Robust Prune ---

def robust_prune(graph, point_idx, candidate_set, alpha, R):
    new_neighbors = []
    vectors = {idx: graph.nodes[idx].vector for idx in candidate_set}
    point_vector = graph.nodes[point_idx].vector

    while candidate_set and len(new_neighbors) < R:
        closest = min(candidate_set, key=lambda idx: l2_distance_fast(point_vector, vectors[idx]))
        new_neighbors.append(closest)
        candidate_set.remove(closest)

        to_remove = set()
        for idx in candidate_set:
            if alpha * l2_distance_fast(vectors[closest], vectors[idx]) <= l2_distance_fast(point_vector, vectors[idx]):
                to_remove.add(idx)
        candidate_set -= to_remove

    graph.nodes[point_idx].neighbors = set(new_neighbors)

# --- 建圖主程式 ---

def build_vamana(points, R=16, L=32, alpha=1.2, show_progress=False):
    n_points = len(points)
    graph = VamanaGraph(R)
    
    # 轉換為 numpy array 以加速計算
    points_array = np.array(points) if not isinstance(points, np.ndarray) else points

    # 添加所有節點
    for idx, vec in enumerate(points):
        graph.add_node(idx, vec)

    # 初始化隨機連接
    if show_progress:
        print("初始化隨機連接...")
    for idx in range(n_points):
        while len(graph.nodes[idx].neighbors) < R:
            neighbor = random.randint(0, n_points - 1)
            if neighbor != idx:
                graph.add_edge(idx, neighbor)

    if show_progress:
        print("計算 medoid（這可能需要一些時間）...")
    

    diff = points_array[:, np.newaxis, :] - points_array[np.newaxis, :, :]
    distances_matrix = np.linalg.norm(diff, axis=2)
    total_distances = np.sum(distances_matrix, axis=1)
    medoid_idx = np.argmin(total_distances)
    
    if show_progress:
        print(f"選擇的 medoid: {medoid_idx}")

    sigma = list(range(n_points))
    random.shuffle(sigma)

    # 第一階段
    iterator1 = sigma
    if show_progress:
        iterator1 = tqdm(sigma, desc='Vamana 第一階段')
    for idx in iterator1:
        candidates, _ = greedy_search(graph, medoid_idx, graph.nodes[idx].vector, L)
        robust_prune(graph, idx, set(candidates), alpha=1.0, R=R)
        for neighbor in graph.nodes[idx].neighbors.copy():
            graph.add_edge(neighbor, idx)
            if len(graph.nodes[neighbor].neighbors) > R:
                robust_prune(graph, neighbor, graph.nodes[neighbor].neighbors.copy(), alpha=1.0, R=R)

    # 第二階段
    iterator2 = sigma
    if show_progress:
        iterator2 = tqdm(sigma, desc='Vamana 第二階段')
    for idx in iterator2:
        candidates, _ = greedy_search(graph, medoid_idx, graph.nodes[idx].vector, L)
        robust_prune(graph, idx, set(candidates), alpha=alpha, R=R)
        for neighbor in graph.nodes[idx].neighbors.copy():
            graph.add_edge(neighbor, idx)
            if len(graph.nodes[neighbor].neighbors) > R:
                robust_prune(graph, neighbor, graph.nodes[neighbor].neighbors.copy(), alpha=alpha, R=R)

    return graph

# --- 查詢用 BeamSearch ---

def beam_search_from_disk(reader,query_vector, start_id, beam_width=8, k=5):
    visited = set()
    beam = []
    top_k = []


    vec, _ = reader.get_node(start_id)
    init_dist = np.linalg.norm(vec - query_vector)
    heapq.heappush(beam, (init_dist, start_id))
    heapq.heappush(top_k, (-init_dist, start_id))

    while beam:
        dist, current_idx = heapq.heappop(beam)
        if current_idx in visited:
            continue
        visited.add(current_idx)

        _, neighbors = reader.get_node(current_idx)

        for nid in neighbors:
            if nid in visited:
                continue
            neighbor_vec, _ = reader.get_node(nid)
            neighbor_dist = np.linalg.norm(neighbor_vec - query_vector)

            heapq.heappush(beam, (neighbor_dist, nid))
            heapq.heappush(top_k, (-neighbor_dist, nid))
            if len(top_k) > k:
                heapq.heappop(top_k)

        if len(beam) > beam_width:
            beam = heapq.nsmallest(beam_width, beam)
            heapq.heapify(beam)

        if beam and -top_k[0][0] < beam[0][0]:
            break

    return sorted([(-d, idx) for d, idx in top_k if idx >= 0])




def beam_search(graph, query_vector, start_idx, beam_width=5, k=3):
    visited = set()
    beam = []
    top_k = []

    heapq.heappush(beam, (l2_distance_fast(graph.nodes[start_idx].vector, query_vector), start_idx))
    heapq.heappush(top_k, (float('-inf'), -1))  # 初始化 top_k

    while beam:
        dist, current_idx = heapq.heappop(beam)
        if current_idx in visited:
            continue
        visited.add(current_idx)

        for neighbor_idx in graph.nodes[current_idx].neighbors:
            if neighbor_idx in visited:
                continue
            neighbor_dist = l2_distance_fast(graph.nodes[neighbor_idx].vector, query_vector)
            heapq.heappush(beam, (neighbor_dist, neighbor_idx))
            heapq.heappush(top_k, (-neighbor_dist, neighbor_idx))
            if len(top_k) > k:
                heapq.heappop(top_k)

        if len(beam) > beam_width:
            beam = heapq.nsmallest(beam_width, beam)
            heapq.heapify(beam)

        if beam and -top_k[0][0] < beam[0][0]:
            break

    return sorted([(-d, idx) for d, idx in top_k if idx >= 0])