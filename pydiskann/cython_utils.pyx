# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as np
import heapq

# Declare numpy array types for Cython
DTYPE_FLOAT = np.float32
DTYPE_UINT8 = np.uint8
ctypedef np.float32_t DTYPE_FLOAT_t
ctypedef np.uint8_t DTYPE_UINT8_t


def l2_distance_fast_cython(np.ndarray[DTYPE_FLOAT_t, ndim=1] x, np.ndarray[DTYPE_FLOAT_t, ndim=1] y):
    cdef DTYPE_FLOAT_t dist_sq = 0.0
    cdef Py_ssize_t i
    cdef Py_ssize_t n = x.shape[0]
    for i in range(n):
        dist_sq += (x[i] - y[i]) * (x[i] - y[i])
    return dist_sq

def pq_distance_fast_cython(pq_model, np.ndarray[DTYPE_UINT8_t, ndim=1] code1, np.ndarray[DTYPE_UINT8_t, ndim=1] code2):
    if not pq_model.is_fitted:
        raise ValueError("PQ 模型未初始化")
    
    cdef DTYPE_FLOAT_t total_dist_sq = 0.0
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t n_subvectors = pq_model.n_subvectors
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] centroid1
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] centroid2
    cdef Py_ssize_t sub_dim
    cdef DTYPE_FLOAT_t diff_val
    cdef DTYPE_FLOAT_t sub_dist_sq
    
    for i in range(n_subvectors):
        centroid1 = pq_model.kmeans_list[i].cluster_centers_[code1[i]]
        centroid2 = pq_model.kmeans_list[i].cluster_centers_[code2[i]]
        
        sub_dim = centroid1.shape[0]
        sub_dist_sq = 0.0
        for j in range(sub_dim):
            diff_val = centroid1[j] - centroid2[j]
            sub_dist_sq += diff_val * diff_val
        total_dist_sq += sub_dist_sq
        
    return total_dist_sq

def cosine_similarity_cython(np.ndarray[DTYPE_FLOAT_t, ndim=1] x, np.ndarray[DTYPE_FLOAT_t, ndim=1] y):
    cdef DTYPE_FLOAT_t dot_product = 0.0
    cdef DTYPE_FLOAT_t norm_x = 0.0
    cdef DTYPE_FLOAT_t norm_y = 0.0
    cdef Py_ssize_t i
    cdef Py_ssize_t n = x.shape[0]

    for i in range(n):
        dot_product += x[i] * y[i]
        norm_x += x[i] * x[i]
        norm_y += y[i] * y[i]

    if norm_x == 0.0 or norm_y == 0.0:
        return 0.0  # Or handle as an error, or return 1.0 if vectors are identical zero vectors
    
    # Cosine distance is 1 - cosine_similarity.
    # We return 1 - similarity to be consistent with distance semantics (smaller is better).
    return 1.0 - (dot_product / (np.sqrt(norm_x) * np.sqrt(norm_y)))

def greedy_search_cython(graph, int start_idx, np.ndarray[DTYPE_FLOAT_t, ndim=1] query_vector, int L, compute_query_distance_fn):
    cdef set visited = set()
    cdef list candidates = [] # (dist, idx)
    cdef list results = []    # (-dist, idx)
    cdef DTYPE_FLOAT_t start_dist
    cdef DTYPE_FLOAT_t dist
    cdef int current_idx
    cdef int neighbor_idx
    cdef DTYPE_FLOAT_t neighbor_dist
    cdef int k_val # for top_k in results

    # Handle deleted start_idx
    if graph.nodes[start_idx].is_deleted:
        for node_id in graph.nodes:
            if not graph.nodes[node_id].is_deleted:
                start_idx = node_id
                break
        else:
            return [] # All nodes deleted

    start_dist = compute_query_distance_fn(graph, query_vector, start_idx, graph.distance_metric)
    
    visited.add(start_idx)
    heapq.heappush(candidates, (start_dist, start_idx))
    heapq.heappush(results, (-start_dist, start_idx))
    
    while candidates:
        dist, current_idx = heapq.heappop(candidates)
        
        # Ensure current_idx is not deleted (could be added to candidates before deletion)
        if graph.nodes[current_idx].is_deleted:
            continue

        if dist > -results[0][0]:
            break

        for neighbor_idx in graph.nodes[current_idx].neighbors:
            if neighbor_idx not in visited and not graph.nodes[neighbor_idx].is_deleted:
                visited.add(neighbor_idx)
                neighbor_dist = compute_query_distance_fn(graph, query_vector, neighbor_idx, graph.distance_metric)
                
                k_val = len(results)
                if k_val < L or neighbor_dist < -results[0][0]:
                    heapq.heappush(candidates, (neighbor_dist, neighbor_idx))
                    heapq.heappush(results, (-neighbor_dist, neighbor_idx))
                    if len(results) > L:
                        heapq.heappop(results)
    
    # Filter out deleted nodes and sort
    cdef list final_results = [(neg_dist, idx) for neg_dist, idx in results if not graph.nodes[idx].is_deleted]
    return [idx for neg_dist, idx in sorted(final_results, key=lambda x: -x[0])]

def robust_prune_cython(graph, int point_idx, candidate_set, DTYPE_FLOAT_t alpha, int R, compute_distance_fn):
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] point_vector = graph.nodes[point_idx].vector
    cdef set new_neighbors = set()
    cdef list candidates_with_dist = [] # (dist, cid)
    cdef int cid
    cdef DTYPE_FLOAT_t dist
    cdef int p_star_idx
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] p_star_vec
    cdef list temp_candidates
    cdef DTYPE_FLOAT_t dist_p_prime
    cdef int p_prime_idx
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] p_prime_vec
    cdef DTYPE_FLOAT_t dist_star_prime

    # Create candidates list and compute exact distances
    for cid in candidate_set:
        if cid in graph.nodes and not graph.nodes[cid].is_deleted:
            dist = compute_distance_fn(graph, point_idx, cid, graph.distance_metric)
            candidates_with_dist.append((dist, cid))
    
    # Sort by distance
    candidates_with_dist.sort()
    
    # Perform pruning
    for dist, p_star_idx in candidates_with_dist:
        if len(new_neighbors) >= R:
            break
        new_neighbors.add(p_star_idx)
        
        p_star_vec = graph.nodes[p_star_idx].vector
        
        # Check and remove other candidates
        temp_candidates = list(candidates_with_dist) # Create a copy to iterate
        for dist_p_prime, p_prime_idx in temp_candidates:
            if p_prime_idx in new_neighbors or graph.nodes[p_prime_idx].is_deleted:
                continue
            p_prime_vec = graph.nodes[p_prime_idx].vector
            dist_star_prime = compute_distance_fn(graph, p_star_idx, p_prime_idx, graph.distance_metric)
            
            if alpha * dist_star_prime <= dist_p_prime:
                # Remove p_prime_idx from candidates_with_dist
                candidates_with_dist = [(d, i) for d, i in candidates_with_dist if i != p_prime_idx]
                
    graph.nodes[point_idx].neighbors = new_neighbors

cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937:
        mt19937()
        mt19937(unsigned int seed)
        unsigned int operator()()
    
    cdef cppclass uniform_int_distribution[T]:
        uniform_int_distribution()
        uniform_int_distribution(T a, T b)
        T operator()(mt19937& gen)

from libc.time cimport time

def generate_initial_neighbors_cython(int n_points, int R):
    cdef np.ndarray[np.int32_t, ndim=2] neighbor_matrix = np.zeros((n_points, R), dtype=np.int32)
    cdef int i, j, k
    cdef int candidate
    cdef mt19937 gen
    cdef uniform_int_distribution[int] dist
    cdef set selected
    
    gen = mt19937(time(NULL))
    dist = uniform_int_distribution[int](0, n_points - 1)
    
    # Note: For very large n_points, we could use parallel loop (prange) 
    # but we need thread-local RNGs. For now, single thread C++ is fast enough.
    
    for i in range(n_points):
        selected = set()
        selected.add(i) # Don't select self
        
        j = 0
        while j < R:
            candidate = dist(gen)
            if candidate not in selected:
                selected.add(candidate)
                neighbor_matrix[i, j] = candidate
                j += 1
                
    return neighbor_matrix

def compute_approximate_medoid_cython(float[:, :] points_array, int sample_size=1000):
    """
    C++ optimized medoid approximation using sampling
    """
    cdef int n_points = points_array.shape[0]
    cdef int dim = points_array.shape[1]
    cdef int i, j, k, sample_idx
    cdef float dist, diff
    cdef np.ndarray[np.float64_t, ndim=1] dist_sums
    cdef np.ndarray[np.int32_t, ndim=1] sample_indices
    cdef mt19937 gen
    cdef uniform_int_distribution[int] dist_gen
    
    # If dataset is small, use all points
    if n_points <= sample_size:
        dist_sums = np.zeros(n_points, dtype=np.float64)
        for i in range(n_points):
            for j in range(n_points):
                if i == j:
                    continue
                dist = 0.0
                for k in range(dim):
                    diff = points_array[i, k] - points_array[j, k]
                    dist += diff * diff
                dist_sums[i] += dist ** 0.5
        return np.argmin(dist_sums)
    
    # Sample points
    gen = mt19937(time(NULL) + 1)
    dist_gen = uniform_int_distribution[int](0, n_points - 1)
    
    cdef set sampled = set()
    sample_indices = np.zeros(sample_size, dtype=np.int32)
    i = 0
    while i < sample_size:
        sample_idx = dist_gen(gen)
        if sample_idx not in sampled:
            sampled.add(sample_idx)
            sample_indices[i] = sample_idx
            i += 1
    
    # Compute distance sums for sampled points against all points
    dist_sums = np.zeros(sample_size, dtype=np.float64)
    for i in range(sample_size):
        sample_idx = sample_indices[i]
        for j in range(n_points):
            dist = 0.0
            for k in range(dim):
                diff = points_array[sample_idx, k] - points_array[j, k]
                dist += diff * diff
            dist_sums[i] += dist ** 0.5
    
    best_sample_idx = np.argmin(dist_sums)
    return sample_indices[best_sample_idx]

from libcpp.vector cimport vector
from libcpp.set cimport set as cpp_set
from libc.stdlib cimport rand, srand, RAND_MAX

def build_vamana_index_cython(float[:, :] points_array, int R, int L, float alpha, int medoid_idx, show_progress=False):
    """
    Pure C++ implementation of Vamana graph construction.
    Returns adjacency list as a Python list of lists.
    """
    cdef int n_points = points_array.shape[0]
    cdef int dim = points_array.shape[1]
    cdef int i, j, k, idx, neighbor_idx
    cdef int pass_num
    cdef float current_alpha
    cdef vector[int] sigma
    cdef vector[vector[int]] adj_list  # Adjacency list
    cdef vector[int] candidates
    cdef cpp_set[int] candidate_set
    cdef cpp_set[int] current_neighbors
    cdef bint edge_exists
    
    # Initialize adjacency list
    adj_list.resize(n_points)
    
    # Seed random
    srand(time(NULL) + 42)
    
    if show_progress:
        print(f"Starting C++ Vamana 2-pass construction for {n_points} points...")
    
    # Two-pass optimization
    for pass_num in range(2):
        # Prepare shuffled order
        sigma.clear()
        for i in range(n_points):
            sigma.push_back(i)
        
        # Shuffle using Python's random for reproducibility
        import random
        py_sigma = list(range(n_points))
        random.shuffle(py_sigma)
        sigma.clear()
        for i in py_sigma:
            sigma.push_back(i)
        
        current_alpha = 1.0 if pass_num == 0 else alpha
        
        if show_progress:
            print(f"Vamana Pass {pass_num + 1}...")
            from tqdm import tqdm
            progress_bar = tqdm(total=n_points, desc=f'Vamana Pass {pass_num + 1}')
        
        # Main construction loop
        for k in range(n_points):
            idx = sigma[k]
            
            # Greedy search from medoid to current point
            candidates = greedy_search_fast_cython(points_array, adj_list, medoid_idx, idx, L)
            
            # Combine candidates with current neighbors
            candidate_set.clear()
            for i in range(candidates.size()):
                candidate_set.insert(candidates[i])
            for i in range(adj_list[idx].size()):
                candidate_set.insert(adj_list[idx][i])
            
            # Robust prune
            robust_prune_fast_cython(points_array, adj_list, idx, candidate_set, current_alpha, R)
            
            # Update reverse edges
            for i in range(adj_list[idx].size()):
                neighbor_idx = adj_list[idx][i]
                # Add reverse edge if not already present
                if neighbor_idx != idx:
                    # Check if edge exists
                    edge_exists = False
                    for j in range(adj_list[neighbor_idx].size()):
                        if adj_list[neighbor_idx][j] == idx:
                            edge_exists = True
                            break
                    if not edge_exists:
                        adj_list[neighbor_idx].push_back(idx)
                    
                    # Prune neighbor if it exceeds R
                    if adj_list[neighbor_idx].size() > R:
                        current_neighbors.clear()
                        for j in range(adj_list[neighbor_idx].size()):
                            current_neighbors.insert(adj_list[neighbor_idx][j])
                        robust_prune_fast_cython(points_array, adj_list, neighbor_idx, current_neighbors, current_alpha, R)
            
            if show_progress and k % 100 == 0:
                progress_bar.update(100)
        
        if show_progress:
            progress_bar.close()
    
    # Convert to Python list of lists
    result = []
    for i in range(n_points):
        neighbors = []
        for j in range(adj_list[i].size()):
            neighbors.append(adj_list[i][j])
        result.append(neighbors)
    
    return result

cdef vector[int] greedy_search_fast_cython(float[:, :] points, vector[vector[int]]& adj_list, 
                                            int start_idx, int query_idx, int L):
    """Fast greedy search using C++ structures"""
    cdef int n_points = points.shape[0]
    cdef int dim = points.shape[1]
    cdef float[:] query_vec = points[query_idx]
    cdef cpp_set[int] visited
    cdef vector[pair[float, int]] candidates  # min heap
    cdef vector[pair[float, int]] results     # max heap (negative distances)
    cdef float dist, neighbor_dist
    cdef int current_idx, neighbor_idx
    cdef int i, k
    
    # Compute start distance
    dist = 0.0
    for k in range(dim):
        dist += (points[start_idx, k] - query_vec[k]) * (points[start_idx, k] - query_vec[k])
    
    candidates.push_back(pair[float, int](dist, start_idx))
    results.push_back(pair[float, int](-dist, start_idx))
    visited.insert(start_idx)
    
    while not candidates.empty():
        # Get minimum from candidates
        dist = candidates[0].first
        current_idx = candidates[0].second
        candidates.erase(candidates.begin())
        
        # Early termination
        if results.size() >= L and dist > -results[0].first:
            break
        
        # Explore neighbors
        for i in range(adj_list[current_idx].size()):
            neighbor_idx = adj_list[current_idx][i]
            if visited.count(neighbor_idx) > 0:
                continue
            visited.insert(neighbor_idx)
            
            # Compute distance
            neighbor_dist = 0.0
            for k in range(dim):
                neighbor_dist += (points[neighbor_idx, k] - query_vec[k]) * (points[neighbor_idx, k] - query_vec[k])
            
            # Add to candidates and results
            if results.size() < L or neighbor_dist < -results[0].first:
                candidates.push_back(pair[float, int](neighbor_dist, neighbor_idx))
                results.push_back(pair[float, int](-neighbor_dist, neighbor_idx))
                
                # Maintain heap size
                if results.size() > L:
                    # Remove max (worst) from results
                    results.erase(results.begin())
        
        # Sort candidates to maintain min-heap property (simple approach)
        sort(candidates.begin(), candidates.end())
    
    # Extract indices from results
    cdef vector[int] result_indices
    for i in range(results.size()):
        result_indices.push_back(results[i].second)
    
    return result_indices

cdef void robust_prune_fast_cython(float[:, :] points, vector[vector[int]]& adj_list,
                                    int point_idx, cpp_set[int]& candidate_set, 
                                    float alpha, int R):
    """Fast robust prune using C++ structures"""
    cdef int dim = points.shape[0]
    cdef vector[pair[float, int]] candidates_with_dist
    cdef cpp_set[int] new_neighbors
    cdef float dist, dist_to_p_star, dist_star_prime
    cdef int cid, p_star_idx, p_prime_idx
    cdef int i, j, k
    cdef bint should_remove
    
    # Compute distances for all candidates
    for cid in candidate_set:
        if cid == point_idx:
            continue
        dist = 0.0
        for k in range(points.shape[1]):
            dist += (points[point_idx, k] - points[cid, k]) * (points[point_idx, k] - points[cid, k])
        candidates_with_dist.push_back(pair[float, int](dist, cid))
    
    # Sort by distance
    sort(candidates_with_dist.begin(), candidates_with_dist.end())
    
    # Greedy pruning
    for i in range(candidates_with_dist.size()):
        if new_neighbors.size() >= R:
            break
        
        p_star_idx = candidates_with_dist[i].second
        dist_to_p_star = candidates_with_dist[i].first
        new_neighbors.insert(p_star_idx)
        
        # Check other candidates for pruning
        j = i + 1
        while j < candidates_with_dist.size():
            p_prime_idx = candidates_with_dist[j].second
            
            if new_neighbors.count(p_prime_idx) > 0:
                j += 1
                continue
            
            # Compute distance between p_star and p_prime
            dist_star_prime = 0.0
            for k in range(points.shape[1]):
                dist_star_prime += (points[p_star_idx, k] - points[p_prime_idx, k]) * (points[p_star_idx, k] - points[p_prime_idx, k])
            
            # Check pruning condition
            if alpha * dist_star_prime <= candidates_with_dist[j].first:
                # Remove this candidate
                candidates_with_dist.erase(candidates_with_dist.begin() + j)
            else:
                j += 1
    
    # Update adjacency list
    adj_list[point_idx].clear()
    for p_star_idx in new_neighbors:
        adj_list[point_idx].push_back(p_star_idx)

# Required C++ declarations
cdef extern from "<algorithm>" namespace "std":
    void sort[T](T first, T last) nogil

cdef extern from "<utility>" namespace "std":
    cdef cppclass pair[T, U]:
        T first
        U second
        pair() except +
        pair(T, U) except +