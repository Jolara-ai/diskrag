"""
pydiskann - DiskANN Python 實作
支援 Cython 優化的高效能向量搜尋
"""

# 嘗試導入 Cython 優化模組（如果可用）
try:
    from .cython_utils import (
        l2_distance_fast_cython,
        pq_distance_fast_cython,
        cosine_similarity_cython,
        greedy_search_cython,
        robust_prune_cython,
        generate_initial_neighbors_cython,
        compute_approximate_medoid_cython,
        build_vamana_index_cython,
    )
    CYTHON_AVAILABLE = True
except ImportError:
    # Cython 模組未編譯或不可用
    CYTHON_AVAILABLE = False
    l2_distance_fast_cython = None
    pq_distance_fast_cython = None
    cosine_similarity_cython = None
    greedy_search_cython = None
    robust_prune_cython = None
    generate_initial_neighbors_cython = None
    compute_approximate_medoid_cython = None
    build_vamana_index_cython = None

__all__ = [
    'CYTHON_AVAILABLE',
    'l2_distance_fast_cython',
    'pq_distance_fast_cython',
    'cosine_similarity_cython',
    'greedy_search_cython',
    'robust_prune_cython',
    'generate_initial_neighbors_cython',
    'compute_approximate_medoid_cython',
    'build_vamana_index_cython',
]

