"""
DiskANN PQ 模組

提供 DiskANN 論文標準的 Product Quantization 實現，
支援 Asymmetric Distance Computation (ADC) 用於快速 beam search
"""

from .fast_pq import DiskANNPQ, FastPQ
from .adaptive_pq import (
    AdaptivePQCalculator,
    PQRecommendation,
    calculate_adaptive_pq_params,
    get_pq_recommendation_summary
)

__all__ = [
    'DiskANNPQ',
    'FastPQ', 
    'AdaptivePQCalculator',
    'PQRecommendation',
    'calculate_adaptive_pq_params',
    'get_pq_recommendation_summary'
]
