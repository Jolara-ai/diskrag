#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自適應 PQ 參數計算模組
根據數據規模和維度動態推薦 PQ 參數
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class PQRecommendation:
    """PQ參數推薦結果"""
    n_subvectors: int
    n_centroids: int
    sub_dimension: int
    recommendation: str
    compression_ratio: float
    expected_recall: float
    expected_spearman: float
    reasoning: str

class AdaptivePQCalculator:
    """自適應PQ參數計算器"""
    
    def __init__(self):
        # 預定義的子向量候選值
        self.subvector_candidates = [4, 8, 16, 32, 48, 64, 96, 128]
        
        # 性能基準數據（基於測試結果）
        self.performance_baseline = {
            4: {"recall": 0.20, "spearman": 0.96, "compression": 128.0},
            8: {"recall": 0.50, "spearman": 0.98, "compression": 64.0},
            16: {"recall": 0.60, "spearman": 0.99, "compression": 32.0},
            32: {"recall": 0.90, "spearman": 1.00, "compression": 16.0},
            48: {"recall": 0.85, "spearman": 0.99, "compression": 10.7},
            64: {"recall": 0.90, "spearman": 1.00, "compression": 8.0},
            96: {"recall": 0.88, "spearman": 0.99, "compression": 5.3},
        }
    
    def calculate_adaptive_pq_params(self, n_points: int, dimension: int, 
                                   target_accuracy: str = "balanced") -> PQRecommendation:
        """
        根據數據規模和維度，動態計算推薦的 PQ 參數
        
        Args:
            n_points: 數據點數量
            dimension: 向量維度
            target_accuracy: 目標精度 ('high_accuracy', 'balanced', 'space_saving')
        
        Returns:
            PQRecommendation: 推薦的PQ參數
        """
        
        # 規則 1: 處理數據量極小的情況
        if n_points < 1000:
            return PQRecommendation(
                n_subvectors=0,
                n_centroids=0,
                sub_dimension=0,
                recommendation="brute_force",
                compression_ratio=1.0,
                expected_recall=1.0,
                expected_spearman=1.0,
                reasoning="數據量過小，建議使用暴力搜索"
            )
        
        # 規則 2: 確定子向量數量的候選範圍
        possible_subvectors = self._get_valid_subvectors(dimension)
        
        if not possible_subvectors:
            # 如果沒有找到合適的候選值，使用默認值
            possible_subvectors = [8, 16, 32]
        
        # 規則 3: 根據數據規模和目標精度選擇最佳參數
        best_params = self._select_best_params(n_points, dimension, possible_subvectors, target_accuracy)
        
        return best_params
    
    def _get_valid_subvectors(self, dimension: int) -> List[int]:
        """獲取有效的子向量候選值"""
        valid_subvectors = []
        
        for m in self.subvector_candidates:
            if dimension % m == 0:
                sub_dim = dimension // m
                # 確保子向量維度在合理範圍內 [2, 64]
                if 2 <= sub_dim <= 64:
                    valid_subvectors.append(m)
        
        return valid_subvectors
    
    def _select_best_params(self, n_points: int, dimension: int, 
                           possible_subvectors: List[int], 
                           target_accuracy: str) -> PQRecommendation:
        """選擇最佳參數"""
        
        # 根據數據規模調整策略
        if n_points <= 50000:  # 中小型數據集
            if target_accuracy == "high_accuracy":
                best_m = max(possible_subvectors)
                recommendation = "high_accuracy"
                reasoning = f"中小型數據集({n_points:,}點)，追求高精度"
            else:
                best_m = possible_subvectors[len(possible_subvectors) // 2]
                recommendation = "balanced"
                reasoning = f"中小型數據集({n_points:,}點)，平衡配置"
                
        elif n_points <= 500000:  # 大型數據集
            if target_accuracy == "space_saving":
                best_m = min(possible_subvectors)
                recommendation = "space_saving"
                reasoning = f"大型數據集({n_points:,}點)，優先節省空間"
            else:
                best_m = possible_subvectors[len(possible_subvectors) // 2]
                recommendation = "balanced"
                reasoning = f"大型數據集({n_points:,}點)，平衡配置"
                
        elif n_points <= 2000000:  # 超大型數據集
            if target_accuracy == "high_accuracy":
                best_m = possible_subvectors[len(possible_subvectors) // 3]
                recommendation = "balanced"
                reasoning = f"超大型數據集({n_points:,}點)，平衡精度和空間"
            else:
                best_m = min(possible_subvectors)
                recommendation = "space_saving"
                reasoning = f"超大型數據集({n_points:,}點)，優先節省空間"
                
        else:  # 極大型數據集
            best_m = min(possible_subvectors)
            recommendation = "space_saving"
            reasoning = f"極大型數據集({n_points:,}點)，最大壓縮比"
        
        # 計算子向量維度
        sub_dimension = dimension // best_m
        
        # 獲取性能預測
        performance = self._predict_performance(best_m, dimension)
        
        return PQRecommendation(
            n_subvectors=best_m,
            n_centroids=256,  # 固定為256
            sub_dimension=sub_dimension,
            recommendation=recommendation,
            compression_ratio=performance["compression"],
            expected_recall=performance["recall"],
            expected_spearman=performance["spearman"],
            reasoning=reasoning
        )
    
    def _predict_performance(self, n_subvectors: int, dimension: int) -> Dict:
        """預測性能指標"""
        # 基於基準數據進行插值預測
        if n_subvectors in self.performance_baseline:
            return self.performance_baseline[n_subvectors].copy()
        
        # 對於不在基準中的值，進行線性插值
        baseline_keys = sorted(self.performance_baseline.keys())
        
        if n_subvectors < baseline_keys[0]:
            # 小於最小值，使用最小值的性能
            return self.performance_baseline[baseline_keys[0]].copy()
        elif n_subvectors > baseline_keys[-1]:
            # 大於最大值，使用最大值的性能
            return self.performance_baseline[baseline_keys[-1]].copy()
        else:
            # 在範圍內，進行線性插值
            for i in range(len(baseline_keys) - 1):
                if baseline_keys[i] <= n_subvectors <= baseline_keys[i + 1]:
                    m1, m2 = baseline_keys[i], baseline_keys[i + 1]
                    p1, p2 = self.performance_baseline[m1], self.performance_baseline[m2]
                    
                    # 線性插值
                    ratio = (n_subvectors - m1) / (m2 - m1)
                    
                    return {
                        "recall": p1["recall"] + ratio * (p2["recall"] - p1["recall"]),
                        "spearman": p1["spearman"] + ratio * (p2["spearman"] - p1["spearman"]),
                        "compression": p1["compression"] + ratio * (p2["compression"] - p1["compression"])
                    }
        
        # 默認值
        return {"recall": 0.8, "spearman": 0.95, "compression": 16.0}
    
    def get_recommendation_summary(self, recommendation: PQRecommendation) -> str:
        """獲取推薦摘要"""
        if recommendation.recommendation == "brute_force":
            return f"💡 推薦: {recommendation.reasoning}"
        
        summary = f"""
🎯 PQ 參數推薦: {recommendation.n_subvectors}×256
📊 子向量維度: {recommendation.sub_dimension}
📈 預期性能:
   - Top-10召回率: {recommendation.expected_recall:.1%}
   - 排序相關性: {recommendation.expected_spearman:.1%}
   - 壓縮比: {recommendation.compression_ratio:.1f}x
💡 策略: {recommendation.reasoning}
        """
        return summary.strip()
    
    def validate_recommendation(self, recommendation: PQRecommendation, 
                              n_points: int, dimension: int) -> Tuple[bool, str]:
        """驗證推薦參數的合理性"""
        
        if recommendation.recommendation == "brute_force":
            return True, "數據量過小，建議使用暴力搜索"
        
        # 檢查子向量維度是否合理
        if recommendation.sub_dimension < 2:
            return False, f"子向量維度過小: {recommendation.sub_dimension}"
        
        if recommendation.sub_dimension > 64:
            return False, f"子向量維度過大: {recommendation.sub_dimension}"
        
        # 檢查壓縮比是否合理
        if recommendation.compression_ratio < 2:
            return False, f"壓縮比過低: {recommendation.compression_ratio:.1f}x"
        
        # 檢查預期性能是否合理
        if recommendation.expected_recall < 0.1:
            return False, f"預期召回率過低: {recommendation.expected_recall:.1%}"
        
        return True, "參數驗證通過"

# 便捷函數
def calculate_adaptive_pq_params(n_points: int, dimension: int, 
                               target_accuracy: str = "balanced") -> Dict:
    """
    便捷函數：計算自適應PQ參數
    
    Args:
        n_points: 數據點數量
        dimension: 向量維度
        target_accuracy: 目標精度 ('high_accuracy', 'balanced', 'space_saving')
    
    Returns:
        Dict: 包含推薦參數的字典
    """
    calculator = AdaptivePQCalculator()
    recommendation = calculator.calculate_adaptive_pq_params(n_points, dimension, target_accuracy)
    
    return {
        "n_subvectors": recommendation.n_subvectors,
        "n_centroids": recommendation.n_centroids,
        "sub_dimension": recommendation.sub_dimension,
        "recommendation": recommendation.recommendation,
        "compression_ratio": recommendation.compression_ratio,
        "expected_recall": recommendation.expected_recall,
        "expected_spearman": recommendation.expected_spearman,
        "reasoning": recommendation.reasoning
    }

def get_pq_recommendation_summary(n_points: int, dimension: int, 
                                target_accuracy: str = "balanced") -> str:
    """獲取PQ推薦摘要"""
    calculator = AdaptivePQCalculator()
    recommendation = calculator.calculate_adaptive_pq_params(n_points, dimension, target_accuracy)
    return calculator.get_recommendation_summary(recommendation)

# 測試函數
def test_adaptive_pq():
    """測試自適應PQ參數計算"""
    calculator = AdaptivePQCalculator()
    
    test_cases = [
        (500, 128, "balanced"),
        (50000, 128, "high_accuracy"),
        (500000, 128, "balanced"),
        (2000000, 128, "space_saving"),
        (100000, 768, "balanced"),
        (1000000, 512, "high_accuracy"),
        (500000, 960, "balanced"),  # 新增 960 維度測試案例
    ]
    
    print("🧪 自適應PQ參數測試")
    print("=" * 60)
    
    for n_points, dimension, target_accuracy in test_cases:
        print(f"\n📊 測試案例: {n_points:,} 點, {dimension} 維, {target_accuracy}")
        print("-" * 50)
        
        recommendation = calculator.calculate_adaptive_pq_params(n_points, dimension, target_accuracy)
        summary = calculator.get_recommendation_summary(recommendation)
        print(summary)
        
        # 驗證參數
        is_valid, message = calculator.validate_recommendation(recommendation, n_points, dimension)
        print(f"✅ 驗證: {message}")

if __name__ == "__main__":
    test_adaptive_pq() 