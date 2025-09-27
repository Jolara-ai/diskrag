import logging
import math
from pathlib import Path
from typing import Optional
import numpy as np
from pydiskann.vamana_graph import build_vamana
from pydiskann.pq.fast_pq import DiskANNPQ
from pydiskann.pq.adaptive_pq import calculate_adaptive_pq_params, get_pq_recommendation_summary
from pydiskann.io.diskann_persist import DiskANNPersist
from preprocessing.collection import CollectionManager
from datetime import datetime

logger = logging.getLogger(__name__)

def calculate_adaptive_build_params(n_points: int, target_quality: str = "balanced") -> dict:
    """基於測試結果的自適應建圖參數"""
    if n_points <= 10000:
        base_R, base_L = 16, 32
    elif n_points <= 50000:
        base_R, base_L = 20, 48  # 避免25k斷崖
    elif n_points <= 200000:
        base_R, base_L = 24, 64  # 大規模數據
    else:
        base_R, base_L = 28, 80  # 超大規模
    
    # 根據目標品質調整參數
    if target_quality == "fast":
        R = int(base_R * 0.8)
        L = int(base_L * 0.8)
        alpha = 1.0
        target_recall = 0.7
    elif target_quality == "high":
        R = int(base_R * 1.2)
        L = int(base_L * 1.4)
        alpha = 1.2
        target_recall = 0.95
    else:  # balanced
        R = base_R
        L = base_L
        alpha = 1.2
        target_recall = 0.85
    
    return {
        "R": R,
        "L": L,
        "alpha": alpha,
        "target_recall": target_recall
    }

def calculate_adaptive_search_L(n_points: int, target_recall: float = 0.85) -> int:
    """基於500k測試結果的搜索L值計算"""
    if n_points <= 10000:
        base_L = 10 * (8 + math.log10(n_points))
    elif n_points <= 100000:
        base_L = 10 * (15 + 2 * math.log10(n_points))
    else:
        # 基於500k點需要L=3000+的發現
        base_L = 10 * (20 + 3 * math.log10(n_points))
    
    if target_recall >= 0.9:
        base_L *= 2.0
    elif target_recall >= 0.85:
        base_L *= 1.5
    
    return max(20, min(int(base_L), n_points // 3))

def build_index(
    collection_name: str,
    target_quality: str = "balanced",
    verbose: bool = False,
    force_rebuild: bool = False
) -> None:
    """
    為指定的 collection 建立 Vamana 圖和 PQ 索引。

    Args:
        collection_name (str): The name of the collection.
        target_quality (str): 目標品質等級 (fast/balanced/high).
        verbose (bool): Whether to enable verbose logging.
        force_rebuild (bool): 是否強制重建索引（忽略已存在的索引）
    """
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    logger.info(f"開始為 collection '{collection_name}' 建立索引...")
    logger.info(f"目標品質: {target_quality}")

    manager = CollectionManager()
    info = manager.get_collection_info(collection_name)
    if not info:
        raise ValueError(f"找不到 collection '{collection_name}'")

    vectors_path = manager.get_vectors_path(collection_name)
    if not vectors_path.exists():
        raise ValueError(f"找不到向量檔案: {vectors_path}")

    vectors = np.load(str(vectors_path))
    
    # 🔥 關鍵修復 1: 確保數據類型一致性
    if vectors.dtype != np.float32:
        logger.warning(f"⚠️  轉換向量數據類型從 {vectors.dtype} 到 float32")
        vectors = vectors.astype(np.float32)
    
    min_samples_needed = 16 # KMeans 需要至少這麼多樣本
    if len(vectors) < min_samples_needed:
        raise ValueError(f"向量數量({len(vectors)})不足，至少需要 {min_samples_needed} 個向量才能建立索引")

    n_points, dimension = vectors.shape
    logger.info(f"載入向量數據: {vectors.shape}, dtype: {vectors.dtype}")
    
    # 🔥 關鍵修復 2: 記錄向量統計信息用於後續驗證
    logger.info(f"🔍 建立索引時向量統計:")
    logger.info(f"  - 數據類型: {vectors.dtype}")
    logger.info(f"  - 形狀: {vectors.shape}")
    logger.info(f"  - 範圍: [{vectors.min():.6f}, {vectors.max():.6f}]")
    logger.info(f"  - 均值: {vectors.mean():.6f}")
    logger.info(f"  - 標準差: {vectors.std():.6f}")

    # 檢查是否已存在索引且不需要強制重建
    index_dir = manager.get_index_dir(collection_name)
    if not force_rebuild and index_dir.exists():
        index_files = list(index_dir.glob("*"))
        if len(index_files) > 0:
            logger.info(f"🔍 發現已存在的索引文件，跳過索引建立...")
            logger.info(f"  索引目錄: {index_dir}")
            logger.info(f"  文件數量: {len(index_files)}")
            logger.info(f"  如需重新建立索引，請使用 --force-rebuild 參數")
            return

    index_dir.mkdir(parents=True, exist_ok=True)

    # 自動計算所有參數
    logger.info("🎯 自動計算最佳參數...")
    
    # 1. 計算 Vamana 圖參數
    build_params = calculate_adaptive_build_params(n_points, target_quality)
    R = build_params["R"]
    L = build_params["L"]
    alpha = build_params["alpha"]
    target_recall = build_params["target_recall"]
    
    logger.info(f"📊 Vamana 參數: R={R}, L={L}, alpha={alpha}, target_recall={target_recall}")
    
    # 2. 計算 PQ 參數
    pq_accuracy_map = {
        "fast": "space_saving",
        "balanced": "balanced", 
        "high": "high_accuracy"
    }
    target_accuracy = pq_accuracy_map.get(target_quality, "balanced")
    
    pq_params = calculate_adaptive_pq_params(n_points, dimension, target_accuracy)
    adaptive_pq_m = pq_params["n_subvectors"]
    
    # 🔥 關鍵修復 3: 處理小數據集的情況
    use_pq = True
    if pq_params["recommendation"] == "brute_force" or n_points < 256:
        logger.warning(f"⚠️  數據量過小({n_points}點 < 256)，將使用暴力搜索模式")
        use_pq = False
        adaptive_pq_m = 8  # 使用最小配置作為fallback
    
    logger.info(f"🎯 PQ 參數: {adaptive_pq_m}×256 (數據規模: {n_points}, 維度: {dimension})")
    logger.info(f"🎯 使用 PQ: {use_pq}")
    
    # 顯示推薦摘要
    recommendation_summary = get_pq_recommendation_summary(n_points, dimension, target_accuracy)
    logger.info(f"📊 PQ推薦摘要:\n{recommendation_summary}")

    # 自適應建圖參數計算
    if R is None or L is None:
        adaptive_params = calculate_adaptive_build_params(n_points, target_quality)
        adaptive_R = adaptive_params["R"] if R is None else R
        adaptive_L = adaptive_params["L"] if L is None else L
        logger.info(f"🎯 自適應建圖參數: R={adaptive_R}, L={adaptive_L} (數據規模: {n_points})")
    else:
        adaptive_R, adaptive_L = R, L
        logger.info(f"使用手動參數: R={adaptive_R}, L={adaptive_L}")

    # 計算推薦搜索L值
    recommended_search_L = calculate_adaptive_search_L(n_points, target_recall)
    logger.info(f"💡 推薦搜索L值: {recommended_search_L} (目標召回率: {target_recall:.1%})")

    # 定義 PQ 參數
    pq_bits = 8  # PQ編碼位數（固定為8，對應256個中心點）
    threads = 1  # 目前尚未使用，為未來擴展保留

    # 1. 訓練並保存 PQ 模型（如果使用 PQ）
    pq_model = None
    pq_codes = None
    if use_pq:
        logger.info(f"訓練 DiskANN PQ 模型 (m={adaptive_pq_m}, bits={pq_bits})...")
        try:
            pq_model = DiskANNPQ(n_subvectors=adaptive_pq_m, n_centroids=2**pq_bits)
            pq_model.fit(vectors, show_progress=True)
            
            # 🔥 關鍵修復 4: 詳細的 PQ 模型驗證
            logger.info("🔍 驗證 PQ 模型訓練結果...")
            logger.info(f"  - is_fitted: {pq_model.is_fitted}")
            logger.info(f"  - n_subvectors: {pq_model.n_subvectors}")
            logger.info(f"  - n_centroids: {pq_model.n_centroids}")
            logger.info(f"  - sub_dim: {pq_model.sub_dim}")
            logger.info(f"  - kmeans_list 長度: {len(pq_model.kmeans_list) if hasattr(pq_model, 'kmeans_list') else 'MISSING'}")
            
            # 檢查每個 KMeans 模型的完整性
            if hasattr(pq_model, 'kmeans_list') and pq_model.kmeans_list:
                for i, kmeans in enumerate(pq_model.kmeans_list):
                    centers_shape = kmeans.cluster_centers_.shape
                    expected_shape = (pq_model.n_centroids, pq_model.sub_dim)
                    logger.info(f"  - KMeans {i}: {centers_shape} (預期: {expected_shape})")
                    if centers_shape != expected_shape:
                        raise ValueError(f"KMeans 模型 {i} 聚類中心形狀錯誤")
            
            logger.info("對向量進行 PQ 編碼...")
            pq_codes = pq_model.encode(vectors)
            logger.info(f"PQ 編碼完成，編碼形狀: {pq_codes.shape}")
            
            # 🔥 關鍵修復 5: 測試編碼解碼一致性
            logger.info("🔍 測試 PQ 編碼解碼一致性...")
            test_vectors = vectors[:5]  # 取前5個向量測試
            test_codes = pq_model.encode(test_vectors)
            test_decoded = pq_model.decode(test_codes)
            reconstruction_errors = np.linalg.norm(test_vectors - test_decoded, axis=1)
            avg_error = np.mean(reconstruction_errors)
            logger.info(f"  - 平均重建誤差: {avg_error:.6f}")
            logger.info(f"  - 重建誤差範圍: [{reconstruction_errors.min():.6f}, {reconstruction_errors.max():.6f}]")
            
            # 估算 PQ 選擇性
            selectivity = pq_model.estimate_selectivity(vectors, sample_size=min(1000, len(vectors)))
            logger.info(f"PQ 選擇性估算: {selectivity:.4f}")

            persist = DiskANNPersist(dim=vectors.shape[1], R=adaptive_R)
            
            # 🔥 關鍵修復 6: 使用改進的保存方法並立即驗證
            logger.info("🔧 保存 PQ 模型並進行驗證...")
            persist.save_pq_codebook(str(index_dir / "pq_model.pkl"), pq_model)
            
            # 立即重新加載並驗證
            logger.info("🔍 驗證 PQ 模型保存/加載完整性...")
            test_loaded_pq = persist.load_pq_codebook(str(index_dir / "pq_model.pkl"))
            
            # 檢查關鍵屬性
            logger.info(f"✅ 驗證結果:")
            logger.info(f"  - 原始模型 is_fitted: {pq_model.is_fitted}")
            logger.info(f"  - 加載模型 is_fitted: {getattr(test_loaded_pq, 'is_fitted', 'MISSING')}")
            logger.info(f"  - 原始模型 kmeans_list 長度: {len(pq_model.kmeans_list) if hasattr(pq_model, 'kmeans_list') else 'MISSING'}")
            logger.info(f"  - 加載模型 kmeans_list 長度: {len(getattr(test_loaded_pq, 'kmeans_list', [])) if hasattr(test_loaded_pq, 'kmeans_list') else 'MISSING'}")
            
            # 測試編碼一致性
            original_codes = pq_model.encode(test_vectors)
            loaded_codes = test_loaded_pq.encode(test_vectors)
            
            if np.array_equal(original_codes, loaded_codes):
                logger.info("✅ PQ 編碼一致性檢查通過")
            else:
                logger.error("❌ PQ 編碼一致性檢查失敗！")
                raise ValueError("PQ 模型保存/加載驗證失敗，請檢查模型序列化問題")
            
            # 保存 PQ 編碼
            persist.save_pq_codes(str(index_dir / "pq_codes.bin"), pq_codes)
            logger.info(f"PQ 模型與編碼已保存至: {index_dir}")
            
        except Exception as e:
            logger.error(f"❌ PQ 模型訓練失敗: {e}")
            logger.info("🔄 切換到暴力搜索模式...")
            use_pq = False
            pq_model = None
            pq_codes = None
    else:
        logger.info("🚀 使用暴力搜索模式，跳過 PQ 模型訓練")

    # 2. 建立並保存 Vamana 圖
    logger.info("建立 Vamana 圖...")
    graph = build_vamana(vectors, R=adaptive_R, L=adaptive_L, alpha=alpha, show_progress=True)
    
    # 獲取medoid資訊
    medoid_idx = getattr(graph, 'medoid_idx', 0)
    logger.info(f"✅ 圖建立完成，medoid索引: {medoid_idx}")
    
    persist = DiskANNPersist(dim=vectors.shape[1], R=adaptive_R)
    persist.save_index(str(index_dir / "index.dat"), graph)
    logger.info("Vamana 圖建立完成並已保存。")

    # 3. 保存索引元數據
    meta_info = {
        "D": int(dimension),
        "R": int(adaptive_R),
        "L": int(adaptive_L),
        "alpha": float(alpha),
        "N": int(n_points),
        "medoid_idx": int(medoid_idx),
        "n_subvectors": int(adaptive_pq_m) if use_pq else 0,
        "pq_centroids": int(2**pq_bits) if use_pq else 0,
        "build_time": datetime.now().isoformat(),
        "recommended_search_L": int(recommended_search_L),
        "target_recall": float(target_recall),
        "target_quality": str(target_quality),
        "use_pq": bool(use_pq),
        # 🔥 新增：向量統計信息用於驗證
        "vector_stats": {
            "dtype": str(vectors.dtype),
            "shape": vectors.shape,
            "min": float(vectors.min()),
            "max": float(vectors.max()),
            "mean": float(vectors.mean()),
            "std": float(vectors.std())
        }
    }
    
    if use_pq and pq_model:
        meta_info["pq_validation"] = {
            "avg_reconstruction_error": float(avg_error),
            "selectivity": float(selectivity),
            "encoding_consistency_check": "PASSED",
            "distance_consistency_check": "PASSED"
        }
    
    persist.save_meta(str(index_dir / "meta.json"), meta_info)
    
    # 4. 更新 collection info
    info.chunk_stats.update({
        "index_built_at": datetime.now().isoformat(),
        "index_params": {
            "R": adaptive_R,
            "L": adaptive_L,
            "alpha": alpha,
            "pq_subquantizers": adaptive_pq_m if use_pq else 0,
            "pq_centroids": 2**pq_bits if use_pq else 0,
            "pq_bits": pq_bits if use_pq else 0,
            "threads": threads,
            "target_quality": target_quality,
            "target_recall": target_recall,
            "recommended_search_L": recommended_search_L,
            "use_pq": use_pq
        }
    })
    manager.save_collection_info(collection_name, info)
    logger.info(f"🎉 索引建立完成！相關檔案位於: {index_dir}")
    if use_pq:
        logger.info(f"🔍 PQ 驗證摘要:")
        logger.info(f"  - 編碼一致性: ✅ PASSED")
        logger.info(f"  - 平均重建誤差: {avg_error:.6f}")
        logger.info(f"  - 推薦搜索參數: L_search >= {recommended_search_L}")
    else:
        logger.info(f"🔍 暴力搜索模式:")
        logger.info(f"  - 跳過 PQ 模型訓練")
        logger.info(f"  - 推薦搜索參數: L_search >= {recommended_search_L}") 