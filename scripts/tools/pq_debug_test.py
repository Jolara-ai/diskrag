#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PQ 模型修復腳本
重新訓練並保存 PQ 模型，確保編碼一致性
"""

import numpy as np
import logging
import sys
from pathlib import Path
from pydiskann.pq.fast_pq import DiskANNPQ
from pydiskann.io.diskann_persist import DiskANNPersist, MMapNodeReader
from preprocessing.collection import CollectionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PQFixer")

def fix_pq_model(collection_name: str):
    """修復指定集合的 PQ 模型"""
    logger.info(f"🔧 開始修復集合 '{collection_name}' 的 PQ 模型...")
    
    try:
        manager = CollectionManager()
        info = manager.get_collection_info(collection_name)
        if not info:
            logger.error(f"❌ 找不到集合: {collection_name}")
            return False
        
        # 載入原始向量數據
        vectors_path = manager.get_vectors_path(collection_name)
        
        if not vectors_path.exists():
            logger.error(f"❌ 向量文件不存在: {vectors_path}")
            return False
            
        vectors = np.load(str(vectors_path))
        
        # 檢查向量數據是否為空或損壞
        if vectors.size == 0:
            logger.error(f"❌ 向量數據為空！文件: {vectors_path}")
            logger.error("請檢查向量數據文件是否存在且包含有效數據")
            return False
        
        # 檢查向量數量是否與集合信息一致
        if info and len(vectors) != info.num_vectors:
            logger.warning(f"⚠️  向量數量不匹配: 文件中有 {len(vectors)} 個，集合信息顯示 {info.num_vectors} 個")
            logger.warning("這可能表示向量文件損壞，需要重新生成")
            
            # 嘗試從索引文件中恢復向量
            logger.info("🔧 嘗試從索引文件中恢復向量...")
            try:
                index_dir = manager.get_index_dir(collection_name)
                reader = MMapNodeReader(str(index_dir / "index.dat"), dim=info.dimension)
                
                # 讀取所有向量
                recovered_vectors = []
                for i in range(info.num_vectors):
                    vec, _ = reader.get_node(i)
                    recovered_vectors.append(vec)
                
                vectors = np.array(recovered_vectors, dtype=np.float32)
                reader.close()
                
                logger.info(f"✅ 成功從索引恢復 {len(vectors)} 個向量")
                
                # 保存恢復的向量
                np.save(str(vectors_path), vectors)
                logger.info(f"💾 已保存恢復的向量到: {vectors_path}")
                
            except Exception as e:
                logger.error(f"❌ 無法從索引恢復向量: {e}")
                return False
        
        # 確保數據類型為 float32
        if vectors.dtype != np.float32:
            logger.info(f"🔄 轉換數據類型從 {vectors.dtype} 到 float32")
            vectors = vectors.astype(np.float32)
        
        logger.info(f"📊 向量數據統計:")
        logger.info(f"  - 形狀: {vectors.shape}")
        logger.info(f"  - 數據類型: {vectors.dtype}")
        logger.info(f"  - 向量數量: {len(vectors)}")
        
        # 安全地計算統計信息
        if len(vectors) > 0:
            try:
                min_val = vectors.min()
                max_val = vectors.max()
                mean_val = vectors.mean()
                std_val = vectors.std()
                
                logger.info(f"  - 範圍: [{min_val:.6f}, {max_val:.6f}]")
                logger.info(f"  - 均值: {mean_val:.6f}")
                logger.info(f"  - 標準差: {std_val:.6f}")
            except Exception as e:
                logger.error(f"❌ 計算向量統計信息失敗: {e}")
                return False
        else:
            logger.error("❌ 沒有有效的向量數據")
            return False
        
        # 載入元數據以獲取 PQ 參數
        index_dir = manager.get_index_dir(collection_name)
        persist = DiskANNPersist(dim=info.dimension)
        meta = persist.load_meta(str(index_dir / "meta.json"))
        
        n_subvectors = meta.get("n_subvectors", 16)
        n_centroids = meta.get("pq_centroids", 256)
        
        logger.info(f"🎯 重新訓練 PQ 模型 ({n_subvectors}×{n_centroids})...")
        
        # 創建新的 PQ 模型
        pq_model = DiskANNPQ(n_subvectors=n_subvectors, n_centroids=n_centroids)
        
        # 訓練模型
        pq_model.fit(vectors, show_progress=True)
        
        # 驗證訓練結果
        logger.info("🔍 驗證 PQ 模型訓練結果...")
        logger.info(f"  - is_fitted: {pq_model.is_fitted}")
        logger.info(f"  - kmeans_list 長度: {len(pq_model.kmeans_list)}")
        logger.info(f"  - means_ 存在: {hasattr(pq_model, 'means_') and pq_model.means_ is not None}")
        logger.info(f"  - stds_ 存在: {hasattr(pq_model, 'stds_') and pq_model.stds_ is not None}")
        
        # 測試編碼
        logger.info("🧪 測試編碼功能...")
        test_vectors = vectors[:5]
        test_codes = pq_model.encode(test_vectors)
        logger.info(f"✅ 測試編碼成功，形狀: {test_codes.shape}")
        
        # 測試解碼
        test_decoded = pq_model.decode(test_codes)
        reconstruction_errors = np.linalg.norm(test_vectors - test_decoded, axis=1)
        avg_error = np.mean(reconstruction_errors)
        logger.info(f"✅ 測試解碼成功，平均重建誤差: {avg_error:.6f}")
        
        # 編碼所有向量
        logger.info("🔢 對所有向量進行 PQ 編碼...")
        pq_codes = pq_model.encode(vectors)
        logger.info(f"✅ 編碼完成，形狀: {pq_codes.shape}")
        
        # 測試編碼一致性
        logger.info("🔍 測試編碼一致性...")
        re_encoded = pq_model.encode(test_vectors)
        if np.array_equal(test_codes, re_encoded):
            logger.info("✅ 編碼一致性檢查通過")
        else:
            logger.error("❌ 編碼一致性檢查失敗")
            return False
        
        # 備份原始文件
        logger.info("💾 備份原始文件...")
        pq_model_path = index_dir / "pq_model.pkl"
        pq_codes_path = index_dir / "pq_codes.bin"
        
        if pq_model_path.exists():
            backup_path = pq_model_path.with_suffix('.pkl.backup')
            pq_model_path.rename(backup_path)
            logger.info(f"  - 備份 PQ 模型: {backup_path}")
        
        if pq_codes_path.exists():
            backup_path = pq_codes_path.with_suffix('.bin.backup')
            pq_codes_path.rename(backup_path)
            logger.info(f"  - 備份 PQ 編碼: {backup_path}")
        
        # 保存新的 PQ 模型和編碼
        logger.info("💾 保存修復後的 PQ 模型...")
        persist.save_pq_codebook(str(pq_model_path), pq_model)
        persist.save_pq_codes(str(pq_codes_path), pq_codes)
        
        # 立即驗證保存和加載
        logger.info("🔍 驗證保存和加載...")
        loaded_pq_model = persist.load_pq_codebook(str(pq_model_path))
        loaded_pq_codes = persist.load_pq_codes(str(pq_codes_path), len(vectors), n_subvectors)
        
        # 驗證加載的模型
        test_codes_loaded = loaded_pq_model.encode(test_vectors)
        if np.array_equal(test_codes, test_codes_loaded):
            logger.info("✅ 加載後編碼一致性檢查通過")
        else:
            logger.error("❌ 加載後編碼一致性檢查失敗")
            return False
        
        # 驗證加載的編碼
        if np.array_equal(pq_codes[:5], loaded_pq_codes[:5]):
            logger.info("✅ PQ 編碼文件一致性檢查通過")
        else:
            logger.error("❌ PQ 編碼文件一致性檢查失敗")
            return False
        
        # 測試距離計算
        logger.info("🎯 測試距離計算...")
        query_vector = test_vectors[0]
        distance_table = loaded_pq_model.compute_distance_table(query_vector)
        
        for i in range(1, 5):
            exact_dist = np.sum((query_vector - test_vectors[i]) ** 2)
            pq_dist = loaded_pq_model.asymmetric_distance_sq(
                test_codes_loaded[i:i+1], distance_table
            )[0]
            ratio = pq_dist / exact_dist if exact_dist > 0 else float('inf')
            logger.info(f"  測試向量 {i}: 精確={exact_dist:.6f}, PQ={pq_dist:.6f}, 比例={ratio:.4f}")
        
        logger.info("🎉 PQ 模型修復完成！")
        return True
        
    except Exception as e:
        logger.error(f"❌ PQ 模型修復失敗: {e}", exc_info=True)
        return False

def main():
    """主函數"""
    if len(sys.argv) != 2:
        print("用法: python pq_debug_test.py <collection_name>")
        sys.exit(1)
    
    collection_name = sys.argv[1]
    
    success = fix_pq_model(collection_name)
    if success:
        print(f"\n✅ 集合 '{collection_name}' 的 PQ 模型修復成功！")
        print("現在可以重新運行診斷測試:")
        print(f"python pq_debug_test.py {collection_name}")
    else:
        print(f"\n❌ 集合 '{collection_name}' 的 PQ 模型修復失敗！")
        sys.exit(1)

if __name__ == "__main__":
    main()