import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
import numpy as np
from pydiskann.vamana_graph import build_vamana
from pydiskann.pq.pq_model import SimplePQ
from pydiskann.io.diskann_persist import DiskANNPersist
from preprocessing.collection import CollectionManager
from preprocessing.config import CollectionInfo
from datetime import datetime

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False) -> None:
    """設置日誌級別"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def build_index(
    collection_name: str,
    R: int = 32,  # 標準化為 32，與 app.py 一致
    L: Optional[int] = None,  # 不再使用 L 參數
    alpha: Optional[float] = None,  # 不再使用 alpha 參數
    threads: int = 1,  # 保留但標記為未使用
    verbose: bool = False
) -> None:
    """建立 DiskANN 索引
    
    Args:
        collection_name: collection 名稱
        R: 圖的度數，固定為 32
        L: 搜索列表大小（不再使用）
        alpha: 剪枝參數（不再使用）
        threads: 線程數（目前未使用）
        verbose: 是否顯示詳細日誌
    """
    try:
        # 設置日誌
        setup_logging(verbose)
        
        # 初始化 collection manager
        manager = CollectionManager()
        
        # 獲取 collection 信息
        info = manager.get_collection_info(collection_name)
        if not info:
            raise ValueError(f"找不到 collection {collection_name}")
            
        # 獲取向量數據
        vectors_path = manager.get_vectors_path(collection_name)
        if not vectors_path.exists():
            raise ValueError(f"找不到向量文件: {vectors_path}")
            
        vectors = np.load(str(vectors_path))
        logger.info(f"載入向量數據: {vectors.shape}")
        
        # 獲取索引目錄
        index_dir = manager.get_index_dir(collection_name)
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # 建立 PQ 模型
        logger.info("訓練 PQ 模型...")
        pq_model = SimplePQ(n_subvectors=8)  # 使用 8 個子量化器，每個使用 16 個中心點
        pq_model.fit(vectors)
        
        # 對向量進行 PQ 編碼
        logger.info("對向量進行 PQ 編碼...")
        pq_codes = pq_model.encode(vectors)
        logger.info(f"PQ 編碼完成，編碼形狀: {pq_codes.shape}")
        
        # 保存 PQ 模型和編碼
        persist = DiskANNPersist(dim=vectors.shape[1], R=R)
        pq_path = index_dir / "pq_model.pkl"
        pq_codes_path = index_dir / "pq_codes.bin"
        persist.save_pq_codebook(str(pq_path), pq_model)
        persist.save_pq_codes(str(pq_codes_path), pq_codes)
        logger.info(f"保存 PQ 模型到: {pq_path}")
        logger.info(f"保存 PQ 編碼到: {pq_codes_path}")
        
        # 建立 Vamana 圖
        logger.info("建立 Vamana 圖...")
        graph = build_vamana(vectors, R=R)
        persist.save_index(str(index_dir / "index.dat"), graph)
        logger.info("Vamana 圖建立完成")
        
        # 保存元數據
        persist.save_meta(str(index_dir / "meta.json"), {
            "D": vectors.shape[1],
            "R": R,
            "N": len(vectors),
            "n_subvectors": 8,
            "build_time": datetime.now().isoformat()
        })
        
        # 更新 collection 信息
        info.chunk_stats.update({
            "index_built_at": datetime.now().isoformat(),
            "index_params": {
                "R": R,
                "pq_subquantizers": 8,
                "pq_centroids": 16,  # 更新為實際使用的中心點數
                "threads": threads
            }
        })
        manager.save_collection_info(collection_name, info)
        logger.info("更新 collection 信息完成")
        
    except Exception as e:
        logger.error(f"建立索引時出錯: {str(e)}")
        sys.exit(1)

def main() -> None:
    """主函數"""
    parser = argparse.ArgumentParser(description="建立 DiskANN 索引")
    parser.add_argument("--collection", required=True, help="collection 名稱")
    parser.add_argument("--R", type=int, default=32, help="圖的度數（固定為 32）")
    parser.add_argument("--threads", type=int, default=1, help="線程數（目前未使用）")
    parser.add_argument("--verbose", action="store_true", help="顯示詳細日誌")
    
    args = parser.parse_args()
    
    # 移除不再使用的參數
    if args.R != 32:
        logger.warning("R 參數已固定為 32，忽略指定的值")
    
    build_index(
        collection_name=args.collection,
        R=32,  # 固定使用 32
        threads=args.threads,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main() 