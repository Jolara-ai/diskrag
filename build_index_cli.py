#!/usr/bin/env python3
"""
DiskANN 索引構建工具
用於構建和管理 collections 的索引
"""

import argparse
import logging
import sys
from pathlib import Path
import time
import json
import numpy as np
import polars as pl
from pydiskann.vamana_graph import build_vamana
from pydiskann.pq.pq_model import SimplePQ
from pydiskann.io.diskann_persist import DiskANNPersist, MMapNodeReader

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def build_index(collection_name: str, R: int = 32, L: int = 64, alpha: float = 1.2, threads: int = 4, verbose: bool = False):
    """構建指定 collection 的索引"""
    try:
        # 設置路徑
        collection_path = Path("collections") / collection_name
        vectors_path = collection_path / "vectors.npy"
        index_path = collection_path / "index"
        model_info_path = collection_path / "model_info.json"
        metadata_path = collection_path / "metadata.parquet"
        
        # 檢查必要文件
        if not vectors_path.exists():
            raise ValueError(f"向量文件不存在: {vectors_path}")
        if not model_info_path.exists():
            raise ValueError(f"模型信息文件不存在: {model_info_path}")
        if not metadata_path.exists():
            raise ValueError(f"元數據文件不存在: {metadata_path}")
        
        # 檢查索引是否已存在
        if (index_path / "index.dat").exists():
            raise ValueError(f"索引已存在: {index_path}")
        
        # 載入數據
        logger.info(f"載入 collection '{collection_name}' 的數據...")
        vectors = np.load(vectors_path)
        with open(model_info_path) as f:
            model_info = json.load(f)
        
        # 確保向量數據是 float32 類型
        vectors = vectors.astype(np.float32)
        
        # 構建 Vamana 圖
        logger.info("構建 Vamana 圖...")
        logger.info(f"向量數據形狀: {vectors.shape}")
        logger.info(f"使用參數: R={R} (圖的最大度數)")
        
        graph = build_vamana(vectors, R=R)
        logger.info(f"Vamana 圖構建完成，節點數: {len(graph.nodes)}")
        
        # 構建 PQ 模型
        logger.info("\n構建 PQ 模型...")
        logger.info("使用參數: n_subvectors=8 (子向量數量)")
        
        pq = SimplePQ(n_subvectors=8)
        pq.fit(vectors)
        
        # 對向量進行 PQ 編碼
        logger.info("對向量進行 PQ 編碼...")
        pq_codes = pq.encode(vectors)
        logger.info(f"PQ 編碼完成，編碼形狀: {pq_codes.shape}")
        
        # 創建索引目錄
        index_path.mkdir(parents=True, exist_ok=True)
        
        # 保存索引和模型
        logger.info("\n保存索引和模型...")
        persist = DiskANNPersist(dim=model_info["dimension"], R=R)
        
        # 保存 Vamana 圖
        logger.info("保存 Vamana 圖...")
        persist.save_index(str(index_path / "index.dat"), graph)
        
        # 保存元數據
        logger.info("保存索引元數據...")
        persist.save_meta(str(index_path / "meta.json"), {
            "D": model_info["dimension"],
            "R": R,
            "N": len(vectors),
            "n_subvectors": 8,
            "build_time": time.time() - start_time
        })
        
        # 保存 PQ 編碼和模型
        logger.info("保存 PQ 編碼和模型...")
        persist.save_pq_codes(str(index_path / "pq_codes.bin"), pq_codes)
        persist.save_pq_codebook(str(index_path / "pq_model.pkl"), pq)
        
        logger.info(f"\n索引構建完成！")
        logger.info(f"Collection: {collection_name}")
        logger.info(f"向量數量: {len(vectors)}")
        logger.info(f"向量維度: {model_info['dimension']}")
        logger.info(f"索引路徑: {index_path}")
        
    except Exception as e:
        logger.error(f"構建索引時出錯: {str(e)}")
        sys.exit(1)

def verify_index(collection_name: str):
    """驗證指定 collection 的索引"""
    try:
        # 設置路徑
        collection_path = Path("collections") / collection_name
        index_path = collection_path / "index"
        
        # 檢查必要文件
        required_files = [
            "index.dat",
            "meta.json",
            "pq_codes.bin",
            "pq_model.pkl"
        ]
        
        missing_files = [
            f for f in required_files
            if not (index_path / f).exists()
        ]
        
        if missing_files:
            raise ValueError(f"缺少必要的索引文件: {', '.join(missing_files)}")
        
        # 載入元數據
        with open(index_path / "meta.json") as f:
            meta = json.load(f)
        
        # 驗證索引
        logger.info(f"驗證 collection '{collection_name}' 的索引...")
        reader = MMapNodeReader(
            str(index_path / "index.dat"),
            dim=meta["D"],
            R=meta["R"]
        )
        
        # 驗證 PQ 模型
        pq_codes = np.fromfile(str(index_path / "pq_codes.bin"), dtype=np.uint8)
        pq_codes = pq_codes.reshape((meta["N"], meta["n_subvectors"]))
        
        logger.info("索引驗證完成！")
        logger.info(f"Collection: {collection_name}")
        logger.info(f"向量數量: {meta['N']}")
        logger.info(f"向量維度: {meta['D']}")
        logger.info(f"圖的度數: {meta['R']}")
        logger.info(f"PQ 子向量數: {meta['n_subvectors']}")
        
        reader.close()
        
    except Exception as e:
        logger.error(f"驗證索引時出錯: {str(e)}")
        sys.exit(1)

def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="DiskANN 索引構建工具")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # build 命令
    build_parser = subparsers.add_parser("build", help="構建索引")
    build_parser.add_argument("collection", help="collection 名稱")
    build_parser.add_argument("--R", type=int, default=32, help="圖的度數")
    build_parser.add_argument("--L", type=int, default=64, help="搜索列表大小")
    build_parser.add_argument("--alpha", type=float, default=1.2, help="圖剪枝參數")
    build_parser.add_argument("--threads", type=int, default=4, help="使用的線程數")
    build_parser.add_argument("--verbose", action="store_true", help="顯示詳細日誌")
    
    # verify 命令
    verify_parser = subparsers.add_parser("verify", help="驗證索引")
    verify_parser.add_argument("collection", help="collection 名稱")
    
    args = parser.parse_args()
    
    if args.command == "build":
        build_index(
            args.collection,
            R=args.R,
            L=args.L,
            alpha=args.alpha,
            threads=args.threads,
            verbose=args.verbose
        )
    elif args.command == "verify":
        verify_index(args.collection)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 