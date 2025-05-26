#!/usr/bin/env python3
"""
驗證資料生成和索引構建的正確性

使用方法：
python scripts/verify_data.py
"""

import os
import json
import numpy as np
import polars as pl
from pathlib import Path
import logging
from typing import Dict, List, Tuple
from openai import OpenAI
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataVerifier:
    def __init__(self,
                 vectors_dir: str = "data/vectors",
                 chunks_dir: str = "data/chunks"):
        self.vectors_dir = Path(vectors_dir)
        self.chunks_dir = Path(chunks_dir)
        self.client = OpenAI()

    def verify_files_exist(self) -> bool:
        """驗證必要的檔案是否存在"""
        required_files = {
            "向量檔案": self.vectors_dir / "vectors.npy",
            "模型資訊": self.vectors_dir / "model_info.json",
            "元資料": self.chunks_dir / "metadata.parquet",
            "索引檔案": self.vectors_dir / "index" / "index.dat",
            "PQ模型": self.vectors_dir / "index" / "pq_model.pkl",
            "PQ編碼": self.vectors_dir / "index" / "pq_codes.bin",
            "索引元資料": self.vectors_dir / "index" / "meta.json"
        }
        
        all_exist = True
        for name, path in required_files.items():
            exists = path.exists()
            logger.info(f"{name}: {'✓' if exists else '✗'} ({path})")
            all_exist = all_exist and exists
            
        return all_exist

    def verify_data_consistency(self) -> Tuple[bool, Dict]:
        """驗證資料一致性"""
        try:
            # 載入資料
            vectors = np.load(self.vectors_dir / "vectors.npy")
            metadata = pl.read_parquet(self.chunks_dir / "metadata.parquet")
            with open(self.vectors_dir / "model_info.json") as f:
                model_info = json.load(f)
            
            # 驗證向量維度
            vector_dim = vectors.shape[1]
            expected_dim = model_info["dimension"]
            dim_match = vector_dim == expected_dim
            
            # 驗證向量數量
            vector_count = len(vectors)
            metadata_count = len(metadata)
            count_match = vector_count == metadata_count
            
            # 驗證模型資訊
            model_match = model_info["num_vectors"] == vector_count
            
            # 驗證元資料完整性
            required_columns = {"id", "text", "image", "section", "manual"}
            columns_match = all(col in metadata.columns for col in required_columns)
            
            # 驗證ID連續性
            ids = metadata["id"].to_list()
            id_continuous = all(i == idx for idx, i in enumerate(ids))
            
            results = {
                "向量維度匹配": dim_match,
                "向量數量匹配": count_match,
                "模型資訊匹配": model_match,
                "元資料列完整": columns_match,
                "ID連續性": id_continuous,
                "向量形狀": vectors.shape,
                "元資料行數": len(metadata),
                "模型維度": expected_dim
            }
            
            return all(results.values()), results
            
        except Exception as e:
            logger.error(f"驗證資料一致性時出錯: {str(e)}")
            return False, {}

    def verify_embeddings(self, sample_size: int = 5) -> bool:
        """驗證向量的一致性（通過重新生成部分向量的方式）"""
        try:
            # 載入資料
            vectors = np.load(self.vectors_dir / "vectors.npy")
            metadata = pl.read_parquet(self.chunks_dir / "metadata.parquet")
            
            # 隨機選擇樣本
            indices = np.random.choice(len(vectors), min(sample_size, len(vectors)), replace=False)
            
            for idx in indices:
                # 獲取原始文字
                text = metadata.filter(pl.col("id") == idx)["text"].item()
                
                # 重新生成向量
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                new_vector = np.array(response.data[0].embedding)
                
                # 計算相似度
                original_vector = vectors[idx]
                similarity = np.dot(original_vector, new_vector) / (
                    np.linalg.norm(original_vector) * np.linalg.norm(new_vector)
                )
                
                logger.info(f"\n驗證樣本 {idx}:")
                logger.info(f"文字: {text[:100]}...")
                logger.info(f"相似度: {similarity:.4f}")
                
                if similarity < 0.99:  # 允許小的數值誤差
                    logger.warning(f"樣本 {idx} 的向量相似度較低")
                    return False
                
                # 避免 API 限制
                time.sleep(1)
            
            return True
            
        except Exception as e:
            logger.error(f"驗證向量時出錯: {str(e)}")
            return False

    def verify_search_results(self, query: str = "元資料儲存在向量資料庫中，並建立索引") -> bool:
        """驗證搜索結果的相關性"""
        try:
            from pydiskann.vamana_graph import beam_search_from_disk
            from pydiskann.io.diskann_persist import MMapNodeReader
            import pickle
            
            # 載入必要的組件
            logger.info("載入搜索組件...")
            
            # 載入模型資訊
            with open(self.vectors_dir / "model_info.json") as f:
                model_info = json.load(f)
            
            # 載入索引讀取器
            reader = MMapNodeReader(
                str(self.vectors_dir / "index" / "index.dat"),
                dim=model_info["dimension"],
                R=32
            )
            
            # 載入 PQ 模型
            with open(self.vectors_dir / "index" / "pq_model.pkl", "rb") as f:
                pq_model = pickle.load(f)
            
            # 載入元資料
            metadata = pl.read_parquet(self.chunks_dir / "metadata.parquet")
            
            # 獲取查詢向量
            logger.info("獲取查詢向量...")
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            query_vector = np.array(response.data[0].embedding, dtype=np.float32)
            
            # 執行搜索
            logger.info("執行搜索...")
            beam_width = 8
            top_k = 5
            
            # 使用 beam_search_from_disk 進行搜索
            results = beam_search_from_disk(
                reader=reader,
                query_vector=query_vector,
                start_id=0,
                beam_width=beam_width,
                k=top_k
            )
            
            if len(results) == 0:
                logger.error("搜索未返回結果")
                return False
            
            # 獲取搜索結果
            search_results = []
            for dist, idx in results:  # 直接解包距離和索引
                # 從元資料中獲取對應的文字資訊
                row = metadata.filter(pl.col("id") == idx).row(0)
                search_results.append({
                    "distance": float(dist),
                    "text": row[metadata.columns.index("text")],
                    "section": row[metadata.columns.index("section")],
                    "manual": row[metadata.columns.index("manual")],
                    "image": row[metadata.columns.index("image")]
                })
            
            # 顯示搜索結果
            logger.info(f"\n搜索查詢: {query}")
            logger.info(f"找到 {len(search_results)} 個結果:")
            
            for i, result in enumerate(search_results, 1):
                logger.info(f"\n{i}. 相關度: {1 - result['distance']:.2%}")
                logger.info(f"   文字: {result['text'][:200]}...")
                logger.info(f"   章節: {result['section']}")
                logger.info(f"   手冊: {result['manual']}")
                if result['image']:
                    logger.info(f"   圖片: {result['image']}")
            
            # 清理資源
            reader.close()
            
            return True
            
        except Exception as e:
            logger.error(f"驗證搜索結果時出錯: {str(e)}")
            if 'reader' in locals():
                reader.close()
            return False

def main():
    verifier = DataVerifier()
    
    print("\n=== 驗證資料檔案 ===")
    if not verifier.verify_files_exist():
        print("\n錯誤: 缺少必要的檔案")
        return
    
    print("\n=== 驗證資料一致性 ===")
    consistency_ok, results = verifier.verify_data_consistency()
    for key, value in results.items():
        print(f"{key}: {'✓' if value else '✗'}")
    
    if not consistency_ok:
        print("\n錯誤: 資料一致性驗證失敗")
        return
    
    print("\n=== 驗證向量一致性 ===")
    if not verifier.verify_embeddings():
        print("\n錯誤: 向量一致性驗證失敗")
        return
    
    print("\n=== 驗證搜索結果 ===")
    if not verifier.verify_search_results():
        print("\n錯誤: 搜索結果驗證失敗")
        return
    
    print("\n✓ 所有驗證通過！")

if __name__ == "__main__":
    main() 