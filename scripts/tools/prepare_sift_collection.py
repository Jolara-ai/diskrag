#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
準備SIFT數據集為DiskRAG collection格式
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.collection import CollectionManager
from preprocessing.config import CollectionInfo

def load_sift_vectors(parquet_path: str) -> np.ndarray:
    """
    從parquet文件載入SIFT向量
    
    Args:
        parquet_path: parquet文件路徑
    
    Returns:
        向量數組 (N, 128)
    """
    print(f"載入SIFT向量: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"數據形狀: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # 嘗試不同的列名來找到向量數據
    if 'vector' in df.columns:
        vectors = np.stack(df['vector'].values)
    elif 'embedding' in df.columns:
        vectors = np.stack(df['embedding'].values)
    elif 'features' in df.columns:
        vectors = np.stack(df['features'].values)
    else:
        # 嘗試找到包含數值數據的列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 128:
            vectors = df[numeric_cols].values.astype(np.float32)
        else:
            # 如果沒有128列，嘗試其他方法
            print(f"警告: 沒有找到128維向量數據")
            print(f"數值列數量: {len(numeric_cols)}")
            print(f"數值列: {numeric_cols.tolist()}")
            
            # 嘗試將所有數值列作為向量
            if len(numeric_cols) > 0:
                vectors = df[numeric_cols].values.astype(np.float32)
                print(f"使用所有數值列作為向量，維度: {vectors.shape[1]}")
            else:
                raise ValueError("無法找到向量數據")
    
    print(f"向量形狀: {vectors.shape}")
    print(f"向量類型: {vectors.dtype}")
    print(f"向量範圍: [{np.min(vectors):.6f}, {np.max(vectors):.6f}]")
    
    return vectors

def create_sift_collection(
    collection_name: str,
    train_path: str,
    test_path: str = None,
    max_train_samples: int = None,
    max_test_samples: int = None
) -> bool:
    """
    創建SIFT collection
    
    Args:
        collection_name: collection名稱
        train_path: 訓練數據路徑
        test_path: 測試數據路徑（可選）
        max_train_samples: 最大訓練樣本數
        max_test_samples: 最大測試樣本數
    
    Returns:
        是否成功創建
    """
    print(f"創建SIFT collection: {collection_name}")
    
    # 載入訓練數據
    train_vectors = load_sift_vectors(train_path)
    
    # 限制訓練樣本數
    if max_train_samples and len(train_vectors) > max_train_samples:
        print(f"限制訓練樣本數為 {max_train_samples}")
        indices = np.random.choice(len(train_vectors), max_train_samples, replace=False)
        train_vectors = train_vectors[indices]
    
    # 創建collection manager
    manager = CollectionManager()
    
    # 檢查collection是否已存在
    if manager.get_collection_info(collection_name):
        print(f"Collection '{collection_name}' 已存在")
        response = input("是否要覆蓋？(y/N): ")
        if response.lower() != 'y':
            print("取消創建")
            return False
    
    # 創建collection目錄
    collection_dir = manager._get_collection_dir(collection_name)
    collection_dir.mkdir(parents=True, exist_ok=True)
    
    # 創建collection
    print(f"創建collection: {collection_name}")
    collection_info = CollectionInfo(
        name=collection_name,
        dimension=train_vectors.shape[1],
        num_vectors=len(train_vectors),
        created_at=pd.Timestamp.now().isoformat(),
        updated_at=pd.Timestamp.now().isoformat(),
        source_files=[train_path],
        config={
            "description": f"SIFT數據集 - {len(train_vectors):,}個向量",
            "type": "sift_dataset"
        },
        chunk_stats={
            "total_chunks": len(train_vectors),
            "avg_chunk_size": train_vectors.shape[1],
            "created_at": pd.Timestamp.now().isoformat()
        }
    )
    
    # 保存collection信息
    manager.save_collection_info(collection_name, collection_info)
    
    # 保存向量數據
    vectors_path = manager.get_vectors_path(collection_name)
    print(f"保存向量到: {vectors_path}")
    np.save(str(vectors_path), train_vectors.astype(np.float32))
    
    # 創建metadata
    metadata = []
    for i in range(len(train_vectors)):
        metadata.append({
            "id": i,
            "text": f"SIFT_vector_{i:06d}",
            "text_hash": f"sift_{i:06d}",
            "chunk_id": i,
            "chunk_index": 0,
            "embedding": train_vectors[i].tolist()
        })
    
    # 保存metadata
    metadata_path = manager.get_metadata_path(collection_name)
    print(f"保存metadata到: {metadata_path}")
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_parquet(str(metadata_path), index=False)
    
    print(f"✅ SIFT collection創建成功")
    print(f"  名稱: {collection_name}")
    print(f"  向量數量: {len(train_vectors):,}")
    print(f"  向量維度: {train_vectors.shape[1]}")
    
    # 如果提供了測試數據，也保存測試向量
    if test_path:
        print(f"\n處理測試數據...")
        test_vectors = load_sift_vectors(test_path)
        
        if max_test_samples and len(test_vectors) > max_test_samples:
            print(f"限制測試樣本數為 {max_test_samples}")
            indices = np.random.choice(len(test_vectors), max_test_samples, replace=False)
            test_vectors = test_vectors[indices]
        
        # 保存測試向量到單獨的文件
        test_vectors_path = Path(f"collections/{collection_name}/test_vectors.npy")
        test_vectors_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(test_vectors_path), test_vectors.astype(np.float32))
        
        print(f"測試向量保存到: {test_vectors_path}")
        print(f"測試向量數量: {len(test_vectors):,}")
    
    return True

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='準備SIFT數據集為DiskRAG collection')
    parser.add_argument('--collection-name', type=str, default='sift500k',
                       help='collection名稱')
    parser.add_argument('--train-path', type=str,
                       default='dataset/sift_small_500k/train_fixed.parquet',
                       help='訓練數據路徑')
    parser.add_argument('--test-path', type=str,
                       default='dataset/sift_small_500k/test_fixed.parquet',
                       help='測試數據路徑')
    parser.add_argument('--max-train-samples', type=int, default=None,
                       help='最大訓練樣本數')
    parser.add_argument('--max-test-samples', type=int, default=10000,
                       help='最大測試樣本數')
    
    args = parser.parse_args()
    
    # 檢查輸入文件是否存在
    if not os.path.exists(args.train_path):
        print(f"❌ 訓練數據文件不存在: {args.train_path}")
        return 1
    
    if args.test_path and not os.path.exists(args.test_path):
        print(f"❌ 測試數據文件不存在: {args.test_path}")
        return 1
    
    # 創建collection
    success = create_sift_collection(
        collection_name=args.collection_name,
        train_path=args.train_path,
        test_path=args.test_path,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples
    )
    
    if success:
        print("\n🎉 SIFT collection準備完成！")
        print(f"現在可以使用以下命令建置索引:")
        print(f"python build_index.py {args.collection_name}")
        return 0
    else:
        print("\n❌ SIFT collection準備失敗")
        return 1

if __name__ == "__main__":
    exit(main()) 