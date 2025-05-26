#!/usr/bin/env python3
"""
DiskANN 搜索測試工具

使用方法：
1. 複製此腳本到您的專案目錄
2. 運行腳本：python search_cli.py <collection_name>

功能：
- 支援多個 collections
- 互動式搜索測試
- 顯示搜索結果和統計資訊
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import time
import argparse
from openai import OpenAI

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from search_engine import SearchEngine
from preprocessing.collection import CollectionManager
from preprocessing.config import CollectionInfo

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False) -> None:
    """設置日誌級別"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def get_embedding_fn(model: str = "text-embedding-3-small") -> callable:
    """獲取 embedding 函數
    
    Args:
        model: OpenAI embedding 模型名稱
        
    Returns:
        callable: 用於生成文本向量的函數
    """
    client = OpenAI()
    
    def embed_text(text: str) -> np.ndarray:
        """生成文本的 embedding 向量"""
        response = client.embeddings.create(
            model=model,
            input=text
        )
        return np.array(response.data[0].embedding)
    
    return embed_text

def list_collections(verbose: bool = False) -> None:
    """列出所有可用的集合"""
    collections = SearchEngine.list_collections()
    if not collections:
        print("沒有找到任何集合")
        return
    
    print("\n可用的集合:")
    print("-" * 80)
    for info in collections:
        print(f"名稱: {info.name}")
        print(f"模型: {info.config['embedding']['model']}")
        print(f"向量數量: {info.num_vectors}")
        print(f"向量維度: {info.dimension}")
        print(f"創建時間: {info.created_at}")
        print(f"更新時間: {info.updated_at}")
        print(f"源文件: {', '.join(info.source_files)}")
        if verbose:
            print(f"配置: {json.dumps(info.config, indent=2, ensure_ascii=False)}")
        print("-" * 80)

def show_collection_info(collection_name: str, verbose: bool = False) -> None:
    """顯示集合的詳細信息"""
    try:
        engine = SearchEngine(collection_name)
        info = engine.get_collection_info()
        
        print(f"\n集合 {collection_name} 的信息:")
        print("-" * 80)
        print(f"模型: {info.config['embedding']['model']}")
        print(f"向量數量: {info.num_vectors}")
        print(f"向量維度: {info.dimension}")
        print(f"創建時間: {info.created_at}")
        print(f"更新時間: {info.updated_at}")
        print(f"源文件: {', '.join(info.source_files)}")
        
        if "index_built_at" in info.chunk_stats:
            print("\n索引信息:")
            print(f"建立時間: {info.chunk_stats['index_built_at']}")
            if "index_params" in info.chunk_stats:
                params = info.chunk_stats["index_params"]
                print(f"圖度數 (R): {params['R']}")
                print(f"搜索列表大小 (L): {params['L']}")
                print(f"剪枝參數 (alpha): {params['alpha']}")
                print(f"線程數: {params['threads']}")
                print(f"PQ 子量化器數: {params['pq_subquantizers']}")
                print(f"PQ 中心點數: {params['pq_centroids']}")
        
        if verbose:
            print("\n完整配置:")
            print(json.dumps(info.config, indent=2, ensure_ascii=False))
        
        print("-" * 80)
        
    except Exception as e:
        logger.error(f"無法獲取集合信息: {str(e)}")
        sys.exit(1)

def interactive_search(collection_name: str, model: str = "text-embedding-3-small") -> None:
    """交互式搜索界面"""
    try:
        engine = SearchEngine(collection_name)
        embed_fn = get_embedding_fn(model)
        
        print(f"\n開始搜索集合 {collection_name}")
        print("輸入 'q' 退出，輸入 'info' 顯示集合信息")
        print("-" * 80)
        
        while True:
            try:
                query = input("\n請輸入搜索文本: ").strip()
                if not query:
                    continue
                
                if query.lower() == 'q':
                    break
                
                if query.lower() == 'info':
                    show_collection_info(collection_name)
                    continue
                
                # 執行搜索並獲取結果
                search_response = engine.search(query, k=5, embedding_fn=embed_fn)
                results = search_response["results"]
                timing = search_response["timing"]
                
                if not results:
                    print("沒有找到相關結果")
                    continue
                
                print("\n搜索結果:")
                print("-" * 80)
                # 顯示時間統計
                print(f"Embedding API 調用時間: {timing['embedding_time']:.3f} 秒")
                print(f"DiskANN 搜索時間: {timing['search_time']:.3f} 秒")
                print(f"總耗時: {timing['total_time']:.3f} 秒")
                print("-" * 40)
                
                # 顯示搜索結果
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. 相似度: {1 - result['distance']:.4f}")
                    print(f"文本: {result['text']}")
                    if "metadata" in result and result["metadata"]:
                        print("元數據:")
                        print(json.dumps(result["metadata"], indent=2, ensure_ascii=False))
                    print("-" * 40)
                
            except KeyboardInterrupt:
                print("\n退出搜索")
                break
            except Exception as e:
                logger.error(f"搜索出錯: {str(e)}")
                continue
        
    except Exception as e:
        logger.error(f"初始化搜索引擎失敗: {str(e)}")
        sys.exit(1)

def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="向量搜索引擎")
    parser.add_argument("collection", nargs="?", help="要搜索的集合名稱")
    parser.add_argument("--list", action="store_true", help="列出所有可用的集合")
    parser.add_argument("--info", action="store_true", help="顯示集合的詳細信息")
    parser.add_argument("--model", default="text-embedding-3-small", help="OpenAI embedding 模型名稱")
    parser.add_argument("--verbose", action="store_true", help="顯示詳細信息")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    try:
        if args.list:
            list_collections(args.verbose)
        elif args.info:
            if not args.collection:
                logger.error("必須指定集合名稱")
                sys.exit(1)
            show_collection_info(args.collection, args.verbose)
        elif args.collection:
            interactive_search(args.collection, args.model)
        else:
            parser.print_help()
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"程序執行失敗: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 