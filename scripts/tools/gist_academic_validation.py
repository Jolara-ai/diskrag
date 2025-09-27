#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DiskRAG Academic Validation on GIST-1M Dataset
學術級 GIST-1M 數據集驗證腳本

使用方法:
1. 解壓 gist.tar.gz 到 data/gist/ 目錄
2. python gist_academic_validation.py --scale small_scale
3. 查看生成的學術報告
"""

import numpy as np
import argparse
import logging
import time
import json
from pathlib import Path
import sys
import os
from tqdm import tqdm

# 導入您的模組
try:
    from pydiskann.pq.fast_pq import DiskANNPQ
    from pydiskann.vamana_graph import build_vamana
    from pydiskann.io.diskann_persist import DiskANNPersist
    from search_engine import SearchEngine
    from preprocessing.collection import CollectionManager
except ImportError as e:
    print(f"錯誤: 無法導入模組: {e}")
    sys.exit(1)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("警告: 未安裝 Faiss，將無法計算動態 Ground Truth")

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GISTAcademicValidation")

# 學術評估標準配置
ACADEMIC_SCALES = {
    "small_scale": {
        "base_size": 100000,
        "learn_size": 50000,
        "query_size": 1000,
        "description": "100K 基礎向量驗證"
    },
    "medium_scale": {
        "base_size": 300000,
        "learn_size": 100000,
        "query_size": 1000,
        "description": "300K 基礎向量驗證"
    },
    "large_scale": {
        "base_size": 500000,
        "learn_size": 150000,
        "query_size": 1000,
        "description": "500K 基礎向量驗證"
    },
    "full_scale": {
        "base_size": 1000000,
        "learn_size": 500000,
        "query_size": 1000,
        "description": "完整 1M 基礎向量驗證"
    }
}

# DiskANN 論文標準配置
DISKANN_CONFIGS = {
    "baseline": {
        "R": 32,
        "L_build": 64,
        "pq_m": 16,
        "description": "基礎配置"
    },
    "balanced": {
        "R": 48,
        "L_build": 100,
        "pq_m": 24,
        "description": "平衡配置（推薦）"
    },
    "high_recall": {
        "R": 64,
        "L_build": 128,
        "pq_m": 32,
        "description": "高召回率配置"
    }
}

# 學術評估標準 L_search 範圍
ACADEMIC_L_SEARCH = [10, 20, 50, 100, 200, 500, 800, 1000, 1500, 2000, 3000,5000,10000]

def read_fvecs(filename):
    """讀取 .fvecs 格式文件"""
    with open(filename, 'rb') as f:
        fv = np.fromfile(f, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    fv = fv.reshape(-1, dim + 1)
    return fv[:, 1:].copy()

def read_ivecs(filename):
    """讀取 .ivecs 格式文件"""
    with open(filename, 'rb') as f:
        iv = np.fromfile(f, dtype=np.int32)
    if iv.size == 0:
        return np.zeros((0, 0))
    dim = iv[0]
    iv = iv.reshape(-1, dim + 1)
    return iv[:, 1:]

def verify_gist_data(data_dir):
    """驗證 GIST 數據集完整性"""
    data_path = Path(data_dir)
    
    required_files = {
        "gist_base.fvecs": (1000000, 960),
        "gist_learn.fvecs": (500000, 960),
        "gist_query.fvecs": (1000, 960),
        "gist_groundtruth.ivecs": (1000, 100)
    }
    
    logger.info("🔍 驗證 GIST 數據集...")
    
    for filename, (expected_count, expected_dim) in required_files.items():
        filepath = data_path / filename
        if not filepath.exists():
            logger.error(f"❌ 缺少文件: {filename}")
            return False
            
        try:
            if filename.endswith('.fvecs'):
                data = read_fvecs(str(filepath))
            else:
                data = read_ivecs(str(filepath))
                
            actual_count, actual_dim = data.shape
            logger.info(f"✅ {filename}: {actual_count} × {actual_dim}")
            
            if actual_count != expected_count:
                logger.warning(f"⚠️  {filename} 數量: 預期 {expected_count}, 實際 {actual_count}")
            if actual_dim != expected_dim:
                logger.warning(f"⚠️  {filename} 維度: 預期 {expected_dim}, 實際 {actual_dim}")
                
        except Exception as e:
            logger.error(f"❌ 讀取 {filename} 失敗: {e}")
            return False
    
    logger.info("✅ GIST 數據集驗證完成")
    return True

def compute_ground_truth(base_vectors, query_vectors, k=100):
    """計算精確的 Ground Truth"""
    if not FAISS_AVAILABLE:
        raise ImportError("需要安裝 Faiss: pip install faiss-cpu")
    
    logger.info(f"🧬 計算 Ground Truth (k={k})...")
    d = base_vectors.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(base_vectors.astype(np.float32))
    _, gt_indices = index.search(query_vectors.astype(np.float32), k)
    logger.info("✅ Ground Truth 計算完成")
    return gt_indices

def build_academic_index(collection_name, base_vectors, learn_vectors, config):
    """建立學術驗證索引"""
    logger.info(f"🔧 建立學術索引: {collection_name}")
    
    base_vectors = base_vectors.astype(np.float32)
    learn_vectors = learn_vectors.astype(np.float32)
    
    build_start_time = time.time()
    n_points, dimension = base_vectors.shape
    
    # 管理 collection
    manager = CollectionManager()
    if manager.get_collection_info(collection_name):
        logger.warning(f"⚠️  Collection '{collection_name}' 已存在，將被覆蓋")
        manager.delete_collection(collection_name)
    
    manager.create_collection(
        collection_name=collection_name,
        config={},
        dimension=dimension,
        source_files=[]
    )
    index_dir = manager.get_index_dir(collection_name)
    os.makedirs(index_dir, exist_ok=True)
    
    # 訓練 PQ 模型
    logger.info(f"📚 訓練 PQ 模型 (m={config['pq_m']})...")
    pq_start_time = time.time()
    pq_model = DiskANNPQ(n_subvectors=config['pq_m'])
    pq_model.fit(learn_vectors, show_progress=True)
    pq_train_time = time.time() - pq_start_time
    
    # 編碼向量
    logger.info("🔢 PQ 編碼...")
    encode_start_time = time.time()
    pq_codes = pq_model.encode(base_vectors)
    pq_encode_time = time.time() - encode_start_time
    
    # 建立圖
    logger.info(f"🕸️  建立 Vamana 圖 (R={config['R']}, L={config['L_build']})...")
    graph_start_time = time.time()
    graph = build_vamana(base_vectors, R=config['R'], L=config['L_build'], 
                        alpha=1.2, show_progress=True)
    graph_build_time = time.time() - graph_start_time
    
    # 保存索引
    logger.info("💾 保存索引...")
    persist_start_time = time.time()
    persist = DiskANNPersist(dim=dimension, R=config['R'])
    persist.save_pq_codebook(str(index_dir / "pq_model.pkl"), pq_model)
    persist.save_pq_codes(str(index_dir / "pq_codes.bin"), pq_codes)
    persist.save_index(str(index_dir / "index.dat"), graph)
    
    medoid_idx = getattr(graph, 'medoid_idx', 0)
    meta_info = {
        "D": int(dimension), "R": int(config['R']), "L": int(config['L_build']),
        "alpha": 1.2, "N": int(n_points), "medoid_idx": int(medoid_idx),
        "n_subvectors": int(config['pq_m']), "pq_centroids": 256
    }
    persist.save_meta(str(index_dir / "meta.json"), meta_info)
    
    # 創建元數據（學術驗證用）
    import polars as pl
    metadata_df = pl.DataFrame({
        "text": [f"gist_vector_{i}" for i in range(n_points)],
        "text_hash": [f"gist_hash_{i}" for i in range(n_points)],
        "metadata": [json.dumps({"id": i, "type": "gist_feature"}) for i in range(n_points)],
        "vector_index": list(range(n_points))
    })
    metadata_df.write_parquet(manager.get_metadata_path(collection_name))
    
    # 更新 collection info
    info = manager.get_collection_info(collection_name)
    info.num_vectors = n_points
    info.updated_at = time.strftime('%Y-%m-%dT%H:%M:%S')
    manager.save_collection_info(collection_name, info)
    
    persist_time = time.time() - persist_start_time
    total_build_time = time.time() - build_start_time
    
    build_stats = {
        "pq_train_time": pq_train_time,
        "pq_encode_time": pq_encode_time,
        "graph_build_time": graph_build_time,
        "persist_time": persist_time,
        "total_build_time": total_build_time
    }
    
    logger.info(f"✅ 索引建立完成 (總耗時: {total_build_time:.2f}s)")
    return build_stats

def evaluate_academic_performance(collection_name, query_vectors, ground_truth, k=10):
    """執行學術級性能評估"""
    logger.info(f"🔍 開始學術級性能評估...")
    
    engine = SearchEngine(collection_name)
    results = []
    
    for L_search in tqdm(ACADEMIC_L_SEARCH, desc="評估不同 L_search 值"):
        if L_search < k:
            continue
            
        logger.info(f"  評估 L_search={L_search}...")
        
        # 預熱
        warmup_queries = min(50, len(query_vectors))
        for i in range(warmup_queries):
            engine.search(
                query="", k=k, 
                embedding_fn=lambda x: query_vectors[i],
                L_search=L_search, use_pq_search=True
            )
        
        # 正式評估
        latencies = []
        total_recall = 0.0
        
        for i in range(len(query_vectors)):
            query_vector = query_vectors[i]
            gt_ids = set(ground_truth[i, :k])
            
            start_time = time.perf_counter()
            search_results = engine.search(
                query="", k=k,
                embedding_fn=lambda x: query_vector,
                L_search=L_search, use_pq_search=True
            )
            latencies.append(time.perf_counter() - start_time)
            
            # 提取返回的 ID
            returned_ids = {
                res["metadata"]["id"] 
                for res in search_results.get("results", []) 
                if "id" in res.get("metadata", {})
            }
            
            # 計算 recall
            recall = len(gt_ids.intersection(returned_ids)) / k
            total_recall += recall
        
        avg_recall = total_recall / len(query_vectors)
        avg_latency_ms = (sum(latencies) / len(latencies)) * 1000
        qps = 1000 / avg_latency_ms
        
        result = {
            "L_search": L_search,
            "recall": avg_recall,
            "latency_ms": avg_latency_ms,
            "qps": qps
        }
        
        results.append(result)
        logger.info(f"    Recall: {avg_recall:.4f}, Latency: {avg_latency_ms:.2f}ms, QPS: {qps:.1f}")
    
    return results

def generate_academic_report(scale_config, diskann_config, build_stats, eval_results, 
                           dataset_info):
    """生成學術驗證報告"""
    
    # 計算關鍵學術指標
    academic_metrics = {}
    
    # 找到達到特定 recall 閾值的最小 L
    for target_recall in [0.8, 0.85, 0.9, 0.95]:
        for result in eval_results:
            if result["recall"] >= target_recall:
                key = f"recall_{int(target_recall*100)}"
                if key not in academic_metrics:
                    academic_metrics[key] = {
                        "min_L_search": result["L_search"],
                        "actual_recall": result["recall"],
                        "latency_ms": result["latency_ms"],
                        "qps": result["qps"]
                    }
                break
    
    # 最佳性能
    best_result = max(eval_results, key=lambda x: x["recall"])
    academic_metrics["best_performance"] = {
        "max_recall": best_result["recall"],
        "L_search": best_result["L_search"],
        "latency_ms": best_result["latency_ms"],
        "qps": best_result["qps"]
    }
    
    # 生成完整報告
    report = {
        "experiment_metadata": {
            "dataset": "GIST-1M",
            "system": "DiskRAG (DiskANN + PQ Implementation)",
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scale": scale_config["description"],
            "configuration": diskann_config["description"]
        },
        "dataset_statistics": dataset_info,
        "index_parameters": {
            "R": diskann_config["R"],
            "L_build": diskann_config["L_build"],
            "pq_subvectors": diskann_config["pq_m"],
            "pq_centroids": 256,
            "alpha": 1.2
        },
        "build_performance": build_stats,
        "search_performance": eval_results,
        "academic_metrics": academic_metrics,
        "diskann_paper_comparison": {
            "note": "與 DiskANN 原論文在 GIST-1M 上的對比",
            "paper_recall_90": "~1000 L_search",
            "paper_recall_95": "~2000 L_search",
            "your_recall_90": academic_metrics.get("recall_90", {}).get("min_L_search", "N/A"),
            "your_recall_95": academic_metrics.get("recall_95", {}).get("min_L_search", "N/A")
        }
    }
    
    return report

def print_academic_summary(report):
    """打印學術驗證摘要"""
    print("\n" + "="*80)
    print("🏆 DiskRAG 學術驗證報告摘要")
    print("="*80)
    
    print(f"📊 數據集: {report['experiment_metadata']['dataset']}")
    print(f"📏 規模: {report['experiment_metadata']['scale']}")
    print(f"⚙️  配置: {report['experiment_metadata']['configuration']}")
    
    print(f"\n📈 關鍵性能指標:")
    metrics = report["academic_metrics"]
    
    best = metrics["best_performance"]
    print(f"  🥇 最佳 Recall: {best['max_recall']:.4f} @ L={best['L_search']}")
    print(f"     延遲: {best['latency_ms']:.2f}ms, QPS: {best['qps']:.1f}")
    
    for recall_level in [80, 85, 90, 95]:
        key = f"recall_{recall_level}"
        if key in metrics:
            data = metrics[key]
            print(f"  📌 Recall≥{recall_level}%: L≥{data['min_L_search']}, "
                  f"實際={data['actual_recall']:.4f}, QPS={data['qps']:.1f}")
    
    print(f"\n🔬 與 DiskANN 論文對比:")
    comparison = report["diskann_paper_comparison"]
    print(f"  Recall≥90%: 論文~L=1000, 您的={comparison['your_recall_90']}")
    print(f"  Recall≥95%: 論文~L=2000, 您的={comparison['your_recall_95']}")
    
    print(f"\n⏱️  建立性能:")
    build = report["build_performance"]
    print(f"  總建立時間: {build['total_build_time']:.1f}s")
    print(f"  PQ 訓練: {build['pq_train_time']:.1f}s")
    print(f"  圖建立: {build['graph_build_time']:.1f}s")
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(
        description="DiskRAG 在 GIST-1M 數據集上的學術驗證",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data_dir', default='data/gist', 
                       help='GIST 數據集目錄')
    parser.add_argument('--scale', choices=list(ACADEMIC_SCALES.keys()),
                       default='small_scale', help='評估規模')
    parser.add_argument('--config', choices=list(DISKANN_CONFIGS.keys()),
                       default='balanced', help='DiskANN 配置')
    parser.add_argument('--output', default='gist_academic_report.json',
                       help='學術報告輸出文件')
    parser.add_argument('--k', type=int, default=10,
                       help='Top-K 評估')
    
    args = parser.parse_args()
    
    try:
        # 驗證數據
        if not verify_gist_data(args.data_dir):
            logger.error("❌ GIST 數據集驗證失敗")
            return 1
        
        # 載入配置
        scale_config = ACADEMIC_SCALES[args.scale]
        diskann_config = DISKANN_CONFIGS[args.config]
        
        logger.info(f"🚀 開始學術驗證:")
        logger.info(f"  規模: {scale_config['description']}")
        logger.info(f"  配置: {diskann_config['description']}")
        
        # 載入 GIST 數據
        data_dir = Path(args.data_dir)
        logger.info("📚 載入 GIST 數據...")
        
        base_full = read_fvecs(data_dir / "gist_base.fvecs")
        learn_full = read_fvecs(data_dir / "gist_learn.fvecs")
        query_full = read_fvecs(data_dir / "gist_query.fvecs")
        gt_full = read_ivecs(data_dir / "gist_groundtruth.ivecs")
        
        # 按規模截取
        base_vectors = base_full[:scale_config["base_size"]]
        learn_vectors = learn_full[:scale_config["learn_size"]]
        query_vectors = query_full[:scale_config["query_size"]]
        
        # 處理 Ground Truth
        if scale_config["base_size"] < len(base_full):
            logger.info("🧬 重新計算子集 Ground Truth...")
            ground_truth = compute_ground_truth(base_vectors, query_vectors, k=100)
        else:
            ground_truth = gt_full[:scale_config["query_size"]]
        
        dataset_info = {
            "base_vectors": len(base_vectors),
            "learn_vectors": len(learn_vectors),
            "query_vectors": len(query_vectors),
            "dimension": base_vectors.shape[1],
            "ground_truth_k": ground_truth.shape[1]
        }
        
        logger.info(f"✅ 數據載入完成: {dataset_info}")
        
        # 建立索引
        collection_name = f"gist_academic_{args.scale}_{args.config}"
        build_stats = build_academic_index(
            collection_name, base_vectors, learn_vectors, diskann_config
        )
        
        # 評估性能
        eval_results = evaluate_academic_performance(
            collection_name, query_vectors, ground_truth, k=args.k
        )
        
        # 生成報告
        report = generate_academic_report(
            scale_config, diskann_config, build_stats, eval_results, dataset_info
        )
        
        # 保存報告
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 學術報告已保存: {args.output}")
        
        # 打印摘要
        print_academic_summary(report)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 學術驗證失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())