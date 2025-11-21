#!/usr/bin/env python3
"""
Simplified Vamana Benchmark Script
Focuses on core functionality: Load Data -> Build Graph -> Ground Truth -> Search -> Report
Includes both In-Memory and Disk-Based (SSD) search tests.
"""

import numpy as np
import pandas as pd
import time
import sys
import os
import argparse
from tqdm import tqdm
import psutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydiskann.vamana_graph import build_vamana, greedy_search, beam_search_from_disk
from pydiskann.io.diskann_persist import DiskANNPersist, MMapNodeReader

def monitor_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def load_vectors(filepath, max_points=None):
    print(f"📂 Loading {filepath}...")
    df = pd.read_parquet(filepath)
    
    if max_points and len(df) > max_points:
        df = df.sample(n=max_points, random_state=42).reset_index(drop=True)
    
    # Find vector column
    vector_col = None
    for col in df.columns:
        if col in ['vector', 'emb', 'embedding']:
            vector_col = col
            break
    
    if not vector_col:
        # Fallback: check for object column with lists
        for col in df.columns:
            if df[col].dtype == 'object' and isinstance(df[col].iloc[0], (list, np.ndarray)):
                vector_col = col
                break
    
    if not vector_col:
        raise ValueError(f"No vector column found in {filepath}")
        
    print(f"   Using column: {vector_col}")
    
    # Convert to numpy float32
    if isinstance(df[vector_col].iloc[0], (list, np.ndarray)):
        vectors = np.stack(df[vector_col].tolist()).astype(np.float32)
    else:
        vectors = df[vector_col].values.astype(np.float32)
        
    print(f"   Shape: {vectors.shape}, Memory: {monitor_memory():.1f} MB")
    return vectors

def compute_ground_truth(train_vecs, test_vecs, k=10):
    print(f"🔍 Computing Ground Truth (Brute Force) for {len(test_vecs)} queries...")
    ground_truth = []
    start = time.time()
    
    for query in tqdm(test_vecs):
        dists = np.linalg.norm(train_vecs - query, axis=1)
        nearest = np.argsort(dists)[:k]
        ground_truth.append(nearest)
        
    print(f"   Ground truth computed in {time.time() - start:.2f}s")
    return np.array(ground_truth)

def run_benchmark(args):
    # 1. Load Data
    train_vecs = load_vectors(args.train_file, args.max_train_points)
    test_vecs = load_vectors(args.test_file, args.max_test_points)
    
    # 2. Build Graph
    print(f"\n🏗️  Building Vamana Graph (R={args.R}, L={args.L}, alpha={args.alpha})...")
    start_build = time.time()
    graph = build_vamana(
        train_vecs, 
        R=args.R, 
        L=args.L, 
        alpha=args.alpha,
        show_progress=True
    )
    build_time = time.time() - start_build
    print(f"✅ Build Complete in {build_time:.2f}s")
    
    # Graph stats
    avg_degree = sum(len(n.neighbors) for n in graph.nodes.values()) / len(graph.nodes)
    print(f"   Avg Degree: {avg_degree:.2f}")
    
    # 3. Ground Truth
    k = args.k
    ground_truth = compute_ground_truth(train_vecs, test_vecs, k)
    
    # 4. In-Memory Search & Evaluate
    search_Ls = [args.search_L] if args.search_L else [50, 100, 200]
    
    print(f"\n🚀 Testing In-Memory Search Performance (k={k})...")
    print(f"{'L':<5} {'Recall':<10} {'Avg Time (ms)':<15} {'QPS':<10}")
    print("-" * 45)
    
    start_node_idx = getattr(graph, 'medoid_idx', 0)
    
    for L in search_Ls:
        latencies = []
        recalls = []
        
        for i, query in enumerate(test_vecs):
            t0 = time.perf_counter()
            res = greedy_search(graph, start_node_idx, query, L)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000) # ms
            
            # Recall
            pred_set = set(res[:k])
            true_set = set(ground_truth[i][:k])
            recall = len(pred_set & true_set) / k
            recalls.append(recall)
            
        avg_recall = np.mean(recalls)
        avg_time = np.mean(latencies)
        qps = 1000 / avg_time
        
        print(f"{L:<5} {avg_recall:<10.4f} {avg_time:<15.2f} {qps:<10.0f}")

    # 5. Disk-Based Search & Evaluate
    print(f"\n💾 Testing Disk-Based Search Performance (SSD Mode)...")
    index_path = "vamana_index.bin"
    persist = DiskANNPersist(dim=train_vecs.shape[1], R=args.R)
    print(f"   Saving index to {index_path}...")
    persist.save_index(index_path, graph)
    
    print(f"   Opening MMapNodeReader...")
    reader = MMapNodeReader(index_path, dim=train_vecs.shape[1], R=args.R)
    
    print(f"{'Beam':<5} {'Recall':<10} {'Avg Time (ms)':<15} {'QPS':<10}")
    print("-" * 45)
    
    beam_widths = [24, 32, 48, 64]
    
    for bw in beam_widths:
        latencies = []
        recalls = []
        
        for i, query in enumerate(test_vecs):
            t0 = time.perf_counter()
            # Use beam_search_from_disk
            res = beam_search_from_disk(reader, query, start_node_idx, beam_width=bw, k=k)
            # res is list of (neg_dist, idx), we need just idxs
            res_idxs = [idx for _, idx in res]
            
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000) # ms
            
            # Recall
            pred_set = set(res_idxs[:k])
            true_set = set(ground_truth[i][:k])
            recall = len(pred_set & true_set) / k
            recalls.append(recall)
            
        avg_recall = np.mean(recalls)
        avg_time = np.mean(latencies)
        qps = 1000 / avg_time
        
        print(f"{bw:<5} {avg_recall:<10.4f} {avg_time:<15.2f} {qps:<10.0f}")
    
    reader.close()
    # Cleanup
    if os.path.exists(index_path):
        os.remove(index_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default="dataset/sift_small_500k/train_fixed.parquet")
    parser.add_argument("--test-file", default="dataset/sift_small_500k/test_fixed.parquet")
    parser.add_argument("--max-train-points", type=int, default=None)
    parser.add_argument("--max-test-points", type=int, default=1000)
    parser.add_argument("--R", type=int, default=32)
    parser.add_argument("--L", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=1.2)
    parser.add_argument("--search-L", type=int, default=None, help="Specific search L to test")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--no-cache", action="store_true", help="Ignored in simplified version")
    
    args = parser.parse_args()
    run_benchmark(args)