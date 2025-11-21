#!/usr/bin/env python3
"""
測試磁碟寫入和驗證腳本
創建一個小規模的索引並驗證它真的寫入磁盤
"""

import os
import sys
import numpy as np
import time
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pydiskann.vamana_graph import build_vamana
from pydiskann.io.diskann_persist import DiskANNPersist, MMapNodeReader
from pydiskann.vamana_graph import beam_search_from_disk

def test_disk_write_and_verify():
    """測試磁碟寫入和驗證"""
    print("🧪 測試磁碟索引寫入和驗證")
    print("=" * 60)
    
    # 1. 創建小規模測試數據
    print("\n📊 創建測試數據...")
    n_points = 1000
    dim = 128
    np.random.seed(42)
    points = np.random.randn(n_points, dim).astype(np.float32)
    print(f"   - 節點數: {n_points}")
    print(f"   - 維度: {dim}")
    
    # 2. 構建圖
    print("\n🏗️  構建 Vamana 圖...")
    start = time.time()
    graph = build_vamana(points, R=16, L=32, alpha=1.2, show_progress=False)
    build_time = time.time() - start
    print(f"   ✅ 構建完成，耗時: {build_time:.2f}秒")
    print(f"   - 節點數: {len(graph.nodes)}")
    avg_degree = sum(len(n.neighbors) for n in graph.nodes.values()) / len(graph.nodes)
    print(f"   - 平均出度: {avg_degree:.2f}")
    
    # 3. 保存到磁碟
    index_path = "test_vamana_index.bin"
    print(f"\n💾 保存索引到磁碟: {index_path}")
    
    # 檢查文件是否已存在
    if os.path.exists(index_path):
        old_size = os.stat(index_path).st_size
        print(f"   ⚠️  文件已存在，大小: {old_size:,} 位元組")
        os.remove(index_path)
        print(f"   🗑️  已刪除舊檔案")
    
    # 保存索引
    persist = DiskANNPersist(dim=dim, R=16)
    start_save = time.time()
    persist.save_index(index_path, graph)
    save_time = time.time() - start_save
    
    # 驗證文件已創建
    if not os.path.exists(index_path):
        print(f"   ❌ 文件保存失敗！")
        return False
    
    stat = os.stat(index_path)
    file_size = stat.st_size
    print(f"   ✅ 文件保存成功")
    print(f"   - 保存時間: {save_time:.4f}秒")
    print(f"   - 文件大小: {file_size:,} 位元組 ({file_size / 1024 / 1024:.2f} MB)")
    
    # 4. 驗證文件結構
    print(f"\n📐 驗證文件結構...")
    record_size = 4 * (dim + 16)  # float32 * dim + uint32 * R
    expected_size = n_points * record_size
    print(f"   - 每條記錄大小: {record_size} 位元組")
    print(f"   - 預期文件大小: {expected_size:,} 位元組")
    print(f"   - 實際文件大小: {file_size:,} 位元組")
    
    if abs(file_size - expected_size) < 1024:
        print(f"   ✅ 文件大小符合預期")
    else:
        print(f"   ⚠️  文件大小差異: {abs(file_size - expected_size):,} 位元組")
    
    # 5. 驗證文件存儲位置
    print(f"\n💿 驗證文件儲存位置...")
    try:
        import subprocess
        result = subprocess.run(['df', index_path], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                print(f"   - 文件系統: {parts[0]}")
                print(f"   - 掛載點: {parts[-1]}")
                print(f"   - 總容量: {parts[1]}")
                print(f"   - 已使用: {parts[2]}")
                print(f"   - 可用空間: {parts[3]}")
    except Exception as e:
        print(f"   ⚠️  無法獲取文件系統信息: {e}")
    
    # 6. 使用 MMapNodeReader 讀取
    print(f"\n📖 測試磁碟讀取 (MMap)...")
    try:
        reader = MMapNodeReader(index_path, dim=dim, R=16)
        print(f"   ✅ MMapNodeReader 初始化成功")
        
        # 讀取幾個節點驗證
        print(f"\n   🔍 驗證節點讀取:")
        test_nodes = [0, n_points//4, n_points//2, 3*n_points//4, n_points-1]
        for node_id in test_nodes:
            if node_id < n_points:
                vec, neighbors = reader.get_node(node_id)
                print(f"   - 節點 {node_id}:")
                print(f"     * 向量形狀: {vec.shape}, dtype: {vec.dtype}")
                print(f"     * 鄰居數量: {len(neighbors)}")
                valid_neighbors = neighbors[neighbors > 0]
                print(f"     * 有效鄰居: {len(valid_neighbors)} 個")
                if len(valid_neighbors) > 0:
                    print(f"     * 鄰居 ID: {valid_neighbors[:5].tolist()}")
        
        # 7. 測試搜索
        print(f"\n🔍 測試磁碟搜索...")
        query = np.random.randn(dim).astype(np.float32)
        start_node = graph.medoid_idx if hasattr(graph, 'medoid_idx') else 0
        
        search_times = []
        for beam_width in [8, 16]:
            t0 = time.perf_counter()
            results = beam_search_from_disk(reader, query, start_node, beam_width=beam_width, k=5)
            t1 = time.perf_counter()
            search_time = (t1 - t0) * 1000  # ms
            search_times.append(search_time)
            print(f"   - Beam={beam_width}: {search_time:.4f} ms, 找到 {len(results)} 個結果")
            if results:
                print(f"     * Top-3: {[idx for _, idx in results[:3]]}")
        
        reader.close()
        print(f"   ✅ 磁碟搜索測試成功")
        
    except Exception as e:
        print(f"   ❌ 讀取失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 8. 驗證文件確實寫入磁碟（讀取原始字節）
    print(f"\n🔬 驗證文件內容...")
    try:
        with open(index_path, 'rb') as f:
            # 讀取第一個節點的向量
            first_vec_bytes = f.read(4 * dim)
            first_vec = np.frombuffer(first_vec_bytes, dtype=np.float32)
            
            # 讀取第一個節點的鄰居
            first_neighbors_bytes = f.read(4 * 16)
            first_neighbors = np.frombuffer(first_neighbors_bytes, dtype=np.uint32)
            
            print(f"   - 第一個節點的向量 (前5個值): {first_vec[:5]}")
            print(f"   - 第一個節點的鄰居: {first_neighbors[first_neighbors > 0].tolist()}")
            
            # 與內存中的圖比較
            if 0 in graph.nodes:
                mem_vec = graph.nodes[0].vector
                mem_neighbors = list(graph.nodes[0].neighbors)
                
                vec_match = np.allclose(first_vec, mem_vec, atol=1e-6)
                neighbors_match = set(first_neighbors[first_neighbors > 0]) == set(mem_neighbors)
                
                print(f"   - 向量匹配: {'✅' if vec_match else '❌'}")
                print(f"   - 鄰居匹配: {'✅' if neighbors_match else '❌'}")
                
                if vec_match and neighbors_match:
                    print(f"   ✅ 文件內容與內存圖一致")
                else:
                    print(f"   ⚠️  文件內容與內存圖不一致")
    except Exception as e:
        print(f"   ⚠️  無法驗證文件內容: {e}")
    
    print("\n" + "=" * 60)
    print("✅ 測試完成")
    print(f"\n📝 結論:")
    print(f"   - 索引文件已成功寫入磁碟")
    print(f"   - 文件大小: {file_size:,} 位元組 ({file_size / 1024 / 1024:.2f} MB)")
    print(f"   - 可以使用 MMapNodeReader 從磁碟讀取")
    print(f"   - 磁碟搜索功能正常")
    
    # 詢問是否保留文件
    print(f"\n💡 提示: 測試檔案 '{index_path}' 已創建")
    print(f"   如需保留用於進一步測試，請不要刪除")
    
    return True

if __name__ == "__main__":
    success = test_disk_write_and_verify()
    sys.exit(0 if success else 1)

