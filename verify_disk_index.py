#!/usr/bin/env python3
"""
驗證磁碟索引文件的腳本
檢查索引文件是否真的存在於磁碟上，並驗證其結構
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pydiskann.io.diskann_persist import MMapNodeReader, DiskANNPersist
import json

def verify_index_file(index_path, meta_path=None):
    """驗證索引文件"""
    print(f"🔍 驗證索引文件: {index_path}")
    print("=" * 60)
    
    # 1. 檢查文件是否存在
    if not os.path.exists(index_path):
        print(f"❌ 文件不存在: {index_path}")
        return False
    
    # 2. 獲取文件信息
    stat = os.stat(index_path)
    file_size = stat.st_size
    print(f"✅ 文件存在")
    print(f"   - 大小: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    print(f"   - 修改時間: {stat.st_mtime}")
    
    # 3. 讀取元數據
    if meta_path and os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        print(f"\n📋 元數據資料:")
        for key, value in meta.items():
            print(f"   - {key}: {value}")
        
        dim = meta.get('dimension', meta.get('D', 128))
        R = meta.get('R', 32)
        N = meta.get('N', 0)
    else:
        print("\n⚠️  未找到 meta.json，使用預設值")
        dim = 128
        R = 32
        N = 0
    
    # 4. 計算預期文件大小
    record_size = 4 * (dim + R)  # float32 * dim + uint32 * R
    if N > 0:
        expected_size = N * record_size
        print(f"\n📐 文件結構驗證:")
        print(f"   - 向量維度 (D): {dim}")
        print(f"   - 最大出度 (R): {R}")
        print(f"   - 節點數量 (N): {N}")
        print(f"   - 每條記錄大小: {record_size} bytes")
        print(f"   - 預期文件大小: {expected_size:,} bytes ({expected_size / 1024 / 1024:.2f} MB)")
        print(f"   - 實際文件大小: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
        
        if abs(file_size - expected_size) < 1024:  # 允許 1KB 誤差
            print(f"   ✅ 文件大小符合預期")
        else:
            print(f"   ⚠️  文件大小與預期不符 (差異: {abs(file_size - expected_size):,} bytes)")
    
    # 5. 嘗試使用 MMapNodeReader 讀取
    print(f"\n💾 測試磁碟讀取 (MMap):")
    try:
        reader = MMapNodeReader(index_path, dim=dim, R=R)
        print(f"   ✅ MMapNodeReader 初始化成功")
        
        # 讀取第一個節點
        if N > 0:
            print(f"\n📖 讀取節點樣本:")
            for node_id in [0, min(100, N-1), N-1]:
                if node_id < N:
                    vec, neighbors = reader.get_node(node_id)
                    print(f"   - 節點 {node_id}:")
                    print(f"     * 向量形狀: {vec.shape}, 類型: {vec.dtype}")
                    print(f"     * 向量前5個值: {vec[:5]}")
                    print(f"     * 鄰居數量: {len(neighbors)}")
                    print(f"     * 鄰居前5個: {neighbors[:5].tolist()}")
                    print(f"     * 非零鄰居: {neighbors[neighbors > 0].tolist()}")
        
        # 測試隨機讀取
        print(f"\n🎲 測試隨機讀取性能:")
        import time
        test_nodes = [0, N//4, N//2, 3*N//4, N-1] if N > 0 else [0]
        test_nodes = [n for n in test_nodes if n < N]
        
        times = []
        for node_id in test_nodes:
            t0 = time.perf_counter()
            vec, neighbors = reader.get_node(node_id)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms
            print(f"   - 節點 {node_id}: {times[-1]:.4f} ms")
        
        if times:
            avg_time = np.mean(times)
            print(f"   - 平均讀取時間: {avg_time:.4f} ms")
        
        reader.close()
        print(f"   ✅ 磁碟讀取測試成功")
        
    except Exception as e:
        print(f"   ❌ 讀取失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. 驗證文件確實寫入磁盤（不是內存映射）
    print(f"\n💿 驗證文件存儲位置:")
    try:
        # 獲取文件所在的設備
        import subprocess
        result = subprocess.run(['df', index_path], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                print(f"   {lines[1]}")
        
        # 檢查文件是否真的在磁碟上（通過讀取一小部分）
        with open(index_path, 'rb') as f:
            f.seek(0)
            header = f.read(min(1024, file_size))
            print(f"   ✅ 文件可以正常讀取（前 {len(header)} bytes）")
            print(f"   - 前16 bytes (hex): {header[:16].hex()}")
            
    except Exception as e:
        print(f"   ⚠️  無法驗證存儲位置: {e}")
    
    print("\n" + "=" * 60)
    print("✅ 驗證完成")
    return True

if __name__ == "__main__":
    # 檢查預設索引
    default_index = Path("collections/default/index/index.dat")
    default_meta = Path("collections/default/index/meta.json")
    
    if default_index.exists():
        verify_index_file(str(default_index), str(default_meta) if default_meta.exists() else None)
    else:
        print("❌ 未找到預設索引文件")
        print("   請先運行 benchmark 或建立索引")
        
        # 提供使用指引
        print("\n使用方法:")
        print("  python verify_disk_index.py")
        print("  或指定索引路徑:")
        print("  python verify_disk_index.py <index_path> [meta_path]")

