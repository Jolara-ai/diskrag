import logging
from pathlib import Path
import numpy as np
import time
import heapq
import threading
from typing import List, Dict, Any, Optional, Tuple
from pydiskann.io.diskann_persist import MMapNodeReader, DiskANNPersist
from preprocessing.collection import CollectionManager
from preprocessing.config import CollectionInfo, validate_vector_dimension
import json

logger = logging.getLogger(__name__)

class SearchEngineCorrect:
    """修正後的搜尋引擎，實現正確的PQ加速"""
    
    def __init__(self, collection_name: str, use_thread_safe_stats: bool = True):
        self.collection_name = collection_name
        self.manager = CollectionManager()
        self.info = self.manager.get_collection_info(collection_name)
        if not self.info:
            raise ValueError(f"找不到集合: {collection_name}")
        
        index_dir = self.manager.get_index_dir(collection_name)
        index_path = index_dir / "index.dat"
        meta_path = index_dir / "meta.json"
        
        if not index_path.exists() or not meta_path.exists():
            raise ValueError(f"集合 {collection_name} 的索引檔案不完整")
        
        persist = DiskANNPersist(dim=self.info.dimension, R=32)
        self.meta = persist.load_meta(str(meta_path))
        
        # 檢查是否使用 PQ
        self.use_pq = self.meta.get("use_pq", True)
        
        # 初始化 PQ 相关属性
        self.pq_model = None
        self.pq_codes = None
        self.n_subvectors = 0
        self.sub_dim = 0
        self.num_centroids = 0
        
        if self.use_pq:
            pq_path = index_dir / "pq_model.pkl"
            pq_codes_path = index_dir / "pq_codes.bin"
            
            if not all(p.exists() for p in [pq_path, pq_codes_path]):
                logger.warning(f"⚠️  PQ 文件不完整，切換到暴力搜索模式")
                self.use_pq = False
            else:
                try:
                    self.pq_model = persist.load_pq_codebook(str(pq_path))
                    self.pq_codes = persist.load_pq_codes(
                        str(pq_codes_path), 
                        self.meta["N"], 
                        self.meta["n_subvectors"]
                    )
                    self.n_subvectors = self.pq_model.n_subvectors
                    self.sub_dim = self.info.dimension // self.n_subvectors
                    # 兼容新旧版本的PQ模型
                    if hasattr(self.pq_model, 'kmeans_list') and self.pq_model.kmeans_list:
                        self.num_centroids = self.pq_model.kmeans_list[0].n_clusters
                    elif hasattr(self.pq_model, 'n_centroids'):
                        self.num_centroids = self.pq_model.n_centroids
                    else:
                        # 默认值
                        self.num_centroids = 256
                except Exception as e:
                    logger.warning(f"⚠️  PQ 模型載入失敗: {e}，切換到暴力搜索模式")
                    self.use_pq = False
        
        self.reader = MMapNodeReader(
            str(index_path),
            dim=self.info.dimension,
            R=self.meta.get("R", 32)
        )
        self.medoid_idx = self.meta.get("medoid_idx", 0)
        
        if not validate_vector_dimension(self.info.dimension):
            raise ValueError(
                f"不支援的向量維度: {self.info.dimension}。"
                f"請使用支援的維度重新建立索引"
            )
        
        # 統計信息 - 添加線程鎖保護
        self.search_stats = {
            'total_searches': 0,
            'total_exact_computations': 0,
            'total_pq_computations': 0,
            'total_search_time': 0.0
        }
        
        # 可選的線程安全統計
        self.use_thread_safe_stats = use_thread_safe_stats
        if use_thread_safe_stats:
            self._stats_lock = threading.Lock()  # 新增線程鎖
        else:
            self._stats_lock = None
        
        logger.info(
            f"已載入集合 {collection_name} 的索引 "
            f"(N={self.meta['N']}, dim={self.info.dimension}, "
            f"使用 PQ: {self.use_pq}, "
            f"線程安全統計: {use_thread_safe_stats})"
        )
        
        if self.use_pq:
            logger.info(f"  PQ 配置: {self.n_subvectors}x{self.num_centroids}")
            # --- ⭐️ 新增診斷步驟 ⭐️ ---
            diagnostic_result = self._run_diagnostic_check()
            if not diagnostic_result:
                logger.warning("⚠️  PQ 診斷檢查失敗，但繼續初始化。建議檢查 PQ 模型。")
        else:
            logger.info("  使用暴力搜索模式")
    
    def _update_stats(self, key: str, value: float = 1):
        """線程安全的統計更新方法"""
        if self.use_thread_safe_stats and self._stats_lock:
            with self._stats_lock:
                self.search_stats[key] += value
        else:
            self.search_stats[key] += value
    
    def _get_stats(self, key: str) -> int:
        """線程安全的統計讀取方法"""
        if self.use_thread_safe_stats and self._stats_lock:
            with self._stats_lock:
                return self.search_stats[key]
        else:
            return self.search_stats[key]
    
    def _get_all_stats(self) -> Dict[str, Any]:
        """線程安全的統計讀取方法（返回所有統計）"""
        if self.use_thread_safe_stats and self._stats_lock:
            with self._stats_lock:
                return self.search_stats.copy()
        else:
            return self.search_stats.copy()
    
    def _run_diagnostic_check(self):
        """運行更詳細的診斷檢查"""
        logger.info("🕵️  運行診斷自檢...")
        try:
            # 1. 基本統計檢查
            num_check = min(10, self.meta['N'])
            check_indices = np.random.choice(self.meta['N'], num_check, replace=False)
            original_vectors = np.array([self.reader.get_node(i)[0] for i in check_indices])
            
            logger.info(f"📊 原始向量統計:")
            logger.info(f"  - 數據類型: {original_vectors.dtype}")
            logger.info(f"  - 形狀: {original_vectors.shape}")
            logger.info(f"  - 範圍: [{original_vectors.min():.4f}, {original_vectors.max():.4f}]")
            logger.info(f"  - 均值: {original_vectors.mean():.4f}")
            logger.info(f"  - 標準差: {original_vectors.std():.4f}")
            
            # 2. 根據搜索模式進行不同的檢查
            if self.use_pq:
                logger.info("🔍 進行 PQ 模式診斷檢查...")
                # 檢查 PQ 模型完整性
                if not hasattr(self, 'pq_model') or not self.pq_model:
                    logger.error("❌ PQ 模型不存在！")
                    return False
                
                if not hasattr(self.pq_model, 'kmeans_list') or not self.pq_model.kmeans_list:
                    logger.error("❌ PQ 模型缺少 kmeans_list！這是導致 recall=0 的主要原因！")
                    return False
                
                if len(self.pq_model.kmeans_list) != self.n_subvectors:
                    logger.error(f"❌ PQ 子向量數量不匹配: 預期 {self.n_subvectors}, 實際 {len(self.pq_model.kmeans_list)}")
                    return False
                
                # 3. 檢查質心是否為零
                for i, kmeans in enumerate(self.pq_model.kmeans_list):
                    centroids = kmeans.cluster_centers_
                    if np.allclose(centroids, 0):
                        logger.error(f"❌ 子向量 {i} 的質心全為零！")
                        return False
                
                logger.info("✅ PQ 模型結構檢查通過")
                
                # 4. 距離計算一致性檢查
                query_vector = original_vectors[0]  # 使用第一個向量作為查詢
                
                # 構建 PQ 查找表
                try:
                    pq_lut = self._build_pq_lut_fixed(query_vector)
                    logger.info(f"✅ PQ 查找表構建成功: 形狀 {pq_lut.shape}")
                except Exception as e:
                    logger.error(f"❌ PQ 查找表構建失敗: {e}")
                    return False
                
                # 比較幾個點的精確距離和 PQ 距離
                logger.info("🔍 距離計算一致性檢查:")
                distance_correlations = []
                
                for i in range(min(5, num_check)):
                    node_id = check_indices[i]
                    
                    # 精確距離
                    exact_dist = self._compute_exact_distance(query_vector, node_id)
                    
                    # PQ 距離
                    if hasattr(self, 'pq_codes') and node_id < len(self.pq_codes):
                        pq_code = self.pq_codes[node_id]
                        pq_dist = self._get_pq_distance(pq_lut, pq_code)
                        ratio = pq_dist / exact_dist if exact_dist > 0 else float('inf')
                        distance_correlations.append((exact_dist, pq_dist))
                        
                        logger.info(f"  Node {node_id}: Exact={exact_dist:.6f}, PQ={pq_dist:.6f}, Ratio={ratio:.3f}")
                        
                        # 檢查是否合理
                        if ratio < 0.1 or ratio > 10:
                            logger.warning(f"⚠️  Node {node_id} 的距離比例異常: {ratio:.3f}")
                    else:
                        logger.warning(f"⚠️  Node {node_id} 沒有對應的 PQ 編碼")
                
                # 計算相關性
                if distance_correlations:
                    exact_dists, pq_dists = zip(*distance_correlations)
                    correlation = np.corrcoef(exact_dists, pq_dists)[0, 1]
                    logger.info(f"📊 精確距離與 PQ 距離的相關性: {correlation:.4f}")
                    
                    if correlation < 0.5:
                        logger.error(f"❌ 距離相關性過低 ({correlation:.4f})，這是導致 recall=0 的原因！")
                        return False
                
                logger.info("✅ 距離計算一致性檢查通過")
            else:
                logger.info("🔍 進行暴力搜索模式診斷檢查...")
                # 暴力搜索模式只需要檢查基本功能
                query_vector = original_vectors[0]
                
                # 測試精確距離計算
                test_distances = []
                for i in range(min(5, num_check)):
                    node_id = check_indices[i]
                    exact_dist = self._compute_exact_distance(query_vector, node_id)
                    test_distances.append(exact_dist)
                    logger.info(f"  Node {node_id}: Distance={exact_dist:.6f}")
                
                if not test_distances:
                    logger.error("❌ 無法計算任何距離！")
                    return False
                
                logger.info("✅ 暴力搜索模式診斷檢查通過")
            
            return True
            
        except Exception as e:
            logger.error(f"診斷過程中發生錯誤: {e}", exc_info=True)
            return False
    
    def __del__(self):
        if hasattr(self, 'reader'):
            self.reader.close()
    
    def _build_pq_lut(self, query_vector: np.ndarray) -> np.ndarray:
        """構建PQ查找表 - 使用DiskANNPQ的ADC方法"""
        # 检查模型类型并调用相应方法
        if hasattr(self.pq_model, 'compute_distance_table'):
            return self.pq_model.compute_distance_table(query_vector)
        else:
            # 兼容旧版本，手动构建距离表
            lut = np.empty((self.n_subvectors, self.num_centroids), dtype=np.float32)
            for i in range(self.n_subvectors):
                start_idx = i * self.sub_dim
                end_idx = (i + 1) * self.sub_dim
                sub_query = query_vector[start_idx:end_idx]
                # 兼容新旧版本的PQ模型
                if hasattr(self.pq_model, 'kmeans_list') and i < len(self.pq_model.kmeans_list):
                    centroids = self.pq_model.kmeans_list[i].cluster_centers_
                else:
                    # 如果kmeans_list不存在，使用默认值
                    centroids = np.zeros((self.num_centroids, self.sub_dim))
                diff = centroids - sub_query[np.newaxis, :]
                lut[i, :] = np.sum(diff * diff, axis=1)
            return lut

    def _build_pq_lut_fixed(self, query_vector: np.ndarray) -> np.ndarray:
        """修復的 PQ 查找表構建方法"""
        # 首先檢查 PQ 模型是否有內建方法
        if hasattr(self.pq_model, 'compute_distance_table'):
            try:
                return self.pq_model.compute_distance_table(query_vector)
            except Exception as e:
                logger.warning(f"使用內建 distance_table 方法失敗: {e}，改用手動構建")
        
        # 手動構建 - 但要確保質心正確
        if not hasattr(self.pq_model, 'kmeans_list') or not self.pq_model.kmeans_list:
            raise ValueError("PQ 模型缺少 kmeans_list，無法進行距離計算")
        
        if len(self.pq_model.kmeans_list) != self.n_subvectors:
            raise ValueError(f"PQ 子向量數量不匹配: 預期 {self.n_subvectors}, 實際 {len(self.pq_model.kmeans_list)}")
        
        lut = np.empty((self.n_subvectors, self.num_centroids), dtype=np.float32)
        
        for i in range(self.n_subvectors):
            start_idx = i * self.sub_dim
            end_idx = (i + 1) * self.sub_dim
            sub_query = query_vector[start_idx:end_idx]
            
            # 確保質心存在且不為零
            kmeans = self.pq_model.kmeans_list[i]
            centroids = kmeans.cluster_centers_
            
            if centroids.shape[0] != self.num_centroids:
                raise ValueError(f"子向量 {i} 的質心數量不匹配: 預期 {self.num_centroids}, 實際 {centroids.shape[0]}")
            
            if np.allclose(centroids, 0):
                raise ValueError(f"子向量 {i} 的質心全為零")
            
            # 計算平方距離
            diff = centroids - sub_query[np.newaxis, :]
            lut[i, :] = np.sum(diff * diff, axis=1)
        
        return lut

    def _debug_search_step_by_step(self, query_vector: np.ndarray, k: int = 5) -> Dict:
        """逐步調試搜索過程"""
        logger.info("🔍 開始逐步調試搜索過程...")
        
        # 1. 檢查 medoid
        logger.info(f"🎯 Medoid 索引: {self.medoid_idx}")
        medoid_exact_dist = self._compute_exact_distance(query_vector, self.medoid_idx)
        logger.info(f"🎯 Medoid 精確距離: {medoid_exact_dist:.6f}")
        
        # 2. 構建 PQ 查找表
        try:
            pq_lut = self._build_pq_lut_fixed(query_vector)
            logger.info(f"✅ PQ 查找表構建成功")
        except Exception as e:
            logger.error(f"❌ PQ 查找表構建失敗: {e}")
            return {"error": str(e)}
        
        # 3. 檢查 medoid 的鄰居
        _, medoid_neighbors = self.reader.get_node(self.medoid_idx)
        logger.info(f"🔗 Medoid 有 {len([n for n in medoid_neighbors if n >= 0])} 個有效鄰居")
        
        # 4. 檢查前幾個鄰居的距離
        valid_neighbors = [n for n in medoid_neighbors if n >= 0 and n < len(self.pq_codes)][:5]
        
        neighbor_info = []
        for neighbor_id in valid_neighbors:
            exact_dist = self._compute_exact_distance(query_vector, neighbor_id)
            pq_code = self.pq_codes[neighbor_id]
            pq_dist = self._get_pq_distance(pq_lut, pq_code)
            
            neighbor_info.append({
                'id': neighbor_id,
                'exact_dist': exact_dist,
                'pq_dist': pq_dist,
                'ratio': pq_dist / exact_dist if exact_dist > 0 else float('inf')
            })
            
            logger.info(f"  鄰居 {neighbor_id}: Exact={exact_dist:.6f}, PQ={pq_dist:.6f}, Ratio={pq_dist/exact_dist:.3f}")
        
        return {
            "medoid_idx": self.medoid_idx,
            "medoid_exact_dist": medoid_exact_dist,
            "neighbor_info": neighbor_info
        }
    
    def _get_pq_distance(self, lut: np.ndarray, pq_code: np.ndarray) -> float:
        """計算PQ距離 - 使用DiskANNPQ的ADC方法"""
        # 检查模型类型并调用相应方法
        if hasattr(self.pq_model, 'asymmetric_distance'):
            return self.pq_model.asymmetric_distance(pq_code.reshape(1, -1), lut)[0]
        else:
            # 兼容旧版本，手动计算距离
            return np.sum(lut[np.arange(self.n_subvectors), pq_code])
    
    def _compute_exact_distance(self, query_vector: np.ndarray, node_id: int) -> float:
        """計算精確的L2距離"""
        self._update_stats('total_exact_computations')
        full_vector, _ = self.reader.get_node(node_id)
        diff = full_vector - query_vector
        return np.sum(diff * diff)
    
    def _should_compute_exact_distance(self, pq_distance: float, current_candidates: List, 
                                     L: int, current_best_distance: float) -> bool:
        """決定是否需要計算精確距離的策略"""
        
        # 策略1: 如果候選列表未滿，總是計算精確距離
        if len(current_candidates) < L:
            return True
        
        # 策略2: 如果PQ距離明顯優於當前最差候選，計算精確距離
        if pq_distance < current_best_distance * 0.8:  # 80%閾值
            return True
        
        # 策略3: 對於邊界情況，以一定概率計算精確距離（避免誤判）
        if pq_distance < current_best_distance * 1.2:  # 120%閾值內
            return np.random.random() < 0.2  # 20%概率
        
        return False
    def _pq_accelerated_graph_search(self, query_vector: np.ndarray, k: int = 10, 
                                   L: int = 100, beam_width: int = None) -> Tuple[List[Tuple[float, int]], Dict]:
        """
        使用PQ加速的圖搜索 - 正確的DiskANN-PQ實現
        
        關鍵：仍然沿著圖的邊進行搜索，但使用PQ來減少精確距離計算
        """
        start_time = time.time()
        self._update_stats('total_searches')
        
        # 重置計數器 - 需要線程安全地讀取
        initial_exact_count = self._get_stats('total_exact_computations')
        initial_pq_count = self._get_stats('total_pq_computations')
        
        # 構建PQ查找表
        pq_lut = self._build_pq_lut_fixed(query_vector)
        
        # 初始化搜索
        visited = {self.medoid_idx}
        
        # 候選隊列：(distance, node_id)，小頂堆
        candidates = []
        # 結果隊列：(-distance, node_id)，大頂堆用於維護top-L
        results = []
        
        # 從medoid開始
        exact_dist = self._compute_exact_distance(query_vector, self.medoid_idx)
        heapq.heappush(candidates, (exact_dist, self.medoid_idx))
        heapq.heappush(results, (-exact_dist, self.medoid_idx))
        
        search_steps = 0
        max_search_steps = min(L * 10, self.meta["N"])  # 避免無限循環
        
        while candidates and search_steps < max_search_steps:
            search_steps += 1
            
            # 取出當前最佳候選
            current_dist, current_node = heapq.heappop(candidates)
            
            # 剪枝：如果當前距離已經比結果集中最差的還差，可以停止
            if len(results) >= L and current_dist > -results[0][0]:
                break
            
            # 獲取當前節點的鄰居
            _, neighbors = self.reader.get_node(current_node)
            
            for neighbor_id in neighbors:
                if neighbor_id < 0 or neighbor_id in visited:
                    continue
                
                visited.add(neighbor_id)
                
                # 關鍵步驟：首先使用PQ估算距離
                if neighbor_id < len(self.pq_codes):
                    pq_code = self.pq_codes[neighbor_id]
                    pq_distance = self._get_pq_distance(pq_lut, pq_code)
                    self._update_stats('total_pq_computations')
                else:
                    # 如果沒有PQ編碼，直接計算精確距離
                    pq_distance = float('inf')
                
                # 決定是否計算精確距離
                current_best_distance = -results[0][0] if results else float('inf')
                
                if (neighbor_id >= len(self.pq_codes) or 
                    self._should_compute_exact_distance(pq_distance, results, L, current_best_distance)):
                    # 計算精確距離
                    exact_distance = self._compute_exact_distance(query_vector, neighbor_id)
                    
                    # 更新候選和結果隊列
                    if len(results) < L or exact_distance < -results[0][0]:
                        heapq.heappush(candidates, (exact_distance, neighbor_id))
                        heapq.heappush(results, (-exact_distance, neighbor_id))
                        
                        # 維護結果隊列大小
                        if len(results) > L:
                            heapq.heappop(results)
            
            # 維護候選隊列大小（beam search）
            if beam_width and len(candidates) > beam_width:
                candidates = heapq.nsmallest(beam_width, candidates)
                heapq.heapify(candidates)
        
        # 提取最終結果
        final_results = []
        for neg_dist, node_id in results:
            final_results.append((-neg_dist, node_id))
        
        # 按距離排序並取前k個
        final_results.sort(key=lambda x: x[0])
        top_k_results = final_results[:k]
        
        search_time = time.time() - start_time
        self._update_stats('total_search_time', search_time)
        
        # 計算統計信息 - 需要線程安全地讀取
        exact_computations_this_search = self._get_stats('total_exact_computations') - initial_exact_count
        pq_computations_this_search = self._get_stats('total_pq_computations') - initial_pq_count
        
        search_stats = {
            'search_time': search_time,
            'nodes_visited': len(visited),
            'exact_distance_computations': exact_computations_this_search,
            'pq_distance_computations': pq_computations_this_search,
            'computation_reduction_rate': 1 - (exact_computations_this_search / max(1, pq_computations_this_search)),
            'search_steps': search_steps
        }
        
        return top_k_results, search_stats
    
    def _exact_graph_search(self, query_vector: np.ndarray, k: int = 10, L: int = 100) -> Tuple[List[Tuple[float, int]], Dict]:
        """基準的精確圖搜索（每個鄰居都計算精確距離）"""
        from pydiskann.vamana_graph import beam_search_from_disk
        
        start_time = time.time()
        results = beam_search_from_disk(
            self.reader,
            query_vector,
            start_id=self.medoid_idx,
            beam_width=8,
            k=k
        )
        search_time = time.time() - start_time
        
        search_stats = {
            'search_time': search_time,
            'exact_distance_computations': len(results) * 2,  # 估算
            'search_type': 'exact_beam_search'
        }
        
        return results, search_stats
    
    def search(self, query: str, k: int = 5, beam_width: int = 8,
               embedding_fn: Optional[callable] = None, L_search: int = None,
               use_pq_search: bool = True, use_simple_pq: bool = False) -> Dict[str, Any]:
        """
        搜索接口 - 支持暴力搜索模式
        """
        if embedding_fn is None:
            raise ValueError("必須提供 embedding_fn 來產生查詢向量")
        
        if L_search is None:
            L_search = max(k * 2, 20)
        
        total_start_time = time.time()
        embedding_start_time = time.time()
        query_vector = embedding_fn(query)
        embedding_time = time.time() - embedding_start_time
        
        if query_vector.shape[0] != self.info.dimension:
            raise ValueError(
                f"查詢向量維度不匹配: 預期 {self.info.dimension}，"
                f"實際 {query_vector.shape[0]}"
            )
        
        try:
            if use_simple_pq:
                # 不再支援錯誤的simple PQ實現
                logger.warning("simple_pq 選項已被移除，使用正確的PQ加速圖搜索")
                use_pq_search = True
            
            # 檢查是否可以使用 PQ 搜索
            if use_pq_search and not self.use_pq:
                logger.info("🔄 檢測到暴力搜索模式，自動切換到精確搜索")
                use_pq_search = False
            
            if use_pq_search and self.use_pq:
                # 使用正確的PQ加速圖搜索
                top_k_results, search_stats = self._pq_accelerated_graph_search(
                    query_vector, k, L_search, beam_width
                )
            else:
                # 使用精確圖搜索
                top_k_results, search_stats = self._exact_graph_search(
                    query_vector, k, L_search
                )
            
            # 獲取文本結果
            search_results = []
            for dist, idx in top_k_results:
                text_data = self.manager.get_text_by_index(self.collection_name, idx)
                if text_data:
                    text, metadata = text_data
                    if not isinstance(metadata, dict):
                        metadata = {"id": idx, "text": text}
                    search_results.append({
                        "text": text,
                        "distance": float(dist),
                        "metadata": metadata
                    })
            
            total_time = time.time() - total_start_time
            
            # 組織返回結果
            timing = {
                'embedding_time': embedding_time,
                'search_time': search_stats.get('search_time', 0),
                'total_time': total_time
            }
            
            # 添加搜索統計信息
            stats = {
                'search_type': 'pq_accelerated' if (use_pq_search and self.use_pq) else 'exact',
                'nodes_visited': search_stats.get('nodes_visited', 0),
                'k': k,
                'L_search': L_search
            }
            
            return {
                "results": search_results,
                "timing": timing,
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"搜索時發生錯誤: {e}")
            raise

    def search_with_debug(self, query: str, k: int = 5, beam_width: int = 8,
                         embedding_fn: Optional[callable] = None, L_search: int = None,
                         use_pq_search: bool = True, debug_mode: bool = False) -> Dict[str, Any]:
        """帶調試功能的搜索方法"""
        
        if embedding_fn is None:
            raise ValueError("必須提供 embedding_fn 來產生查詢向量")
        
        if L_search is None:
            L_search = max(k * 2, 20)
        
        query_vector = embedding_fn(query)
        
        if debug_mode:
            # 運行詳細診斷
            diagnostic_result = self._run_diagnostic_check()
            if not diagnostic_result:
                logger.error("❌ 診斷檢查失敗，搜索可能不會正常工作")
            
            # 逐步調試
            debug_info = self._debug_search_step_by_step(query_vector, k)
            
            # 比較精確搜索和 PQ 搜索
            logger.info("🔄 比較精確搜索和 PQ 搜索結果...")
            
            exact_results, _ = self._exact_graph_search(query_vector, k, L_search)
            logger.info(f"精確搜索結果: {[idx for _, idx in exact_results[:k]]}")
            
            if use_pq_search:
                try:
                    pq_results, _ = self._pq_accelerated_graph_search(query_vector, k, L_search, beam_width)
                    logger.info(f"PQ 搜索結果: {[idx for _, idx in pq_results[:k]]}")
                except Exception as e:
                    logger.error(f"PQ 搜索失敗: {e}")
                    pq_results = []
            
            return {
                "debug_info": debug_info,
                "exact_results": exact_results,
                "pq_results": pq_results if use_pq_search else [],
                "diagnostic_passed": diagnostic_result
            }
        
        # 正常搜索流程
        return self.search(query, k, beam_width, embedding_fn, L_search, use_pq_search)

    def get_search_statistics(self) -> Dict[str, Any]:
        """獲取搜索統計信息"""
        all_stats = self._get_all_stats()
        if all_stats['total_searches'] == 0:
            return {"message": "尚未執行任何搜索"}
            
        avg_exact_per_search = all_stats['total_exact_computations'] / all_stats['total_searches']
        avg_pq_per_search = all_stats['total_pq_computations'] / all_stats['total_searches']
        avg_search_time = all_stats['total_search_time'] / all_stats['total_searches']
        
        return {
            'total_searches': all_stats['total_searches'],
            'avg_exact_computations_per_search': avg_exact_per_search,
            'avg_pq_computations_per_search': avg_pq_per_search,
            'avg_search_time': avg_search_time,
            'total_exact_computations': all_stats['total_exact_computations'],
            'total_pq_computations': all_stats['total_pq_computations'],
            'overall_computation_reduction_rate': 1 - (avg_exact_per_search / max(1, avg_pq_per_search))
        }
    
    def get_text_by_hash(self, text_hash: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        return self.manager.get_text_by_hash(self.collection_name, text_hash)
    
    @classmethod
    def list_collections(cls) -> List[CollectionInfo]:
        """列出所有可用的集合"""
        return CollectionManager().list_collections()
    
    def get_collection_info(self) -> CollectionInfo:
        """取得當前集合的資訊"""
        return self.info

    def faq_search(self, query: str, k: int = 5, beam_width: int = 8,
                   embedding_fn: Optional[callable] = None, L_search: int = None,
                   use_pq_search: bool = True) -> Dict[str, Any]:
        """
        FAQ 專用搜索 - 支持暴力搜索模式
        """
        if embedding_fn is None:
            raise ValueError("必須提供 embedding_fn 來產生查詢向量")
        
        if L_search is None:
            L_search = max(k * 2, 20)
        
        total_start_time = time.time()
        embedding_start_time = time.time()
        query_vector = embedding_fn(query)
        embedding_time = time.time() - embedding_start_time
        
        if query_vector.shape[0] != self.info.dimension:
            raise ValueError(
                f"查詢向量維度不匹配: 預期 {self.info.dimension}，"
                f"實際 {query_vector.shape[0]}"
            )
        
        try:
            # 檢查是否可以使用 PQ 搜索
            if use_pq_search and not self.use_pq:
                logger.info("🔄 檢測到暴力搜索模式，自動切換到精確搜索")
                use_pq_search = False
            
            # 獲取更多結果以便去重
            search_k = k * 3
            
            if use_pq_search and self.use_pq:
                top_k_results, search_stats = self._pq_accelerated_graph_search(
                    query_vector, search_k, L_search, beam_width
                )
            else:
                top_k_results, search_stats = self._exact_graph_search(
                    query_vector, search_k, L_search
                )
            
            # 處理結果去重
            final_results = []
            seen_qa_ids = set()
            
            for dist, idx in top_k_results:
                text_data = self.manager.get_text_by_index(self.collection_name, idx)
                if not text_data:
                    continue
                
                text, metadata = text_data
                
                # 解析元數據
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {"id": idx, "text": text}
                
                # 檢查是否為FAQ類型 - 支持嵌套metadata結構
                metadata_type = metadata.get("type")
                if not metadata_type:
                    # 檢查嵌套的metadata字段
                    nested_metadata = metadata.get("metadata")
                    if isinstance(nested_metadata, str):
                        try:
                            nested_metadata = json.loads(nested_metadata)
                            metadata_type = nested_metadata.get("type")
                        except json.JSONDecodeError:
                            pass
                    elif isinstance(nested_metadata, dict):
                        metadata_type = nested_metadata.get("type")
                
                if metadata_type != "faq":
                    continue
                
                # 獲取qa_id
                qa_id = metadata.get("qa_id")
                if not qa_id or qa_id in seen_qa_ids:
                    continue  # 跳過沒有qa_id或已經處理過的
                
                seen_qa_ids.add(qa_id)
                
                final_results.append({
                    "text": text,
                    "distance": float(dist),
                    "metadata": metadata
                })
                
                if len(final_results) >= k:
                    break
            
            total_time = time.time() - total_start_time
            
            # 組織返回結果
            timing = {
                'embedding_time': embedding_time,
                'search_time': search_stats.get('search_time', 0),
                'total_time': total_time
            }
            
            stats = {
                'search_type': 'pq_accelerated' if (use_pq_search and self.use_pq) else 'exact',
                'nodes_visited': search_stats.get('nodes_visited', 0),
                'k': k,
                'L_search': L_search,
                'total_results_before_dedup': len(top_k_results),
                'final_results_after_dedup': len(final_results)
            }
            
            return {
                "results": final_results,
                "timing": timing,
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"FAQ搜索時發生錯誤: {e}")
            raise


# 向後兼容的別名，但應該逐步遷移到新的實現
SearchEngine = SearchEngineCorrect

def performance_test_search_engine(collection_name: str, num_queries: int = 100):
    """
    性能測試：比較線程安全修改前後的性能差異
    
    Args:
        collection_name: 集合名稱
        num_queries: 測試查詢數量
    """
    import time
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # 創建搜索引擎實例
    engine = SearchEngineCorrect(collection_name)
    
    # 模擬查詢向量（使用隨機向量）
    def mock_embedding(query: str) -> np.ndarray:
        """模擬embedding函數"""
        return np.random.randn(engine.info.dimension).astype(np.float32)
    
    def single_search(query_id: int) -> float:
        """單次搜索測試"""
        start_time = time.time()
        try:
            result = engine.search(
                query=f"test_query_{query_id}",
                k=5,
                embedding_fn=mock_embedding,
                use_pq_search=True
            )
            search_time = time.time() - start_time
            return search_time
        except Exception as e:
            print(f"搜索 {query_id} 失敗: {e}")
            return -1
    
    print(f"🔍 開始性能測試：{num_queries} 個並發查詢")
    print(f"📊 集合信息：{collection_name}, 維度：{engine.info.dimension}")
    
    # 單線程測試（基準）
    print("\n📈 單線程測試（基準）...")
    single_thread_times = []
    start_time = time.time()
    
    for i in range(num_queries):
        search_time = single_search(i)
        if search_time > 0:
            single_thread_times.append(search_time)
    
    single_thread_total = time.time() - start_time
    single_thread_avg = np.mean(single_thread_times) if single_thread_times else 0
    
    print(f"   總時間：{single_thread_total:.3f}s")
    print(f"   平均搜索時間：{single_thread_avg*1000:.2f}ms")
    print(f"   成功查詢：{len(single_thread_times)}/{num_queries}")
    
    # 多線程測試
    print("\n🚀 多線程測試（線程安全）...")
    multi_thread_times = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        # 提交所有任務
        future_to_query = {
            executor.submit(single_search, i): i 
            for i in range(num_queries)
        }
        
        # 收集結果
        for future in as_completed(future_to_query):
            query_id = future_to_query[future]
            try:
                search_time = future.result()
                if search_time > 0:
                    multi_thread_times.append(search_time)
            except Exception as e:
                print(f"查詢 {query_id} 異常：{e}")
    
    multi_thread_total = time.time() - start_time
    multi_thread_avg = np.mean(multi_thread_times) if multi_thread_times else 0
    
    print(f"   總時間：{multi_thread_total:.3f}s")
    print(f"   平均搜索時間：{multi_thread_avg*1000:.2f}ms")
    print(f"   成功查詢：{len(multi_thread_times)}/{num_queries}")
    
    # 性能比較
    print("\n📊 性能比較結果：")
    if single_thread_avg > 0 and multi_thread_avg > 0:
        speedup = single_thread_total / multi_thread_total
        overhead = (multi_thread_avg - single_thread_avg) / single_thread_avg * 100
        
        print(f"   並發加速比：{speedup:.2f}x")
        print(f"   單次搜索開銷：{overhead:+.1f}%")
        
        if overhead < 5:
            print("   ✅ 性能消耗可接受（< 5%）")
        elif overhead < 10:
            print("   ⚠️  性能消耗中等（5-10%）")
        else:
            print("   ❌ 性能消耗較大（> 10%）")
    
    # 統計資料
    print("\n📈 統計資料：")
    stats = engine.get_search_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    return {
        'single_thread_avg': single_thread_avg,
        'multi_thread_avg': multi_thread_avg,
        'speedup': speedup if 'speedup' in locals() else 0,
        'overhead': overhead if 'overhead' in locals() else 0
    }