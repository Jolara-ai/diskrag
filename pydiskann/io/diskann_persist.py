import numpy as np
import json
import pickle
import mmap
import logging
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger(__name__)

class DiskANNPersist:
    def __init__(self, dim=128, R=16, record_bytes=None):
        self.D = dim
        self.R = R
        self.record_size = 4 * (dim + R) if record_bytes is None else record_bytes

    def save_index(self, filepath, graph):
        with open(filepath, 'wb') as f:
            for idx in range(len(graph.nodes)):
                vec = graph.nodes[idx].vector.astype(np.float32)
                f.write(vec.tobytes())
                neighbors = list(graph.nodes[idx].neighbors)
                neighbors += [0] * (self.R - len(neighbors))
                f.write(np.array(neighbors[:self.R], dtype=np.uint32).tobytes())

    def save_meta(self, filepath, meta_dict):
        with open(filepath, 'w') as f:
            json.dump(meta_dict, f)

    def save_pq_codes(self, filepath, pq_codes):
        pq_codes.astype(np.uint8).tofile(filepath)

    def save_pq_codebook(self, filepath, pq_model):
        """改進的 PQ 模型保存方法 - 解決序列化問題"""
        logger.info(f"🔧 開始保存 PQ 模型到: {filepath}")
        
        # 檢查模型完整性
        if not hasattr(pq_model, 'is_fitted') or not pq_model.is_fitted:
            raise ValueError("PQ 模型未訓練完成，無法保存")
        
        if not hasattr(pq_model, 'kmeans_list') or not pq_model.kmeans_list:
            raise ValueError("PQ 模型缺少 kmeans_list，無法保存")
        
        # 創建包含所有必要信息的字典
        model_data = {
            'n_subvectors': pq_model.n_subvectors,
            'n_centroids': pq_model.n_centroids,
            'sub_dim': pq_model.sub_dim,
            'is_fitted': pq_model.is_fitted,
            'kmeans_list': pq_model.kmeans_list,
            'means_': getattr(pq_model, 'means_', None),
            'stds_': getattr(pq_model, 'stds_', None),
            'epsilon': getattr(pq_model, 'epsilon', 1e-8),
            'model_type': 'DiskANNPQ',
            'version': '2.0'
        }
        
        # 驗證關鍵數據完整性
        logger.info("🔍 驗證 PQ 模型數據完整性...")
        logger.info(f"  - n_subvectors: {model_data['n_subvectors']}")
        logger.info(f"  - n_centroids: {model_data['n_centroids']}")
        logger.info(f"  - sub_dim: {model_data['sub_dim']}")
        logger.info(f"  - is_fitted: {model_data['is_fitted']}")
        logger.info(f"  - kmeans_list 長度: {len(model_data['kmeans_list'])}")
        logger.info(f"  - means_ 存在: {model_data['means_'] is not None}")
        logger.info(f"  - stds_ 存在: {model_data['stds_'] is not None}")
        
        # 檢查每個 KMeans 模型
        for i, kmeans in enumerate(model_data['kmeans_list']):
            if not hasattr(kmeans, 'cluster_centers_'):
                raise ValueError(f"KMeans 模型 {i} 缺少 cluster_centers_")
            centers_shape = kmeans.cluster_centers_.shape
            expected_shape = (model_data['n_centroids'], model_data['sub_dim'])
            if centers_shape != expected_shape:
                raise ValueError(f"KMeans 模型 {i} 聚類中心形狀錯誤: {centers_shape}, 預期: {expected_shape}")
        
        logger.info("✅ PQ 模型數據完整性檢查通過")
        
        # 保存到文件
        try:
            # 先保存到臨時文件
            temp_filepath = str(filepath) + '.tmp'
            with open(temp_filepath, 'wb') as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # 驗證保存的文件可以正確加載
            with open(temp_filepath, 'rb') as f:
                test_data = pickle.load(f)
            
            # 基本驗證
            assert test_data['model_type'] == 'DiskANNPQ'
            assert test_data['n_subvectors'] == model_data['n_subvectors']
            assert len(test_data['kmeans_list']) == len(model_data['kmeans_list'])
            
            # 如果驗證通過，重命名為最終文件
            Path(temp_filepath).rename(filepath)
            logger.info(f"✅ PQ 模型已成功保存至: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ PQ 模型保存失敗: {e}")
            # 清理臨時文件
            temp_path = Path(str(filepath) + '.tmp')
            if temp_path.exists():
                temp_path.unlink()
            raise

    def load_pq_codebook(self, filepath):
        """改進的 PQ 模型加載方法 - 解決反序列化問題"""
        logger.info(f"🔧 開始加載 PQ 模型從: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # 檢查是否為新格式
            if isinstance(model_data, dict) and 'model_type' in model_data:
                logger.info("✅ 檢測到新格式 PQ 模型")
                return self._load_new_format_pq(model_data)
            else:
                logger.warning(f"⚠️  檢測到舊格式 PQ 模型，嘗試兼容加載...")
                return self._load_legacy_format_pq(model_data)
                
        except Exception as e:
            logger.error(f"❌ PQ 模型加載失敗: {e}")
            raise
    
    def _load_new_format_pq(self, model_data):
        """加載新格式的 PQ 模型"""
        from pydiskann.pq.fast_pq import DiskANNPQ
        
        # 驗證數據完整性
        required_keys = ['n_subvectors', 'n_centroids', 'sub_dim', 'is_fitted', 'kmeans_list']
        for key in required_keys:
            if key not in model_data:
                raise ValueError(f"PQ 模型數據缺少必要字段: {key}")
        
        logger.info("🔍 重建 PQ 模型...")
        logger.info(f"  - n_subvectors: {model_data['n_subvectors']}")
        logger.info(f"  - n_centroids: {model_data['n_centroids']}")
        logger.info(f"  - sub_dim: {model_data['sub_dim']}")
        logger.info(f"  - kmeans_list 長度: {len(model_data['kmeans_list'])}")
        
        # 重建 PQ 模型
        pq_model = DiskANNPQ(
            n_subvectors=model_data['n_subvectors'],
            n_centroids=model_data['n_centroids']
        )
        
        # 恢復所有屬性
        pq_model.sub_dim = model_data['sub_dim']
        pq_model.is_fitted = model_data['is_fitted']
        pq_model.kmeans_list = model_data['kmeans_list']
        pq_model.means_ = model_data.get('means_')
        pq_model.stds_ = model_data.get('stds_')
        pq_model.epsilon = model_data.get('epsilon', 1e-8)
        
        # 驗證加載的模型
        if not pq_model.kmeans_list:
            raise ValueError("加載的 PQ 模型缺少 kmeans_list")
        
        if len(pq_model.kmeans_list) != pq_model.n_subvectors:
            raise ValueError(f"KMeans 模型數量不匹配: {len(pq_model.kmeans_list)} != {pq_model.n_subvectors}")
        
        # 檢查標準化參數
        if pq_model.means_ is not None and pq_model.stds_ is not None:
            logger.info("✅ 包含標準化參數 (means_, stds_)")
            expected_dim = pq_model.n_subvectors * pq_model.sub_dim
            if len(pq_model.means_) != expected_dim:
                raise ValueError(f"means_ 維度錯誤: {len(pq_model.means_)} != {expected_dim}")
            if len(pq_model.stds_) != expected_dim:
                raise ValueError(f"stds_ 維度錯誤: {len(pq_model.stds_)} != {expected_dim}")
        else:
            logger.warning("⚠️  缺少標準化參數，可能影響搜索精度")
        
        # 檢查每個 KMeans 模型
        for i, kmeans in enumerate(pq_model.kmeans_list):
            if not hasattr(kmeans, 'cluster_centers_'):
                raise ValueError(f"KMeans 模型 {i} 缺少 cluster_centers_")
            centers_shape = kmeans.cluster_centers_.shape
            expected_shape = (pq_model.n_centroids, pq_model.sub_dim)
            if centers_shape != expected_shape:
                raise ValueError(f"KMeans 模型 {i} 聚類中心形狀錯誤: {centers_shape}, 預期: {expected_shape}")
        
        logger.info("✅ 新格式 PQ 模型加載成功")
        return pq_model
    
    def _load_legacy_format_pq(self, model_data):
        """加載舊格式的 PQ 模型（向後兼容）"""
        logger.warning("⚠️  正在加載舊格式 PQ 模型，建議重新訓練以獲得最佳性能")
        
        # 檢查舊格式模型的基本完整性
        if hasattr(model_data, 'is_fitted') and model_data.is_fitted:
            if hasattr(model_data, 'kmeans_list') and model_data.kmeans_list:
                logger.info("✅ 舊格式 PQ 模型基本完整性檢查通過")
                return model_data
            else:
                raise ValueError("舊格式 PQ 模型缺少 kmeans_list")
        else:
            raise ValueError("舊格式 PQ 模型未訓練或缺少 is_fitted 標記")

    def load_meta(self, filepath):
        with open(filepath, 'r') as f:
            return json.load(f)

    def load_pq_codes(self, filepath, num_nodes, n_subvectors):
        return np.fromfile(filepath, dtype=np.uint8).reshape((num_nodes, n_subvectors))


class MMapNodeReader:
    def __init__(self, filepath, dim=128, R=16, cache_size=1024):
        self.D = dim
        self.R = R
        self.record_size = 4 * (dim + R)
        self.file = open(filepath, 'rb')
        self.mmap_obj = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.cache = OrderedDict()
        self.cache_size = cache_size

    def get_node(self, node_id):
        if node_id in self.cache:
            self.cache.move_to_end(node_id)
            return self.cache[node_id]
        offset = node_id * self.record_size
        self.mmap_obj.seek(offset)
        vec = np.frombuffer(self.mmap_obj.read(4 * self.D), dtype=np.float32)
        neighbors = np.frombuffer(self.mmap_obj.read(4 * self.R), dtype=np.uint32)
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)
        self.cache[node_id] = (vec, neighbors)
        return vec, neighbors

    def close(self):
        self.mmap_obj.close()
        self.file.close()
