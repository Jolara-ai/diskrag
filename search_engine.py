import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import time
from pydiskann.vamana_graph import beam_search_from_disk
from pydiskann.io.diskann_persist import MMapNodeReader, DiskANNPersist
from pydiskann.pq.pq_model import SimplePQ
from preprocessing.collection import CollectionManager
from preprocessing.config import CollectionInfo, validate_vector_dimension

logger = logging.getLogger(__name__)

class SearchEngine:
    """搜尋引擎類別"""
    
    def __init__(self, collection_name: str):
        """初始化搜尋引擎
        
        Args:
            collection_name: 要搜尋的集合名稱
        """
        self.collection_name = collection_name
        self.manager = CollectionManager()
        
        # 取得集合資訊
        self.info = self.manager.get_collection_info(collection_name)
        if not self.info:
            raise ValueError(f"找不到集合: {collection_name}")
        
        # 驗證索引是否存在
        index_dir = self.manager.get_index_dir(collection_name)
        index_path = index_dir / "index.dat"
        pq_path = index_dir / "pq_model.pkl"
        pq_codes_path = index_dir / "pq_codes.bin"
        meta_path = index_dir / "meta.json"
        
        if not all(p.exists() for p in [index_path, pq_path, pq_codes_path, meta_path]):
            raise ValueError(f"集合 {collection_name} 的索引檔案不完整")
        
        # 載入元資料
        persist = DiskANNPersist(dim=self.info.dimension, R=32)
        meta = persist.load_meta(str(meta_path))
        
        # 載入 PQ 模型和編碼
        self.pq_model = persist.load_pq_codebook(str(pq_path))
        self.pq_codes = persist.load_pq_codes(str(pq_codes_path), meta["N"], meta["n_subvectors"])
        
        # 初始化圖讀取器
        self.reader = MMapNodeReader(
            str(index_path),
            dim=self.info.dimension,
            R=32  # 與建立索引時使用的 R 值相同
        )
        
        # 驗證向量維度
        if not validate_vector_dimension(self.info.dimension):
            raise ValueError(
                f"不支援的向量維度: {self.info.dimension}。"
                f"請使用支援的維度重新建立索引"
            )
        
        logger.info(f"已載入集合 {collection_name} 的索引")
    
    def __del__(self):
        """清理資源"""
        if hasattr(self, 'reader'):
            self.reader.close()
    
    @classmethod
    def list_collections(cls) -> List[CollectionInfo]:
        """列出所有可用的集合"""
        return CollectionManager().list_collections()
    
    def search(
        self,
        query: str,
        k: int = 5,
        beam_width: int = 8,
        embedding_fn: Optional[callable] = None
    ) -> Dict[str, Any]:
        """搜尋最相似的文字
        
        Args:
            query: 查詢文字
            k: 回傳的結果數量
            beam_width: 搜尋時的波束寬度
            embedding_fn: 用於產生查詢向量的函數
            
        Returns:
            Dict[str, Any]: 包含搜尋結果和時間統計的字典
            {
                "results": List[Dict[str, Any]],  # 搜尋結果列表
                "timing": {
                    "embedding_time": float,  # embedding API 呼叫時間（秒）
                    "search_time": float,     # DiskANN 搜尋時間（秒）
                    "total_time": float       # 總耗時（秒）
                }
            }
        """
        if embedding_fn is None:
            raise ValueError("必須提供 embedding_fn 來產生查詢向量")
        
        total_start_time = time.time()
        
        # 產生查詢向量
        embedding_start_time = time.time()
        query_vector = embedding_fn(query)
        embedding_time = time.time() - embedding_start_time
        
        if query_vector.shape[0] != self.info.dimension:
            raise ValueError(
                f"查詢向量維度不匹配: 預期 {self.info.dimension}，"
                f"實際 {query_vector.shape[0]}"
            )
        
        try:
            # 執行搜尋
            search_start_time = time.time()
            results = beam_search_from_disk(
                self.reader,
                query_vector,
                start_id=0,  # 從第一個節點開始搜尋
                beam_width=beam_width,
                k=k
            )
            search_time = time.time() - search_start_time
            
            # 取得對應的文字和元資料
            search_results = []
            for dist, idx in results:
                text_data = self.manager.get_text_by_index(self.collection_name, idx)
                if text_data:
                    text, metadata = text_data
                    # 確保 metadata 是字典類型
                    if isinstance(metadata, dict):
                        result = {
                            "text": text,
                            "distance": float(dist),
                            "metadata": metadata
                        }
                    else:
                        # 如果 metadata 不是字典，建立一個包含基本資訊的字典
                        result = {
                            "text": text,
                            "distance": float(dist),
                            "metadata": {
                                "id": idx,
                                "text": text
                            }
                        }
                    search_results.append(result)
            
            total_time = time.time() - total_start_time
            
            return {
                "results": search_results,
                "timing": {
                    "embedding_time": embedding_time,
                    "search_time": search_time,
                    "total_time": total_time
                }
            }
            
        except Exception as e:
            logger.error(f"搜尋時發生錯誤: {str(e)}")
            raise
    
    def get_text_by_hash(self, text_hash: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """根據文字雜湊取得文字和元資料
        
        Args:
            text_hash: 文字雜湊
            
        Returns:
            Optional[Tuple[str, Dict[str, Any]]]: 文字和元資料的元組，如果找不到則回傳 None
        """
        return self.manager.get_text_by_hash(self.collection_name, text_hash)
    
    def get_collection_info(self) -> CollectionInfo:
        """取得當前集合的資訊"""
        return self.info 