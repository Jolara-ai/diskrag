from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Set
import json
import logging
from pathlib import Path
import polars as pl
import numpy as np
from datetime import datetime
from .config import CollectionInfo, PreprocessingConfig, get_text_hash
import shutil

logger = logging.getLogger(__name__)

class CollectionManager:
    """管理向量集合的類"""
    
    def __init__(self, base_dir: Optional[Path] = None):
        """初始化集合管理器
        
        Args:
            base_dir: 集合存儲的基礎目錄，默認為當前目錄下的 collections
        """
        self.base_dir = Path(base_dir) if base_dir else Path("collections")
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_collection_dir(self, collection_name: str) -> Path:
        """獲取集合目錄"""
        return self.base_dir / collection_name
    
    def get_vectors_path(self, collection_name: str) -> Path:
        """獲取向量文件路徑"""
        return self._get_collection_dir(collection_name) / "vectors.npy"
    
    def get_metadata_path(self, collection_name: str) -> Path:
        """獲取元數據文件路徑"""
        return self._get_collection_dir(collection_name) / "metadata.parquet"
    
    def get_info_path(self, collection_name: str) -> Path:
        """獲取集合信息文件路徑"""
        return self._get_collection_dir(collection_name) / "collection_info.json"
    
    def get_index_dir(self, collection_name: str) -> Path:
        """獲取索引目錄"""
        return self._get_collection_dir(collection_name) / "index"
    
    def list_collections(self) -> List[CollectionInfo]:
        """列出所有可用的集合"""
        collections = []
        for path in self.base_dir.iterdir():
            if path.is_dir() and (path / "collection_info.json").exists():
                try:
                    info = self.get_collection_info(path.name)
                    if info:
                        collections.append(info)
                except Exception as e:
                    logger.warning(f"無法讀取集合 {path.name} 的信息: {e}")
        return sorted(collections, key=lambda x: x.created_at, reverse=True)
    
    def get_collection_info(self, collection_name: str) -> Optional[CollectionInfo]:
        """獲取 collection 信息
        
        Args:
            collection_name: collection 名稱
            
        Returns:
            Optional[CollectionInfo]: collection 信息，如果不存在或損壞則返回 None
        """
        info_path = self.get_info_path(collection_name)
        if not info_path.exists():
            return None
        
        try:
            return CollectionInfo.load(info_path)
        except json.JSONDecodeError as e:
            logger.error(f"collection_info.json 損毀，無法解析: {str(e)}")
            # 嘗試備份損壞的文件
            try:
                backup_path = info_path.with_suffix('.json.bak')
                shutil.copy2(info_path, backup_path)
                logger.info(f"已備份損壞的 collection_info.json 到: {backup_path}")
            except Exception as backup_error:
                logger.error(f"備份損壞的 collection_info.json 失敗: {str(backup_error)}")
            
            # 嘗試從備份恢復
            try:
                backup_path = info_path.with_suffix('.json.bak')
                if backup_path.exists():
                    logger.info(f"嘗試從備份恢復: {backup_path}")
                    return CollectionInfo.load(backup_path)
            except Exception as restore_error:
                logger.error(f"從備份恢復失敗: {str(restore_error)}")
            
            return None
        except Exception as e:
            logger.error(f"讀取 collection_info.json 時出錯: {str(e)}")
            return None
    
    def save_collection_info(self, collection_name: str, info: CollectionInfo) -> None:
        """保存 collection 信息
        
        Args:
            collection_name: collection 名稱
            info: collection 信息
        """
        info_path = self.get_info_path(collection_name)
        
        # 在保存之前先驗證數據
        try:
            # 驗證 info 是否可以正確序列化
            json_str = json.dumps(info.to_dict(), ensure_ascii=False, indent=2)
            # 驗證是否可以正確解析
            json.loads(json_str)
            
            # 如果驗證通過，先寫入臨時文件
            temp_path = info_path.with_suffix('.json.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
            
            # 如果原文件存在，先備份
            if info_path.exists():
                backup_path = info_path.with_suffix('.json.bak')
                shutil.copy2(info_path, backup_path)
            
            # 將臨時文件重命名為目標文件
            temp_path.replace(info_path)
            
        except Exception as e:
            logger.error(f"保存 collection_info.json 時出錯: {str(e)}")
            # 如果保存失敗，嘗試恢復備份
            try:
                backup_path = info_path.with_suffix('.json.bak')
                if backup_path.exists():
                    shutil.copy2(backup_path, info_path)
                    logger.info("已恢復備份的 collection_info.json")
            except Exception as restore_error:
                logger.error(f"恢復備份失敗: {str(restore_error)}")
            raise ValueError(f"保存 collection 信息失敗: {str(e)}")
    
    def create_collection(
        self,
        collection_name: str,
        config: Dict[str, Any],
        dimension: int,
        source_files: List[str]
    ) -> CollectionInfo:
        """創建新的集合
        
        Args:
            collection_name: 集合名稱
            config: 集合配置
            dimension: 向量維度
            source_files: 源文件列表
            
        Returns:
            CollectionInfo: 創建的集合信息
        """
        collection_dir = self._get_collection_dir(collection_name)
        if collection_dir.exists():
            raise ValueError(f"集合 {collection_name} 已存在")
        
        # 創建集合目錄
        collection_dir.mkdir(parents=True)
        
        # 創建集合信息
        info = CollectionInfo(
            name=collection_name,
            config=config,
            dimension=dimension,
            num_vectors=0,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            source_files=source_files,
            text_hashes=set(),
            vector_offsets={},
            chunk_stats={}
        )
        
        # 初始化空向量文件
        np.save(self.get_vectors_path(collection_name), np.array([], dtype=np.float32))
        
        # 初始化空元數據文件
        pl.DataFrame({
            "text": [],
            "text_hash": [],
            "metadata": [],
            "vector_index": []
        }).write_parquet(self.get_metadata_path(collection_name))
        
        # 保存集合信息
        self.save_collection_info(collection_name, info)
        logger.info(f"已創建集合 {collection_name}")
        
        return info
    
    def update_collection(
        self,
        collection_name: str,
        vectors: np.ndarray,
        texts: List[str],
        metadata_list: List[Dict[str, Any]]
    ) -> CollectionInfo:
        """更新集合，添加新的向量和文本
        
        Args:
            collection_name: 集合名稱
            vectors: 新的向量數組
            texts: 新的文本列表
            metadata_list: 新的元數據列表
            
        Returns:
            CollectionInfo: 更新後的集合信息
        """
        info = self.get_collection_info(collection_name)
        if not info:
            raise ValueError(f"找不到集合 {collection_name}")
        
        if vectors.shape[1] != info.dimension:
            raise ValueError(
                f"向量維度不匹配: 預期 {info.dimension}，"
                f"實際 {vectors.shape[1]}"
            )
        
        # 計算文本哈希
        text_hashes = [get_text_hash(text) for text in texts]
        
        # 加載現有元數據
        metadata_df = pl.read_parquet(self.get_metadata_path(collection_name))
        existing_hashes = set(metadata_df["text_hash"].to_list())
        
        # 過濾重複文本
        new_indices = []
        new_vectors = []
        new_texts = []
        new_metadata = []
        new_hashes = []
        
        for i, (text, text_hash, metadata) in enumerate(zip(texts, text_hashes, metadata_list)):
            if text_hash not in existing_hashes:
                new_indices.append(i)
                new_vectors.append(vectors[i])
                new_texts.append(text)
                new_metadata.append(metadata)
                new_hashes.append(text_hash)
        
        if not new_vectors:
            logger.info(f"沒有新的文本需要添加到集合 {collection_name}")
            return info
        
        # 更新向量文件
        vectors_path = self.get_vectors_path(collection_name)
        if info.num_vectors > 0:
            existing_vectors = np.load(vectors_path)
            new_vectors = np.vstack([existing_vectors, new_vectors])
        np.save(vectors_path, new_vectors)
        
        # 更新元數據
        new_metadata_df = pl.DataFrame({
            "text": new_texts,
            "text_hash": new_hashes,
            "metadata": new_metadata,
            "vector_index": list(range(info.num_vectors, info.num_vectors + len(new_vectors)))
        })
        
        if info.num_vectors > 0:
            metadata_df = pl.concat([metadata_df, new_metadata_df])
        else:
            metadata_df = new_metadata_df
        
        metadata_df.write_parquet(self.get_metadata_path(collection_name))
        
        # 更新集合信息
        info.num_vectors = len(new_vectors)
        info.updated_at = datetime.now().isoformat()
        info.text_hashes.update(new_hashes)
        
        # 更新向量偏移量
        for i, text_hash in enumerate(new_hashes):
            info.vector_offsets[text_hash] = info.num_vectors - len(new_vectors) + i
        
        self.save_collection_info(collection_name, info)
        logger.info(
            f"已更新集合 {collection_name}，"
            f"添加了 {len(new_vectors)} 個新向量，"
            f"跳過了 {len(texts) - len(new_vectors)} 個重複文本"
        )
        
        return info
    
    def rebuild_collection(
        self,
        collection_name: str,
        vectors: np.ndarray,
        texts: List[str],
        metadata_list: List[Dict[str, Any]],
        config: Dict[str, Any],
        dimension: int,
        source_files: List[str]
    ) -> CollectionInfo:
        """重建集合，完全替換現有數據
        
        Args:
            collection_name: 集合名稱
            vectors: 新的向量數組
            texts: 新的文本列表
            metadata_list: 新的元數據列表
            config: 新的集合配置
            dimension: 向量維度
            source_files: 源文件列表
            
        Returns:
            CollectionInfo: 重建後的集合信息
        """
        # 刪除現有集合
        collection_dir = self._get_collection_dir(collection_name)
        if collection_dir.exists():
            shutil.rmtree(collection_dir)
        
        # 創建新集合
        info = self.create_collection(
            collection_name,
            config,
            dimension,
            source_files
        )
        
        # 更新集合
        return self.update_collection(
            collection_name,
            vectors,
            texts,
            metadata_list
        )
    
    def delete_collection(self, collection_name: str) -> None:
        """刪除集合"""
        collection_dir = self._get_collection_dir(collection_name)
        if not collection_dir.exists():
            raise ValueError(f"找不到集合 {collection_name}")
        
        shutil.rmtree(collection_dir)
        logger.info(f"已刪除集合 {collection_name}")
    
    def get_text_by_index(self, collection_name: str, index: int) -> Optional[Tuple[str, Dict[str, Any]]]:
        """根據向量索引獲取文本和元數據
        
        Args:
            collection_name: 集合名稱
            index: 向量索引
            
        Returns:
            Optional[Tuple[str, Dict[str, Any]]]: 文本和元數據的元組，如果找不到則返回 None
        """
        metadata_df = pl.read_parquet(self.get_metadata_path(collection_name))
        row = metadata_df.filter(pl.col("vector_index") == index)
        if row.is_empty():
            return None
        
        text = row["text"][0]
        metadata = row["metadata"][0]
        
        # 確保 metadata 是字典類型
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                # 如果無法解析為 JSON，創建一個包含基本信息的字典
                metadata = {
                    "text": text,
                    "id": index
                }
        elif not isinstance(metadata, dict):
            # 如果 metadata 既不是字符串也不是字典，創建一個基本字典
            metadata = {
                "text": text,
                "id": index
            }
        
        return text, metadata
    
    def get_text_by_hash(self, collection_name: str, text_hash: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """根據文本哈希獲取文本和元數據
        
        Args:
            collection_name: 集合名稱
            text_hash: 文本哈希
            
        Returns:
            Optional[Tuple[str, Dict[str, Any]]]: 文本和元數據的元組，如果找不到則返回 None
        """
        info = self.get_collection_info(collection_name)
        if not info or text_hash not in info.vector_offsets:
            return None
        
        return self.get_text_by_index(collection_name, info.vector_offsets[text_hash]) 