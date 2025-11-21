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
    """管理向量集合的類別"""
    
    def __init__(self, base_dir: Optional[Path] = None):
        """初始化集合管理器
        
        Args:
            base_dir: 集合儲存的基礎資料夾，預設為當前資料夾下的 collections
        """
        self.base_dir = Path(base_dir) if base_dir else Path("collections")
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_collection_dir(self, collection_name: str) -> Path:
        """獲取集合資料夾"""
        return self.base_dir / collection_name
    
    def get_vectors_path(self, collection_name: str) -> Path:
        """獲取向量檔案路徑"""
        return self._get_collection_dir(collection_name) / "vectors.npy"
    
    def get_metadata_path(self, collection_name: str) -> Path:
        """獲取元數據檔案路徑"""
        return self._get_collection_dir(collection_name) / "metadata.parquet"
    
    def get_info_path(self, collection_name: str) -> Path:
        """獲取集合資訊檔案路徑"""
        return self._get_collection_dir(collection_name) / "collection_info.json"
    
    def get_index_dir(self, collection_name: str) -> Path:
        """獲取索引資料夾"""
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
                    logger.warning(f"無法讀取集合 {path.name} 的資訊: {e}")
        return sorted(collections, key=lambda x: x.created_at, reverse=True)
    
    def get_collection_info(self, collection_name: str) -> Optional[CollectionInfo]:
        """獲取 collection 資訊
        
        Args:
            collection_name: collection 名稱
            
        Returns:
            Optional[CollectionInfo]: collection 資訊，如果不存在或損壞則返回 None
        """
        info_path = self.get_info_path(collection_name)
        if not info_path.exists():
            return None
        
        try:
            return CollectionInfo.load(info_path)
        except json.JSONDecodeError as e:
            logger.error(f"collection_info.json 損毀，無法解析: {str(e)}")
            # 嘗試備份損壞的檔案
            try:
                backup_path = info_path.with_suffix('.json.bak')
                shutil.copy2(info_path, backup_path)
                logger.info(f"已備份損壞的 collection_info.json 到 {backup_path}")
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
        """保存 collection 資訊
        
        Args:
            collection_name: collection 名稱
            info: collection 資訊
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
            
            # 將臨時檔案重命名為目標檔案
            temp_path.replace(info_path)
            
        except Exception as e:
            logger.error(f"儲存 collection_info.json 時出錯: {str(e)}")
            # 如果保存失敗，嘗試恢復備份
            try:
                backup_path = info_path.with_suffix('.json.bak')
                if backup_path.exists():
                    shutil.copy2(backup_path, info_path)
                    logger.info("已恢復備份的 collection_info.json")
            except Exception as restore_error:
                logger.error(f"恢復備份失敗: {str(restore_error)}")
            raise ValueError(f"儲存 collection 資訊失敗: {str(e)}")
    
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
            CollectionInfo: 創建的集合資訊
        """
        collection_dir = self._get_collection_dir(collection_name)
        if collection_dir.exists():
            raise ValueError(f"集合 {collection_name} 已存在")
        
        # 建立集合資料夾
        collection_dir.mkdir(parents=True)
        
        # 建立集合資訊
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
        
        # 初始化空向量檔案
        np.save(self.get_vectors_path(collection_name), np.array([], dtype=np.float32))
        
        # 初始化空元數據檔案
        pl.DataFrame({
            "text": [],
            "text_hash": [],
            "metadata": [],
            "vector_index": []
        }).write_parquet(self.get_metadata_path(collection_name))
        
        # 保存集合資訊
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
        """更新集合，添加新的向量和文字
        
        Args:
            collection_name: 集合名稱
            vectors: 新的向量數組
            texts: 新的文字列表
            metadata_list: 新的元數據列表
            
        Returns:
            CollectionInfo: 更新後的集合資訊
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
        
        # 確保現有 metadata 是字串格式（統一格式）
        if "metadata" in metadata_df.columns:
            # 如果 metadata 是 Struct 或其他類型，轉換為字串
            if metadata_df["metadata"].dtype != pl.String:
                logger.debug("轉換現有 metadata 為字串格式...")
                # 將 metadata 轉換為字串
                existing_metadata_list = []
                for row in metadata_df.iter_rows(named=True):
                    meta = row["metadata"]
                    if isinstance(meta, dict):
                        meta_str = json.dumps(meta, ensure_ascii=False)
                    elif isinstance(meta, str):
                        meta_str = meta
                    else:
                        meta_str = str(meta)
                    existing_metadata_list.append(meta_str)
                
                metadata_df = metadata_df.with_columns([
                    pl.Series("metadata", existing_metadata_list, dtype=pl.String)
                ])
        
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
            logger.warning(f"⚠️  沒有新的文字需要添加到集合 {collection_name}")
            logger.warning(f"   輸入文本數量: {len(texts)}")
            logger.warning(f"   現有文本哈希數量: {len(existing_hashes)}")
            logger.warning(f"   這可能意味著所有文本都是重複的，或者處理過程中出現問題")
            # 即使沒有新向量，也確保向量文件是正確的二維形狀
            vectors_path = self.get_vectors_path(collection_name)
            existing_vectors = np.load(vectors_path)
            if existing_vectors.ndim != 2 or (existing_vectors.ndim == 2 and existing_vectors.shape[1] != info.dimension):
                # 修正向量文件形狀
                if existing_vectors.size == 0:
                    all_vectors = np.empty((0, info.dimension), dtype=np.float32)
                else:
                    if existing_vectors.ndim == 1:
                        all_vectors = existing_vectors.reshape(-1, info.dimension)
                    else:
                        all_vectors = existing_vectors
                np.save(vectors_path, all_vectors)
                logger.info(f"已修正向量文件形狀: {all_vectors.shape}")
            return info
        
        # 記錄新增的向量數量（在合併前）
        num_new_vectors = len(new_vectors)
        
        # 更新向量文件
        vectors_path = self.get_vectors_path(collection_name)
        if info.num_vectors > 0:
            existing_vectors = np.load(vectors_path)
            all_vectors = np.vstack([existing_vectors, new_vectors])
        else:
            all_vectors = np.array(new_vectors)
        np.save(vectors_path, all_vectors)
        
        # 更新元數據
        # 確保所有列表長度一致（使用新增的數量，不是合併後的總數）
        assert len(new_texts) == len(new_hashes) == len(new_metadata) == num_new_vectors, (
            f"資料長度不一致: text={len(new_texts)}, hashes={len(new_hashes)}, "
            f"metadata={len(new_metadata)}, vectors={num_new_vectors}"
        )
        
        # 更新 metadata 中的 vector_index（使用實際的向量索引）
        # 注意：vector_index 應該對應到合併後的向量陣列中的位置
        for i, metadata in enumerate(new_metadata):
            if isinstance(metadata, dict):
                metadata["vector_index"] = info.num_vectors + i
            # 如果 metadata 是字串（JSON），則在建立 DataFrame 時處理
        
        # 建立 vector_index 列表（確保長度與 new_texts 一致，使用新增的數量）
        vector_indices = list(range(info.num_vectors, info.num_vectors + num_new_vectors))
        
        # 確保 metadata 是字串格式（如果是 dict 則轉換為 JSON）
        metadata_strs = []
        for m in new_metadata:
            if isinstance(m, dict):
                metadata_strs.append(json.dumps(m, ensure_ascii=False))
            else:
                metadata_strs.append(m)
        
        new_metadata_df = pl.DataFrame({
            "text": new_texts,
            "text_hash": new_hashes,
            "metadata": metadata_strs,
            "vector_index": vector_indices
        })
        
        # 驗證 DataFrame 的長度一致性
        if len(new_metadata_df) != len(new_texts):
            raise ValueError(
                f"DataFrame 建立失敗: 行數不一致 "
                f"(text={len(new_texts)}, df={len(new_metadata_df)})"
            )
        
        if info.num_vectors > 0:
            # 確保兩個 DataFrame 的 metadata 欄位都是字串類型
            # 重新讀取並轉換現有 metadata（以防萬一）
            if metadata_df["metadata"].dtype != pl.String:
                logger.debug("再次轉換現有 metadata 為字串格式...")
                metadata_list = []
                for row in metadata_df.iter_rows(named=True):
                    meta = row["metadata"]
                    if isinstance(meta, dict):
                        meta_str = json.dumps(meta, ensure_ascii=False)
                    elif isinstance(meta, str):
                        meta_str = meta
                    else:
                        meta_str = str(meta)
                    metadata_list.append(meta_str)
                
                metadata_df = metadata_df.with_columns([
                    pl.Series("metadata", metadata_list, dtype=pl.String)
                ])
            
            # 確保 new_metadata_df 的 metadata 也是字串類型
            if new_metadata_df["metadata"].dtype != pl.String:
                new_metadata_df = new_metadata_df.with_columns([
                    pl.Series("metadata", new_metadata_df["metadata"].to_list(), dtype=pl.String)
                ])
            
            # 合併 DataFrame
            metadata_df = pl.concat([metadata_df, new_metadata_df])
        else:
            metadata_df = new_metadata_df
        
        metadata_df.write_parquet(self.get_metadata_path(collection_name))
        
        # 更新集合資訊
        info.num_vectors = len(all_vectors)
        info.updated_at = datetime.now().isoformat()
        info.text_hashes.update(new_hashes)
        
        # 更新向量偏移量
        for i, text_hash in enumerate(new_hashes):
            info.vector_offsets[text_hash] = info.num_vectors - num_new_vectors + i
        
        self.save_collection_info(collection_name, info)
        logger.info(
            f"已更新集合 {collection_name}，"
            f"添加了 {num_new_vectors} 個新向量，"
            f"跳過了 {len(texts) - num_new_vectors} 個重複文字，"
            f"總共 {len(all_vectors)} 個向量"
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
            texts: 新的文字列表
            metadata_list: 新的元數據列表
            config: 新的集合配置
            dimension: 向量維度
            source_files: 源文件列表
            
        Returns:
            CollectionInfo: 重建後的集合資訊
        """
        # 刪除現有集合
        collection_dir = self._get_collection_dir(collection_name)
        if collection_dir.exists():
            shutil.rmtree(collection_dir)
        
        # 建立新集合
        info = self.create_collection(
            collection_name=collection_name,
            config=config,
            dimension=dimension,
            source_files=source_files
        )
        
        # 更新集合資訊
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
        """根據向量索引獲取文字和元數據
        
        Args:
            collection_name: 集合名稱
            index: 向量索引
            
        Returns:
            Optional[Tuple[str, Dict[str, Any]]]: 文字和元數據的元組，如果找不到則返回 None
        """
        metadata_df = pl.read_parquet(self.get_metadata_path(collection_name))
        row = metadata_df.filter(pl.col("vector_index") == index)
        if row.is_empty():
            return None
        
        text = row["text"][0]
        metadata_raw = row["metadata"][0]
        
        # 處理 metadata：可能是 Polars Struct、字串或字典
        metadata = {}
        
        if isinstance(metadata_raw, str):
            # 字串格式，嘗試解析為 JSON
            try:
                metadata = json.loads(metadata_raw)
            except json.JSONDecodeError:
                metadata = {"text": text, "id": index}
        elif isinstance(metadata_raw, dict):
            # 已經是字典
            metadata = metadata_raw
        else:
            # Polars Struct 類型，需要轉換
            try:
                # 使用 struct.field() 提取所有欄位
                struct_schema = row.schema["metadata"]
                if hasattr(struct_schema, 'fields'):
                    # 提取所有欄位
                    for field in struct_schema.fields:
                        field_name = field.name
                        try:
                            # 從 row 中提取 struct 欄位值
                            field_series = row.select(pl.col("metadata").struct.field(field_name))
                            if len(field_series) > 0:
                                field_value = field_series[0, 0]  # 取得實際值
                                metadata[field_name] = field_value
                        except Exception as e:
                            logger.debug(f"無法提取欄位 {field_name}: {e}")
                            pass
            except Exception as e:
                logger.warning(f"無法轉換 metadata Struct: {e}")
                metadata = {"text": text, "id": index}
        
        # 如果 metadata 中有嵌套的 metadata 字串，解析它
        if isinstance(metadata, dict) and "metadata" in metadata:
            nested_meta_str = metadata.get("metadata")
            if isinstance(nested_meta_str, str):
                try:
                    nested_meta = json.loads(nested_meta_str)
                    # 合併嵌套的 metadata（不覆蓋頂層欄位）
                    for key, value in nested_meta.items():
                        if key not in metadata:
                            metadata[key] = value
                except json.JSONDecodeError:
                    pass
        
        return text, metadata
    
    def get_text_by_hash(self, collection_name: str, text_hash: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """根據文字哈希獲取文字和元數據
        
        Args:
            collection_name: 集合名稱
            text_hash: 文字哈希
            
        Returns:
            Optional[Tuple[str, Dict[str, Any]]]: 文字和元數據的元組，如果找不到則返回 None
        """
        info = self.get_collection_info(collection_name)
        if not info or text_hash not in info.vector_offsets:
            return None
        
        return self.get_text_by_index(collection_name, info.vector_offsets[text_hash]) 