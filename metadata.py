import polars as pl
from pathlib import Path
from typing import List, Optional, Dict
import logging
import mmap
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetadataManager:
    def __init__(self, metadata_path: str = "data/chunks/metadata.parquet"):
        self.metadata_path = Path(metadata_path)
        self._df = None
        self._mmap = None
        
    @property
    def df(self) -> pl.DataFrame:
        """延遲載入 DataFrame，使用記憶體映射"""
        if self._df is None:
            if not self.metadata_path.exists():
                raise FileNotFoundError(f"元資料檔案不存在: {self.metadata_path}")
            
            logger.info("使用記憶體映射載入元資料...")
            # 使用記憶體映射模式讀取 parquet 檔案
            self._df = pl.read_parquet(
                self.metadata_path,
                use_pyarrow=True,
                memory_map=True
            )
            logger.info(f"已載入元資料，共 {len(self._df)} 筆記錄")
        return self._df
    
    def get_by_ids(self, ids: List[int]) -> List[Dict]:
        """根據 ID 列表取得元資料"""
        if not ids:
            return []
            
        result = self.df.filter(pl.col("id").is_in(ids))
        return result.to_dicts()
    
    def get_by_manual(self, manual: str) -> List[Dict]:
        """取得指定手冊的所有內容"""
        result = self.df.filter(pl.col("manual") == manual)
        return result.to_dicts()
    
    def get_by_section(self, section: str) -> List[Dict]:
        """取得指定章節的所有內容"""
        result = self.df.filter(pl.col("section") == section)
        return result.to_dicts()
    
    def get_stats(self) -> Dict:
        """取得元資料統計資訊"""
        stats = {
            "total_chunks": len(self.df),
            "manuals": self.df.select("manual").unique().to_series().to_list(),
            "sections": self.df.select("section").unique().to_series().to_list(),
            "total_images": self.df.filter(pl.col("image").is_not_null()).height,
            "memory_mapped": True
        }
        return stats
    
    def preview(self, n: int = 5) -> List[Dict]:
        """預覽資料"""
        return self.df.head(n).to_dicts()
    
    def search_text(self, keyword: str) -> List[Dict]:
        """簡單文字搜尋（用於除錯）"""
        result = self.df.filter(pl.col("text").str.contains(keyword, literal=True))
        return result.to_dicts()
    
    def __del__(self):
        """清理資源"""
        if self._df is not None:
            del self._df
        if self._mmap is not None:
            self._mmap.close()
            del self._mmap 