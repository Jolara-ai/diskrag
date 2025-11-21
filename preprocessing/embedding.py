from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import logging
import time
import os
import hashlib
import json
from pathlib import Path

# 載入環境變數
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # 如果沒有安裝 python-dotenv，嘗試手動載入 .env 檔案
    from pathlib import Path
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

from openai import OpenAI
from .config import EmbeddingConfig, get_text_hash

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingResult:
    """Embedding 生成結果"""
    vector: np.ndarray
    text: str
    metadata: Optional[dict] = None

class EmbeddingGenerator:
    def __init__(self, config: EmbeddingConfig, cache_dir: Optional[Path] = None):
        self.config = config
        self._setup_clients()
        
        # 設定暫存目錄
        if cache_dir is None:
            cache_dir = Path(".cache/embeddings")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 建立暫存 key（基於 provider 和 model）
        cache_key = f"{config.provider}_{config.model}".replace("/", "_").replace(":", "_")
        self.cache_subdir = self.cache_dir / cache_key
        self.cache_subdir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Embedding 暫存目錄: {self.cache_subdir}")

    def _setup_clients(self):
        """建立 embedding 客戶端基於配置"""
        if self.config.provider == "openai":
            # 檢查 OpenAI API Key
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY 環境變數未設置。請：\n"
                    "1. 在 .env 檔案中設置 OPENAI_API_KEY=your-api-key\n"
                    "2. 或設置環境變數：export OPENAI_API_KEY=your-api-key"
                )
            self.openai_client = OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def _get_cache_path(self, text: str) -> Path:
        """取得暫存檔案路徑"""
        text_hash = get_text_hash(text)
        return self.cache_subdir / f"{text_hash}.npz"
    
    def _load_from_cache(self, text: str) -> Optional[np.ndarray]:
        """從暫存載入 embedding"""
        cache_path = self._get_cache_path(text)
        if cache_path.exists():
            try:
                data = np.load(cache_path)
                vector = data['vector']
                logger.debug(f"從暫存載入 embedding: {text[:50]}...")
                return vector
            except Exception as e:
                logger.warning(f"載入暫存失敗: {str(e)}")
                return None
        return None
    
    def _save_to_cache(self, text: str, vector: np.ndarray) -> None:
        """儲存 embedding 到暫存"""
        cache_path = self._get_cache_path(text)
        try:
            np.savez_compressed(cache_path, vector=vector)
            logger.debug(f"儲存 embedding 到暫存: {text[:50]}...")
        except Exception as e:
            logger.warning(f"儲存暫存失敗: {str(e)}")
    
    def generate(self, text: str, use_cache: bool = True) -> np.ndarray:
        """生成單個文本的 embedding 向量
        
        Args:
            text: 要生成向量的文本
            use_cache: 是否使用暫存（預設為 True）
            
        Returns:
            np.ndarray: 文本的 embedding 向量
            
        Raises:
            RuntimeError: 如果生成向量失敗
        """
        # 嘗試從暫存載入
        if use_cache:
            cached_vector = self._load_from_cache(text)
            if cached_vector is not None:
                return cached_vector
        
        # 生成新的 embedding
        if self.config.provider == "openai":
            vector = self._get_openai_embedding(text, self.config.max_retries)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
            
        if vector is None:
            raise RuntimeError(f"Failed to generate embedding for text: {text[:50]}...")
        
        # 儲存到暫存
        if use_cache:
            self._save_to_cache(text, vector)
            
        return vector

    def _get_openai_embedding(self, text: str, max_retries: int = 3) -> Optional[np.ndarray]:
        """Get embedding using OpenAI API"""
        for attempt in range(max_retries):
            try:
                response = self.openai_client.embeddings.create(
                    model=self.config.model,
                    input=text
                )
                return np.array(response.data[0].embedding)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to get embedding after {max_retries} attempts: {str(e)}")
                    return None
                logger.warning(f"Retrying embedding generation (attempt {attempt + 1}/{max_retries})")
                time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff

    def generate_embeddings(self, 
                          texts: List[str],
                          metadata_list: Optional[List[dict]] = None,
                          use_cache: bool = True) -> Tuple[List[EmbeddingResult], List[int]]:
        """Generate embeddings for a list of texts
        
        Args:
            texts: 文字列表
            metadata_list: 元數據列表（可選）
            use_cache: 是否使用暫存（預設為 True）
            
        Returns:
            Tuple[List[EmbeddingResult], List[int]]: (結果列表, 有效索引列表)
        """
        results = []
        valid_indices = []
        
        # 統計暫存命中率
        cache_hits = 0
        cache_misses = 0
        
        for i, (text, metadata) in enumerate(zip(texts, metadata_list or [None] * len(texts))):
            try:
                # 檢查暫存
                if use_cache:
                    cached_vector = self._load_from_cache(text)
                    if cached_vector is not None:
                        vector = cached_vector
                        cache_hits += 1
                    else:
                        vector = self.generate(text, use_cache=True)
                        cache_misses += 1
                else:
                    vector = self.generate(text, use_cache=False)
                    cache_misses += 1
                
                results.append(EmbeddingResult(
                    vector=vector,
                    text=text,
                    metadata=metadata
                ))
                valid_indices.append(i)
            except Exception as e:
                logger.error(f"Failed to generate embedding for text {i}: {str(e)}")
                continue
        
        # 記錄暫存統計
        if use_cache and (cache_hits > 0 or cache_misses > 0):
            total = cache_hits + cache_misses
            hit_rate = (cache_hits / total * 100) if total > 0 else 0
            logger.info(f"Embedding 暫存統計: {cache_hits}/{total} 命中 ({hit_rate:.1f}%)")
                
        return results, valid_indices

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for the current model"""
        # Test embedding to get dimension
        test_text = "This is a test."
        vector = self.generate(test_text)
        return len(vector) 