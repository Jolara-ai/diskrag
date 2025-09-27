from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal, Dict, Any, List, Set
import yaml
from pydantic import BaseModel, Field
from datetime import datetime
import json
import hashlib
import numpy as np

@dataclass
class EmbeddingConfig:
    """Embedding 生成配置"""
    provider: str  # "openai" 或 "vertex"
    model: str  # 模型名稱
    project_id: Optional[str] = None  # Vertex AI 專案 ID
    api_key: Optional[str] = None  # OpenAI API 金鑰
    max_retries: int = 3  # 最大重試次數
    retry_delay: int = 2  # 重試延遲（秒）

@dataclass
class QuestionGenerationConfig:
    """問題生成配置"""
    enabled: bool = True  # 是否啟用問題生成
    provider: str = "openai"  # LLM 提供商
    model: str = "gpt-3.5-turbo"  # 模型名稱
    max_questions: int = 5  # 每個問答對生成的最大問題數
    temperature: float = 0.7  # 生成溫度
    max_retries: int = 3  # 最大重試次數
    retry_delay: int = 2  # 重試延遲（秒）
    project_id: Optional[str] = None  # Vertex AI 專案 ID

@dataclass
class ChunkConfig:
    """文本分塊配置"""
    size: int = 300  # 分塊大小
    overlap: int = 50  # 重疊大小
    min_size: int = 50  # 最小分塊大小

@dataclass
class OutputConfig:
    """輸出配置"""
    format: str = "parquet"  # 輸出格式
    compression: str = "snappy"  # 壓縮方式

@dataclass
class PreprocessingConfig:
    """預處理配置"""
    collection: str  # collection 名稱
    embedding: EmbeddingConfig  # embedding 配置
    question_generation: QuestionGenerationConfig  # 問題生成配置
    chunk: ChunkConfig = field(default_factory=ChunkConfig)  # 分塊配置（用於文章類型）
    output: OutputConfig = field(default_factory=OutputConfig)  # 輸出配置

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            'collection': self.collection,
            'embedding': {
                'provider': self.embedding.provider,
                'model': self.embedding.model,
                'project_id': self.embedding.project_id,
                'api_key': self.embedding.api_key,
                'max_retries': self.embedding.max_retries,
                'retry_delay': self.embedding.retry_delay
            },
            'question_generation': {
                'enabled': self.question_generation.enabled,
                'provider': self.question_generation.provider,
                'model': self.question_generation.model,
                'max_questions': self.question_generation.max_questions,
                'temperature': self.question_generation.temperature,
                'max_retries': self.question_generation.max_retries,
                'retry_delay': self.question_generation.retry_delay,
                'project_id': self.question_generation.project_id
            },
            'chunk': {
                'size': self.chunk.size,
                'overlap': self.chunk.overlap,
                'min_size': self.chunk.min_size
            },
            'output': {
                'format': self.output.format,
                'compression': self.output.compression
            }
        }

# 支持的向量維度列表（用於 PQ 量化）
SUPPORTED_DIMENSIONS = {128, 256, 768, 960, 1536}

def validate_vector_dimension(dimension: int) -> bool:
    """驗證向量維度是否支持 PQ 量化"""
    return dimension in SUPPORTED_DIMENSIONS

def get_text_hash(text: str) -> str:
    """生成文本的哈希值"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

@dataclass
class CollectionInfo:
    """Collection 資訊"""
    name: str  # collection 名稱
    config: Dict[str, Any]  # 配置資訊
    dimension: int  # 向量維度
    num_vectors: int  # 向量數量
    created_at: str  # 創建時間（ISO 格式字符串）
    updated_at: str  # 更新時間（ISO 格式字符串）
    source_files: List[str]  # 來源檔案列表
    text_hashes: Set[str] = field(default_factory=set)  # 已處理文本的哈希值集合
    vector_offsets: Dict[str, int] = field(default_factory=dict)  # 文本哈希到向量索引的映射
    chunk_stats: Dict[str, Any] = field(default_factory=dict)  # 統計資訊

    def __post_init__(self):
        """初始化後處理"""
        # 驗證向量維度
        if not validate_vector_dimension(self.dimension):
            raise ValueError(
                f"不支持的向量維度: {self.dimension}。"
                f"支持的維度: {sorted(SUPPORTED_DIMENSIONS)}"
            )

    def add_text(self, text: str, vector_index: int) -> bool:
        """添加文本及其向量索引
        
        Returns:
            bool: 如果文本是新的（未重複），返回 True
        """
        text_hash = get_text_hash(text)
        if text_hash in self.text_hashes:
            return False
        
        self.text_hashes.add(text_hash)
        self.vector_offsets[text_hash] = vector_index
        return True

    def get_vector_index(self, text: str) -> Optional[int]:
        """獲取文本對應的向量索引"""
        text_hash = get_text_hash(text)
        return self.vector_offsets.get(text_hash)

    @classmethod
    def load(cls, path: Path) -> 'CollectionInfo':
        """從文件加載 CollectionInfo"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, path: Path) -> None:
        """保存 CollectionInfo 到文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CollectionInfo':
        """從字典創建 CollectionInfo 實例"""
        # 轉換文本哈希集合
        data['text_hashes'] = set(data.get('text_hashes', []))
        
        # 轉換向量偏移映射
        data['vector_offsets'] = data.get('vector_offsets', {})
        
        # 轉換統計信息
        data['chunk_stats'] = data.get('chunk_stats', {})
        
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            'name': self.name,
            'config': self.config,
            'dimension': self.dimension,
            'num_vectors': self.num_vectors,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'source_files': self.source_files,
            'text_hashes': list(self.text_hashes),
            'vector_offsets': self.vector_offsets,
            'chunk_stats': self.chunk_stats
        }

def load_config(config_path: str) -> PreprocessingConfig:
    """從 YAML 文件加載配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # 轉換配置
    return PreprocessingConfig(
        collection=data['collection'],
        embedding=EmbeddingConfig(**data['embedding']),
        question_generation=QuestionGenerationConfig(**data['question_generation']),
        chunk=ChunkConfig(**data.get('chunk', {})),
        output=OutputConfig(**data.get('output', {}))
    )

def save_config(config: PreprocessingConfig, config_path: str) -> None:
    """保存配置到 YAML 文件"""
    data = {
        'collection': config.collection,
        'embedding': {
            'provider': config.embedding.provider,
            'model': config.embedding.model,
            'project_id': config.embedding.project_id,
            'api_key': config.embedding.api_key,
            'max_retries': config.embedding.max_retries,
            'retry_delay': config.embedding.retry_delay
        },
        'question_generation': {
            'enabled': config.question_generation.enabled,
            'provider': config.question_generation.provider,
            'model': config.question_generation.model,
            'max_questions': config.question_generation.max_questions,
            'temperature': config.question_generation.temperature,
            'max_retries': config.question_generation.max_retries,
            'retry_delay': config.question_generation.retry_delay,
            'project_id': config.question_generation.project_id
        },
        'chunk': {
            'size': config.chunk.size,
            'overlap': config.chunk.overlap,
            'min_size': config.chunk.min_size
        },
        'output': {
            'format': config.output.format,
            'compression': config.output.compression
        }
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False) 