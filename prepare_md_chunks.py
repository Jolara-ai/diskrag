import os
import json
import polars as pl
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from openai import OpenAI
from openai import APIError, RateLimitError, Timeout

from tqdm import tqdm
import docx
import re
from dataclasses import dataclass
import logging
import time
import yaml
from preprocessing.collection import CollectionManager
from preprocessing.config import PreprocessingConfig, CollectionInfo, EmbeddingConfig, ChunkConfig, OutputConfig
from preprocessing.embedding import EmbeddingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 文字過濾配置
MIN_TEXT_LENGTH = 50  # 最小文字長度
MAX_TEXT_LENGTH = 300  # 最大文字長度
MAX_RETRIES = 3  # API 調用最大重試次數
RETRY_DELAY = 2  # 重試延遲（秒）

@dataclass
class Chunk:
    id: int
    text: str
    image: Optional[str]
    section: str
    manual: str

    @classmethod
    def is_valid_text(cls, text: str) -> bool:
        """檢查文字是否有效"""
        # 移除空白字符
        text = re.sub(r'\s+', ' ', text).strip()
        # 檢查長度
        if not MIN_TEXT_LENGTH <= len(text) <= MAX_TEXT_LENGTH:
            return False
        # 檢查是否只包含標點符號
        if re.match(r'^[\s\W]+$', text):
            return False
        return True

class DocumentProcessor:
    def __init__(self, 
                 collection_name: str,
                 manual_dir: str = "data/manual",
                 config_path: str = "config.yaml",
                 embedding_model: Optional[str] = None,
                 batch_size: int = 50):
        """初始化文檔處理器
        
        Args:
            collection_name: collection 名稱
            manual_dir: 手冊檔案目錄
            config_path: 配置文件路徑
            embedding_model: embedding 模型名稱（可選，優先使用配置文件中的設定）
            batch_size: 批量處理大小
        """
        self.collection_name = collection_name
        self.manual_dir = Path(manual_dir)
        self.batch_size = batch_size
        
        # 創建必要的目錄
        self.manual_dir.mkdir(parents=True, exist_ok=True)
        
        # 加載配置文件
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"配置文件 {config_path} 不存在，使用預設設定")
            self.config = {}
        except yaml.YAMLError as e:
            logger.error(f"解析配置文件時出錯: {e}")
            raise
        
        # 初始化 embedding generator
        embedding_config = EmbeddingConfig(
            provider=self.config.get('embedding', {}).get('provider', 'openai'),
            model=embedding_model or self.config.get('embedding', {}).get('model', 'text-embedding-3-small'),
            api_key=None,  # 從環境變數讀取
            max_retries=self.config.get('embedding', {}).get('max_retries', MAX_RETRIES),
            retry_delay=self.config.get('embedding', {}).get('retry_delay', RETRY_DELAY)
        )
        self.embedding_generator = EmbeddingGenerator(embedding_config)
        
        # 初始化 collection manager
        self.collection_manager = CollectionManager()
        
        # 獲取或創建 collection
        collection_info = self.collection_manager.get_collection_info(collection_name)
        if collection_info is None:
            # 創建新的 collection
            dimension = self.embedding_generator.get_embedding_dimension()
            config = self._create_collection_config(embedding_model)
            self.collection_manager.create_collection(
                collection_name=collection_name,
                config=config,
                dimension=dimension,
                source_files=[]
            )
            collection_info = self.collection_manager.get_collection_info(collection_name)
        
        self.collection_info = collection_info
    
    def _create_collection_config(self, embedding_model: Optional[str] = None) -> Dict[str, Any]:
        """創建 collection 配置"""
        return {
            "collection": self.collection_name,
            "embedding": {
                "provider": self.config.get('embedding', {}).get('provider', 'openai'),
                "model": embedding_model or self.config.get('embedding', {}).get('model', 'text-embedding-3-small'),
                "api_key": None,  # 從環境變數讀取
                "max_retries": self.config.get('embedding', {}).get('max_retries', MAX_RETRIES),
                "retry_delay": self.config.get('embedding', {}).get('retry_delay', RETRY_DELAY)
            },
            "question_generation": self.config.get('question_generation', {
                "enabled": False,
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "max_questions": 5,
                "temperature": 0.7,
                "max_retries": 3,
                "retry_delay": 2
            }),
            "chunk": self.config.get('chunk', {
                "size": MAX_TEXT_LENGTH,
                "overlap": 50,
                "min_size": MIN_TEXT_LENGTH
            }),
            "output": self.config.get('output', {
                "format": "parquet",
                "compression": "snappy"
            })
        }

    def get_embeddings(self, texts: List[str]) -> Tuple[np.ndarray, List[int]]:
        """批量獲取文字的 embedding 向量，返回向量陣列和有效文字的索引"""
        embeddings = []
        valid_indices = []
        
        for i, text in enumerate(tqdm(texts, desc="生成向量")):
            if not Chunk.is_valid_text(text):
                logger.debug(f"跳過無效文字 (ID: {i}): {text[:50]}...")
                continue
                
            try:
                embedding = self.embedding_generator.generate(text)
                embeddings.append(embedding)
                valid_indices.append(i)
            except Exception as e:
                logger.warning(f"跳過文字 (ID: {i}) 的向量生成: {str(e)}")
                continue
        
        return np.array(embeddings) if embeddings else np.array([]), valid_indices

    def process_markdown(self, file_path: Path) -> List[Chunk]:
        """處理 Markdown 檔案，返回文字塊列表"""
        chunks = []
        current_section = "未分類"
        current_text = []
        current_image = None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 分割文檔為章節
        sections = re.split(r'(?=^# )', content, flags=re.MULTILINE)
        
        for section in sections:
            lines = section.strip().split('\n')
            if not lines:
                continue
                
            # 提取標題
            if lines[0].startswith('# '):
                current_section = lines[0][2:].strip()
                lines = lines[1:]
            
            # 處理內容
            for line in lines:
                # 檢查是否包含圖片
                img_match = re.search(r'!\[.*?\]\((.*?)\)', line)
                if img_match:
                    current_image = img_match.group(1)
                    continue
                
                # 跳過空行或只包含空白字符的行
                if not line.strip():
                    continue
                    
                current_text.append(line)
                
                # 如果累積的文字達到最小長度，創建一個新的塊
                chunk_text = ' '.join(current_text)
                if Chunk.is_valid_text(chunk_text):
                    chunks.append(Chunk(
                        id=len(chunks),
                        text=chunk_text,
                        image=current_image,
                        section=current_section,
                        manual=file_path.name
                    ))
                    current_text = []
                    current_image = None
        
        # 處理剩餘的文字
        if current_text:
            chunk_text = ' '.join(current_text)
            if Chunk.is_valid_text(chunk_text):
                chunks.append(Chunk(
                    id=len(chunks),
                    text=chunk_text,
                    image=current_image,
                    section=current_section,
                    manual=file_path.name
                ))
        
        return chunks

    def process_docx(self, file_path: Path) -> List[Chunk]:
        """處理 DOCX 檔案，返回文字塊列表"""
        doc = docx.Document(file_path)
        chunks = []
        current_section = "未分類"
        current_text = []
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
                
            # 檢查是否是標題
            if para.style.name.startswith('Heading'):
                if current_text:
                    chunk_text = ' '.join(current_text)
                    if Chunk.is_valid_text(chunk_text):
                        chunks.append(Chunk(
                            id=len(chunks),
                            text=chunk_text,
                            image=None,
                            section=current_section,
                            manual=file_path.name
                        ))
                current_section = text
                current_text = []
            else:
                current_text.append(text)
                
                chunk_text = ' '.join(current_text)
                if Chunk.is_valid_text(chunk_text):
                    chunks.append(Chunk(
                        id=len(chunks),
                        text=chunk_text,
                        image=None,
                        section=current_section,
                        manual=file_path.name
                    ))
                    current_text = []
        
        # 處理剩餘的文字
        if current_text:
            chunk_text = ' '.join(current_text)
            if Chunk.is_valid_text(chunk_text):
                chunks.append(Chunk(
                    id=len(chunks),
                    text=chunk_text,
                    image=None,
                    section=current_section,
                    manual=file_path.name
                ))
        
        return chunks

    def process_all_documents(self):
        """處理所有文檔並更新 collection"""
        try:
            # 加載配置
            with open(self.config_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
            config = PreprocessingConfig(**config_dict)
            
            # 初始化預處理器
            preprocessor = Preprocessor(config)
            
            # 獲取所有文件
            manual_path = Path(self.manual_dir)
            if not manual_path.exists():
                logger.error(f"目錄不存在: {self.manual_dir}")
                return
            
            all_files = []
            for ext in [".md", ".docx"]:
                all_files.extend(list(manual_path.glob(f"*{ext}")))
            
            if not all_files:
                logger.warning(f"在 {self.manual_dir} 中沒有找到需要處理的文檔")
                return
            
            # 按文件名排序
            all_files.sort()
            
            # 處理每個文件
            total_chunks = 0
            processed_files = []
            
            for file_path in tqdm(all_files, desc="處理文檔"):
                logger.info(f"處理文件: {file_path}")
                
                try:
                    if file_path.suffix == ".md":
                        chunks = self.process_markdown(file_path)
                    elif file_path.suffix == ".docx":
                        chunks = self.process_docx(file_path)
                    else:
                        logger.warning(f"不支持的文件類型: {file_path.suffix}")
                        continue
                    
                    if chunks:
                        total_chunks += len(chunks)
                        processed_files.append(file_path.name)
                        
                except Exception as e:
                    logger.error(f"處理文件 {file_path} 時出錯: {str(e)}")
                    # 繼續處理下一個文件
                    continue
            
            # 檢查是否成功處理了任何文件
            if not total_chunks:
                logger.warning("沒有成功處理任何文檔")
                return
            
            # 更新統計信息
            info = self.collection_manager.get_collection_info(self.collection_name)
            if info:
                info.chunk_stats.update({
                    "total_chunks": total_chunks,
                    "last_processed_files": processed_files
                })
                self.collection_manager.save_collection_info(self.collection_name, info)
            
            logger.info(f"成功處理 {total_chunks} 個文本塊，來自 {len(processed_files)} 個文件")
            
        except Exception as e:
            logger.error(f"處理文檔時出錯: {str(e)}")
            raise

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Process manual documents into vector collection")
    parser.add_argument("--collection", help="Collection name (optional, will use config.yaml if not specified)")
    parser.add_argument("--manual-dir", default="data/manual", help="Directory containing manual documents")
    parser.add_argument("--model", help="Embedding model to use (optional, will use config.yaml if not specified)")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # 嘗試從配置文件讀取 collection 名稱
    collection_name = args.collection
    if not collection_name:
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                collection_name = config.get('collection')
                if not collection_name:
                    parser.error("Collection name not found in config.yaml and not specified via --collection")
        except FileNotFoundError:
            parser.error(f"Config file {args.config} not found")
        except yaml.YAMLError as e:
            parser.error(f"Error parsing config file: {e}")
    
    processor = DocumentProcessor(
        collection_name=collection_name,
        manual_dir=args.manual_dir,
        config_path=args.config,
        embedding_model=args.model
    )
    processor.process_all_documents() 