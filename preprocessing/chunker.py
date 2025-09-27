from dataclasses import dataclass
from typing import List, Optional, Literal, Dict, Any, Tuple
import re
import polars as pl
import numpy as np
from pathlib import Path
import docx
import logging
from tqdm import tqdm
from .config import ChunkConfig
from .embedding import EmbeddingGenerator
from .collection import CollectionManager

logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """A chunk of text with metadata"""
    id: int
    text: str
    source_type: Literal["faq", "article", "document"]
    source_id: str  # Original row ID or title
    section: Optional[str] = None
    metadata: Optional[dict] = None
    image: Optional[str] = None
    manual: Optional[str] = None

@dataclass
class DocumentChunk:
    """Document chunk with additional metadata for markdown/docx files"""
    id: int
    text: str
    image: Optional[str]
    section: str
    manual: str
    
    @classmethod
    def is_valid_text(cls, text: str, min_length: int = 50, max_length: int = 300) -> bool:
        text = re.sub(r'\s+', ' ', text).strip()
        if not min_length <= len(text) <= max_length:
            return False
        if re.match(r'^[\s\W]+$', text):
            return False
        return True

class TextChunker:
    def __init__(self, config: ChunkConfig):
        self.config = config
        self._current_id = 0

    def _get_next_id(self) -> int:
        """Get next chunk ID"""
        self._current_id += 1
        return self._current_id

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove special characters that might affect chunking
        text = re.sub(r'[\r\n\t]', ' ', text)
        return text

    def _split_into_chunks(self, text: str, source_id: str, 
                          source_type: Literal["faq", "article"],
                          section: Optional[str] = None,
                          metadata: Optional[dict] = None) -> List[TextChunk]:
        """Split text into overlapping chunks"""
        text = self._clean_text(text)
        chunks = []
        
        # If text is shorter than chunk size, return as single chunk
        if len(text) <= self.config.size:
            return [TextChunk(
                id=self._get_next_id(),
                text=text,
                source_type=source_type,
                source_id=source_id,
                section=section,
                metadata=metadata
            )]

        # Split into overlapping chunks
        start = 0
        while start < len(text):
            # Get chunk of text
            end = min(start + self.config.size, len(text))
            chunk_text = text[start:end]
            
            # Try to find a good break point (sentence end or punctuation)
            if end < len(text):
                # Look for sentence endings
                for sep in ['. ', '! ', '? ', '。', '！', '？']:
                    pos = chunk_text.rfind(sep)
                    if pos > self.config.size // 2:  # Only break if not too close to start
                        end = start + pos + len(sep)
                        chunk_text = text[start:end]
                        break

            chunks.append(TextChunk(
                id=self._get_next_id(),
                text=chunk_text.strip(),
                source_type=source_type,
                source_id=source_id,
                section=section,
                metadata=metadata
            ))
            
            # Move start position, accounting for overlap
            start = end - self.config.overlap

        return chunks

    def process_faq_csv(self, df: pl.DataFrame) -> List[TextChunk]:
        """Process FAQ format CSV"""
        chunks = []
        
        for row in df.iter_rows(named=True):
            # For FAQ, we treat each Q&A pair as a single chunk
            text = f"問題：{row['question']}\n答案：{row['answer_text']}"
            if 'note' in row and row['note']:
                text += f"\n備註：{row['note']}"
                
            chunks.extend(self._split_into_chunks(
                text=text,
                source_id=str(row.get('id', row['question'])),
                source_type="faq",
                metadata={"question": row['question']}
            ))
            
        return chunks

    def process_article_csv(self, df: pl.DataFrame) -> List[TextChunk]:
        """Process article format CSV"""
        chunks = []
        
        for row in df.iter_rows(named=True):
            chunks.extend(self._split_into_chunks(
                text=row['paragraph_text'],
                source_id=str(row.get('id', row['title'])),
                source_type="article",
                section=row.get('section'),
                metadata={"title": row['title']}
            ))
            
        return chunks

    def process_csv(self, file_path: str) -> List[TextChunk]:
        """Process CSV file and determine format automatically"""
        df = pl.read_csv(file_path)
        
        # Determine CSV format based on columns
        if 'question' in df.columns and 'answer_text' in df.columns:
            return self.process_faq_csv(df)
        elif 'title' in df.columns and 'paragraph_text' in df.columns:
            return self.process_article_csv(df)
        else:
            raise ValueError(
                "Unsupported CSV format. Must be either FAQ (question, answer_text) "
                "or Article (title, paragraph_text) format."
            )

    def process_markdown(self, file_path: Path) -> List[DocumentChunk]:
        """Process markdown file and extract chunks"""
        chunks = []
        current_section = "未分類"
        current_text = []
        current_image = None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        sections = re.split(r'(?=^# )', content, flags=re.MULTILINE)
        
        for section in sections:
            lines = section.strip().split('\n')
            if not lines:
                continue
                
            if lines[0].startswith('# '):
                current_section = lines[0][2:].strip()
                lines = lines[1:]
                
            for line in lines:
                img_match = re.search(r'!\[.*?\]\((.*?)\)', line)
                if img_match:
                    current_image = img_match.group(1)
                    continue
                    
                if not line.strip():
                    continue
                    
                current_text.append(line)
                chunk_text = ' '.join(current_text)
                
                if DocumentChunk.is_valid_text(chunk_text, 
                                              min_length=self.config.min_size,
                                              max_length=self.config.size):
                    chunks.append(DocumentChunk(
                        id=len(chunks),
                        text=chunk_text,
                        image=current_image,
                        section=current_section,
                        manual=file_path.name
                    ))
                    current_text = []
                    current_image = None
        
        if current_text:
            chunk_text = ' '.join(current_text)
            if DocumentChunk.is_valid_text(chunk_text,
                                          min_length=self.config.min_size,
                                          max_length=self.config.size):
                chunks.append(DocumentChunk(
                    id=len(chunks),
                    text=chunk_text,
                    image=current_image,
                    section=current_section,
                    manual=file_path.name
                ))
        
        return chunks

    def process_docx(self, file_path: Path) -> List[DocumentChunk]:
        """Process docx file and extract chunks"""
        doc = docx.Document(file_path)
        chunks = []
        current_section = "未分類"
        current_text = []
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
                
            if para.style.name.startswith('Heading'):
                if current_text:
                    chunk_text = ' '.join(current_text)
                    if DocumentChunk.is_valid_text(chunk_text,
                                                  min_length=self.config.min_size,
                                                  max_length=self.config.size):
                        chunks.append(DocumentChunk(
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
                
                if DocumentChunk.is_valid_text(chunk_text,
                                              min_length=self.config.min_size,
                                              max_length=self.config.size):
                    chunks.append(DocumentChunk(
                        id=len(chunks),
                        text=chunk_text,
                        image=None,
                        section=current_section,
                        manual=file_path.name
                    ))
                    current_text = []
        
        if current_text:
            chunk_text = ' '.join(current_text)
            if DocumentChunk.is_valid_text(chunk_text,
                                          min_length=self.config.min_size,
                                          max_length=self.config.size):
                chunks.append(DocumentChunk(
                    id=len(chunks),
                    text=chunk_text,
                    image=None,
                    section=current_section,
                    manual=file_path.name
                ))
        
        return chunks

    def get_embeddings(self, texts: List[str], embedding_generator: EmbeddingGenerator) -> Tuple[np.ndarray, List[int]]:
        """Generate embeddings for text chunks"""
        embeddings = []
        valid_indices = []
        
        for i, text in enumerate(tqdm(texts, desc="生成向量")):
            if not DocumentChunk.is_valid_text(text,
                                              min_length=self.config.min_size,
                                              max_length=self.config.size):
                logger.debug(f"跳過無效文字 (ID: {i}): {text[:50]}...")
                continue
                
            try:
                embedding = embedding_generator.generate(text)
                embeddings.append(embedding)
                valid_indices.append(i)
            except Exception as e:
                logger.warning(f"跳過文字 (ID: {i}) 的向量生成: {str(e)}")
                continue
                
        return np.array(embeddings) if embeddings else np.array([]), valid_indices


class DocumentProcessor:
    """Document processor for handling markdown and docx files with collection management"""
    
    def __init__(self,
                 collection_name: str,
                 manual_dir: str = "data/manual",
                 config_path: str = "config.yaml",
                 embedding_model: Optional[str] = None,
                 batch_size: int = 50):
        self.collection_name = collection_name
        self.manual_dir = Path(manual_dir)
        self.batch_size = batch_size
        self.config_path = config_path
        self.manual_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"配置文件 {config_path} 不存在，使用預設設定")
            self.config = {}
        except yaml.YAMLError as e:
            logger.error(f"解析配置文件時出錯: {e}")
            raise
        
        # Initialize embedding generator
        from .config import EmbeddingConfig
        embedding_config = EmbeddingConfig(
            provider=self.config.get('embedding', {}).get('provider', 'openai'),
            model=embedding_model or self.config.get('embedding', {}).get('model', 'text-embedding-3-small'),
            api_key=None,
            max_retries=self.config.get('embedding', {}).get('max_retries', 3),
            retry_delay=self.config.get('embedding', {}).get('retry_delay', 2)
        )
        self.embedding_generator = EmbeddingGenerator(embedding_config)
        
        # Initialize collection manager
        self.collection_manager = CollectionManager()
        collection_info = self.collection_manager.get_collection_info(collection_name)
        
        if collection_info is None:
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
        
        # Initialize chunker
        from .config import ChunkConfig
        chunk_config = ChunkConfig(
            size=self.config.get('chunk', {}).get('size', 300),
            overlap=self.config.get('chunk', {}).get('overlap', 50),
            min_size=self.config.get('chunk', {}).get('min_size', 50)
        )
        self.chunker = TextChunker(chunk_config)

    def _create_collection_config(self, embedding_model: Optional[str] = None) -> Dict[str, Any]:
        return {
            "collection": self.collection_name,
            "embedding": {
                "provider": self.config.get('embedding', {}).get('provider', 'openai'),
                "model": embedding_model or self.config.get('embedding', {}).get('model', 'text-embedding-3-small'),
                "api_key": None,
                "max_retries": self.config.get('embedding', {}).get('max_retries', 3),
                "retry_delay": self.config.get('embedding', {}).get('retry_delay', 2)
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
                "size": 300,
                "overlap": 50,
                "min_size": 50
            }),
            "output": self.config.get('output', {
                "format": "parquet",
                "compression": "snappy"
            })
        }

    def process_all_documents(self):
        """Process all markdown and docx files in the manual directory"""
        try:
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
            
            all_files.sort()
            total_chunks = 0
            processed_files = []
            
            for file_path in tqdm(all_files, desc="處理文檔"):
                logger.info(f"處理文件: {file_path}")
                try:
                    if file_path.suffix == ".md":
                        chunks = self.chunker.process_markdown(file_path)
                    elif file_path.suffix == ".docx":
                        chunks = self.chunker.process_docx(file_path)
                    else:
                        logger.warning(f"不支持的文件類型: {file_path.suffix}")
                        continue
                    
                    if chunks:
                        total_chunks += len(chunks)
                        processed_files.append(file_path.name)
                        
                except Exception as e:
                    logger.error(f"處理文件 {file_path} 時出錯: {str(e)}")
                    continue
            
            if not total_chunks:
                logger.warning("沒有成功處理任何文檔")
                return
            
            # Update collection info
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

def split_markdown(content: str, chunk_size: int = 300, overlap: int = 50) -> List[Dict[str, str]]:
    """將 Markdown 文本分割成塊
    
    Args:
        content: Markdown 文本內容
        chunk_size: 每個塊的最大字符數
        overlap: 相鄰塊之間的重疊字符數
        
    Returns:
        List[Dict[str, str]]: 包含文本塊和元數據的列表
    """
    try:
        # 使用正則表達式分割文本
        sections = re.split(r'(#{1,6}\s+.*?$)', content, flags=re.MULTILINE)
        
        chunks = []
        current_section = ""
        current_text = ""
        
        for i, section in enumerate(sections):
            if not section.strip():
                continue
                
            # 檢查是否為標題
            if re.match(r'^#{1,6}\s+', section):
                # 如果當前有文本，先處理它
                if current_text:
                    # 分割當前文本
                    text_chunks = split_text(current_text, chunk_size, overlap)
                    for chunk in text_chunks:
                        chunks.append({
                            "text": chunk,
                            "section": current_section.strip() if current_section else "未分類",
                            "image": extract_image_from_text(chunk)
                        })
                    current_text = ""
                current_section = section.strip()
            else:
                # 處理圖片
                image_matches = re.finditer(r'!\[.*?\]\((.*?)\)', section)
                for match in image_matches:
                    image_url = match.group(1)
                    # 如果當前有文本，先處理它
                    if current_text:
                        text_chunks = split_text(current_text, chunk_size, overlap)
                        for chunk in text_chunks:
                            chunks.append({
                                "text": chunk,
                                "section": current_section.strip() if current_section else "未分類",
                                "image": None
                            })
                        current_text = ""
                    # 添加圖片塊
                    chunks.append({
                        "text": f"圖片: {image_url}",
                        "section": current_section.strip() if current_section else "未分類",
                        "image": image_url
                    })
                
                # 處理普通文本
                text = re.sub(r'!\[.*?\]\((.*?)\)', '', section)
                if text.strip():
                    current_text += text.strip() + "\n\n"
        
        # 處理最後的文本
        if current_text:
            text_chunks = split_text(current_text, chunk_size, overlap)
            for chunk in text_chunks:
                chunks.append({
                    "text": chunk,
                    "section": current_section.strip() if current_section else "未分類",
                    "image": extract_image_from_text(chunk)
                })
        
        return chunks
        
    except Exception as e:
        logger.error(f"分割 Markdown 文本時出錯: {str(e)}")
        raise ValueError(f"文本分割失敗: {str(e)}")

def split_docx(paragraphs: List[str], chunk_size: int = 300, overlap: int = 50) -> List[Dict[str, str]]:
    """將 Word 文檔段落分割成塊
    
    Args:
        paragraphs: 段落列表
        chunk_size: 每個塊的最大字符數
        overlap: 相鄰塊之間的重疊字符數
        
    Returns:
        List[Dict[str, str]]: 包含文本塊和元數據的列表
    """
    try:
        chunks = []
        current_section = "未分類"
        current_text = ""
        
        for para in paragraphs:
            if not para.strip():
                continue
                
            # 檢查是否為標題（簡單啟發式方法）
            if len(para) < 100 and para.strip().endswith(':'):
                # 如果當前有文本，先處理它
                if current_text:
                    text_chunks = split_text(current_text, chunk_size, overlap)
                    for chunk in text_chunks:
                        chunks.append({
                            "text": chunk,
                            "section": current_section,
                            "image": None
                        })
                    current_text = ""
                current_section = para.strip()
            else:
                current_text += para.strip() + "\n\n"
        
        # 處理最後的文本
        if current_text:
            text_chunks = split_text(current_text, chunk_size, overlap)
            for chunk in text_chunks:
                chunks.append({
                    "text": chunk,
                    "section": current_section,
                    "image": None
                })
        
        return chunks
        
    except Exception as e:
        logger.error(f"分割 Word 文檔時出錯: {str(e)}")
        raise ValueError(f"文檔分割失敗: {str(e)}")

def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """將文本分割成指定大小的塊
    
    Args:
        text: 要分割的文本
        chunk_size: 每個塊的最大字符數
        overlap: 相鄰塊之間的重疊字符數
        
    Returns:
        List[str]: 文本塊列表
    """
    try:
        if not text.strip():
            return []
            
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # 計算當前塊的結束位置
            end = min(start + chunk_size, text_length)
            
            # 如果不是最後一塊，嘗試在句子邊界分割
            if end < text_length:
                # 在 chunk_size 範圍內尋找最後一個句子結束標記
                last_period = text.rfind('.', start, end)
                last_question = text.rfind('?', start, end)
                last_exclamation = text.rfind('!', start, end)
                last_newline = text.rfind('\n', start, end)
                
                # 找到最接近 chunk_size 的句子邊界
                sentence_end = max(last_period, last_question, last_exclamation, last_newline)
                if sentence_end > start + chunk_size // 2:  # 確保不會產生太小的塊
                    end = sentence_end + 1
            
            # 提取當前塊
            chunk = text[start:end].strip()
            if chunk:  # 只添加非空塊
                chunks.append(chunk)
            
            # 更新起始位置，考慮重疊
            start = end - overlap if end < text_length else text_length
        
        return chunks
        
    except Exception as e:
        logger.error(f"分割文本時出錯: {str(e)}")
        raise ValueError(f"文本分割失敗: {str(e)}")

def extract_image_from_text(text: str) -> Optional[str]:
    """從文本中提取圖片 URL
    
    Args:
        text: 包含圖片標記的文本
        
    Returns:
        Optional[str]: 圖片 URL，如果沒有找到則返回 None
    """
    try:
        match = re.search(r'!\[.*?\]\((.*?)\)', text)
        return match.group(1) if match else None
    except Exception:
        return None 