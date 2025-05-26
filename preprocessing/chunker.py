from dataclasses import dataclass
from typing import List, Optional, Literal, Dict
import re
import polars as pl
from .config import ChunkConfig

@dataclass
class TextChunk:
    """A chunk of text with metadata"""
    id: int
    text: str
    source_type: Literal["faq", "article"]
    source_id: str  # Original row ID or title
    section: Optional[str] = None
    metadata: Optional[dict] = None

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