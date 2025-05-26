from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import logging
import time
from openai import OpenAI
from vertexai.preview.language_models import TextEmbeddingModel
from .config import EmbeddingConfig

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    vector: np.ndarray
    text: str
    metadata: Optional[dict] = None

class EmbeddingGenerator:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._setup_clients()

    def _setup_clients(self):
        """Setup embedding clients based on configuration"""
        if self.config.provider == "openai":
            self.openai_client = OpenAI()
        elif self.config.provider == "vertexai":
            # Vertex AI client will be initialized when needed
            pass
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def generate(self, text: str) -> np.ndarray:
        """生成單個文本的 embedding 向量
        
        Args:
            text: 要生成向量的文本
            
        Returns:
            np.ndarray: 文本的 embedding 向量
            
        Raises:
            RuntimeError: 如果生成向量失敗
        """
        if self.config.provider == "openai":
            vector = self._get_openai_embedding(text, self.config.max_retries)
        elif self.config.provider == "vertexai":
            vector = self._get_vertexai_embedding(text, self.config.max_retries)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
            
        if vector is None:
            raise RuntimeError(f"Failed to generate embedding for text: {text[:50]}...")
            
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

    def _get_vertexai_embedding(self, text: str, max_retries: int = 3) -> Optional[np.ndarray]:
        """Get embedding using Vertex AI API"""
        if not self.config.project_id:
            raise ValueError("project_id is required for Vertex AI")

        for attempt in range(max_retries):
            try:
                model = TextEmbeddingModel.from_pretrained(self.config.model)
                response = model.get_embeddings(
                    [text],
                    project=self.config.project_id
                )
                return np.array(response[0].values)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to get embedding after {max_retries} attempts: {str(e)}")
                    return None
                logger.warning(f"Retrying embedding generation (attempt {attempt + 1}/{max_retries})")
                time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff

    def generate_embeddings(self, 
                          texts: List[str],
                          metadata_list: Optional[List[dict]] = None) -> Tuple[List[EmbeddingResult], List[int]]:
        """Generate embeddings for a list of texts"""
        results = []
        valid_indices = []
        
        for i, (text, metadata) in enumerate(zip(texts, metadata_list or [None] * len(texts))):
            try:
                vector = self.generate(text)
                results.append(EmbeddingResult(
                    vector=vector,
                    text=text,
                    metadata=metadata
                ))
                valid_indices.append(i)
            except Exception as e:
                logger.error(f"Failed to generate embedding for text {i}: {str(e)}")
                continue
                
        return results, valid_indices

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for the current model"""
        # Test embedding to get dimension
        test_text = "This is a test."
        vector = self.generate(test_text)
        return len(vector) 