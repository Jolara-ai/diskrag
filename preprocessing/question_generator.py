import json
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os

# 載入環境變數
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # 如果沒有安裝 python-dotenv，嘗試手動載入 .env 文件
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

logger = logging.getLogger(__name__)

@dataclass
class GeneratedQuestion:
    """A generated question with its source chunk"""
    question: str
    chunk_id: int
    chunk_text: str
    source_type: str
    source_id: str
    metadata: Dict[str, Any]

class QuestionGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.provider = config.get("provider", "openai")
        self.model = config.get("model", "gpt-3.5-turbo")
        self.max_questions = config.get("max_questions", 5)  # 每個問答對生成的最大問題數
        self.temperature = config.get("temperature", 0.7)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 2)
        
        # 初始化客戶端
        if self.provider == "openai":
            # 檢查 OpenAI API Key
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY 環境變數未設置。請：\n"
                    "1. 在 .env 文件中設置 OPENAI_API_KEY=your-api-key\n"
                    "2. 或設置環境變數：export OPENAI_API_KEY=your-api-key"
                )
            self.client = OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _get_completion_with_retry(self, prompt: str) -> Optional[str]:
        """使用重試機制獲取 LLM 回應"""
        for attempt in range(self.max_retries):
            try:
                if self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=1000
                    )
                    return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"獲取 LLM 回應失敗 (重試{attempt + 1}/{self.max_retries}): {str(e)}")
                    return None
                logger.warning(f"獲取 LLM 回應失敗，{self.retry_delay}秒後重試: {str(e)}")
                time.sleep(self.retry_delay)
        return None

    def generate_similar_questions(self,
                                 original_question: str,
                                 answer: str,
                                 source_type: str,
                                 source_id: str,
                                 metadata: Dict[str, Any],
                                 provider: str = "openai",
                                 project_id: Optional[str] = None) -> List[GeneratedQuestion]:
        """基於原始問題和答案生成多個相似問題"""
        if not self.enabled:
            return []

        # 構建提示詞
        prompt = f"""請基於以下問答對，生成 {self.max_questions} 個語義相似但表達方式不同的問題。
要求：
1. 生成的問題必須與原始問題表達相同的意圖
2. 使用不同的表達方式、詞彙和句式
3. 保持問題的清晰度和可理解性
4. 考慮用戶可能使用的不同問法
5. 每個問題都應該能通過原始答案得到解答

原始問題：{original_question}
原始答案：{answer}

請以 JSON 格式返回生成的問題列表，格式如下：
{{
    "questions": [
        "問題1",
        "問題2",
        ...
    ]
}}

只返回 JSON 格式的內容，不要包含其他文字。"""

        # 獲取 LLM 回應
        response = self._get_completion_with_retry(prompt)
        if not response:
            return []

        try:
            # 解析 JSON 回應
            result = json.loads(response)
            questions = result.get("questions", [])
            
            # 確保問題數量不超過限制
            questions = questions[:self.max_questions]
            
            # 過濾掉與原始問題完全相同或過於相似的問題
            filtered_questions = []
            for q in questions:
                # 移除多餘的空白字符
                q = " ".join(q.split())
                # 跳過空問題或與原始問題完全相同的情況
                if not q or q == original_question:
                    continue
                filtered_questions.append(q)
            
            # 生成問題對象
            return [
                GeneratedQuestion(
                    question=q,
                    chunk_id=source_id,  # 使用 source_id 作為 chunk_id
                    chunk_text=f"問題：{original_question}\n答案：{answer}",
                    source_type=source_type,
                    source_id=source_id,
                    metadata={
                        **metadata,
                        "original_question": original_question,
                        "answer": answer,
                        "is_generated": True
                    }
                )
                for q in filtered_questions
            ]
            
        except json.JSONDecodeError as e:
            logger.error(f"解析 LLM 回應失敗: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"生成相似問題時出錯: {str(e)}")
            return []

    def generate_questions(self,
                          chunk_id: str,
                          chunk_text: str,
                          source_type: str,
                          source_id: str,
                          metadata: Dict[str, Any],
                          provider: str = "openai",
                          project_id: Optional[str] = None) -> List[GeneratedQuestion]:
        """為文章塊生成問題（保留此方法以向後兼容）"""
        if not self.enabled or source_type != "article":
            return []
            
        # 構建提示詞
        prompt = f"""請基於以下文章內容，生成 {self.max_questions} 個相關的問題。
要求：
1. 問題應該針對文章中的重要信息
2. 問題應該清晰且容易理解
3. 問題應該能夠通過文章內容得到解答
4. 避免生成過於籠統或與文章無關的問題

文章內容：
{chunk_text}

請以 JSON 格式返回生成的問題列表，格式如下：
{{
    "questions": [
        "問題1",
        "問題2",
        ...
    ]
}}

只返回 JSON 格式的內容，不要包含其他文字。"""

        # 獲取 LLM 回應
        response = self._get_completion_with_retry(prompt)
        if not response:
            return []

        try:
            # 解析 JSON 回應
            result = json.loads(response)
            questions = result.get("questions", [])
            
            # 確保問題數量不超過限制
            questions = questions[:self.max_questions]
            
            # 生成問題對象
            return [
                GeneratedQuestion(
                    question=q,
                    metadata={
                        **metadata,
                        "chunk_id": chunk_id,
                        "chunk_text": chunk_text
                    }
                )
                for q in questions
            ]
            
        except json.JSONDecodeError as e:
            logger.error(f"解析 LLM 回應失敗: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"生成問題時出錯: {str(e)}")
            return [] 