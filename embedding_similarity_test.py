import os
import numpy as np
from openai import OpenAI
from numpy.linalg import norm

# 從環境變數取得設定
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text: str) -> np.ndarray:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text.strip()
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def main():
    full_text = "將生成的嵌入向量連同其對應的文字內容和元資料儲存在向量資料庫中，並建立索引以實現高效的相似性搜尋"
    partial_text = "將生成的嵌入向量連同其對應的文字內容和元資料"

    print(f"比較句子：\n1. {partial_text}\n2. {full_text}\n")

    emb1 = get_embedding(partial_text)
    emb2 = get_embedding(full_text)

    sim = cosine_similarity(emb1, emb2)
    print(f"Cosine similarity: {sim:.4f} ({sim * 100:.2f}%)")

if __name__ == "__main__":
    main()
