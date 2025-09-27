from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple, Any
import logging
from pathlib import Path
import uvicorn
import os
import time
import numpy as np
from openai import OpenAI
from search_engine import SearchEngine
from preprocessing.collection import CollectionManager
from preprocessing.config import CollectionInfo

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# 建立 FastAPI 應用程式
app = FastAPI(
    title="手冊搜尋 API",
    description="基於 DiskANN 的語義搜尋 API，支援多份手冊內容的檢索",
    version="1.0.0"
)

# 初始化 OpenAI 客戶端
client = OpenAI()

def get_embedding(text: str) -> Tuple[np.ndarray, float]:
    """取得文字的 embedding 向量"""
    try:
        start_time = time.time()
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        elapsed_time = time.time() - start_time
        return np.array(response.data[0].embedding, dtype=np.float32), elapsed_time
    except Exception as e:
        logger.error(f"取得 embedding 時發生錯誤: {str(e)}")
        raise

# 用於儲存不同 collection 的搜尋引擎實例
search_engines: Dict[str, SearchEngine] = {}

def get_search_engine(collection_name: str) -> SearchEngine:
    """取得或建立指定 collection 的搜尋引擎實例"""
    try:
        if collection_name not in search_engines:
            logger.info(f"初始化 collection '{collection_name}' 的搜尋引擎...")
            search_engines[collection_name] = SearchEngine(collection_name)
            logger.info(f"Collection '{collection_name}' 的搜尋引擎初始化完成")
        return search_engines[collection_name]
    except Exception as e:
        logger.error(f"初始化 collection '{collection_name}' 的搜尋引擎時發生錯誤: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' 不存在或無法載入: {str(e)}"
        )

class SearchRequest(BaseModel):
    collection: str = Field(..., description="要搜尋的 collection 名稱")
    query: str = Field(..., min_length=1, max_length=500, description="搜尋查詢文字")
    top_k: int = Field(5, ge=1, le=20, description="回傳結果數量")
    use_faq_search: bool = Field(False, description="是否使用FAQ搜索模式（自動去重和格式化）")

class SearchResponse(BaseModel):
    """搜尋回應模型"""
    results: List[Dict[str, Any]] = Field(..., description="搜尋結果列表")
    timing: Dict[str, float] = Field(..., description="時間統計資訊", example={
        "embedding_time": 0.5,
        "search_time": 0.1,
        "total_time": 0.6
    })
    stats: Dict[str, Any] = Field(..., description="搜索統計信息")

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """執行語義搜尋"""
    try:
        engine = get_search_engine(request.collection)
        
        # 根據是否使用FAQ搜索選擇不同的搜索方法
        if request.use_faq_search:
            # 使用FAQ搜索（自動去重和格式化）
            results = engine.faq_search(
                query=request.query,
                k=request.top_k,
                beam_width=8,
                embedding_fn=lambda text: get_embedding(text)[0]  # 只回傳向量，不回傳時間
            )
        else:
            # 使用普通搜索
            results = engine.search(
                query=request.query,
                k=request.top_k,
                beam_width=8,
                embedding_fn=lambda text: get_embedding(text)[0]  # 只回傳向量，不回傳時間
            )
        
        return results
    except Exception as e:
        logger.error(f"搜尋時發生錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/faq-search", response_model=SearchResponse)
async def faq_search(request: SearchRequest):
    """執行FAQ專用搜尋（自動去重和格式化）"""
    try:
        engine = get_search_engine(request.collection)
        
        # 執行FAQ搜索
        results = engine.faq_search(
            query=request.query,
            k=request.top_k,
            beam_width=8,
            embedding_fn=lambda text: get_embedding(text)[0]  # 只回傳向量，不回傳時間
        )
        
        return results
    except Exception as e:
        logger.error(f"FAQ搜尋時發生錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
async def list_collections():
    """列出所有可用的 collections"""
    try:
        manager = CollectionManager()
        collections = manager.list_collections()
        
        if not collections:
            logger.info("沒有找到任何 collections")
            return []
        
        result = []
        for collection in collections:
            try:
                # 檢查必要檔案
                collection_dir = manager._get_collection_dir(collection.name)
                index_dir = manager.get_index_dir(collection.name)
                
                # 檢查索引目錄是否存在
                if not index_dir.exists():
                    logger.warning(f"索引目錄不存在: {index_dir}")
                    collection_info = {
                        "name": collection.name,
                        "status": "no_index",
                        "missing_files": ["index directory"]
                    }
                    result.append(collection_info)
                    continue
                
                # 檢查 meta.json 文件
                meta_path = index_dir / "meta.json"
                if not meta_path.exists():
                    logger.warning(f"meta.json 不存在: {meta_path}")
                    collection_info = {
                        "name": collection.name,
                        "status": "incomplete",
                        "missing_files": ["index/meta.json"]
                    }
                    result.append(collection_info)
                    continue
                
                # 載入 meta.json 來檢查是否使用 PQ
                try:
                    import json
                    with open(meta_path, 'r') as f:
                        meta_data = json.load(f)
                    use_pq = meta_data.get("use_pq", True)
                except Exception as e:
                    logger.warning(f"無法讀取 meta.json: {e}")
                    use_pq = True  # 默認使用 PQ
                
                # 定義所有需要檢查的檔案
                required_files = {
                    # 向量和元資料檔案
                    "vectors": manager.get_vectors_path(collection.name),
                    "metadata": manager.get_metadata_path(collection.name),
                    
                    # 索引相關檔案
                    "index": {
                        "index.dat": index_dir / "index.dat",
                        "meta.json": index_dir / "meta.json"
                    }
                }
                
                # 如果使用 PQ，則需要檢查 PQ 文件
                if use_pq:
                    required_files["index"]["pq_model.pkl"] = index_dir / "pq_model.pkl"
                    required_files["index"]["pq_codes.bin"] = index_dir / "pq_codes.bin"
                
                # 詳細檢查每個檔案
                file_status = {}
                index_files_status = {}
                
                # 檢查向量和元資料檔案
                for name, path in {k: v for k, v in required_files.items() if k != "index"}.items():
                    exists = path.exists()
                    file_status[name] = {
                        "exists": exists,
                        "path": str(path.absolute())
                    }
                    if exists:
                        try:
                            size = path.stat().st_size
                            file_status[name]["size"] = f"{size / 1024:.1f}KB"
                            logger.info(f"檔案存在: {path.name}, 大小: {size / 1024:.1f}KB")
                        except Exception as e:
                            logger.error(f"取得檔案大小時發生錯誤 {path.name}: {str(e)}")
                            file_status[name]["error"] = str(e)
                    else:
                        logger.warning(f"檔案不存在: {path.name}")
                
                # 檢查索引檔案
                for name, path in required_files["index"].items():
                    exists = path.exists()
                    index_files_status[name] = {
                        "exists": exists,
                        "path": str(path.absolute())
                    }
                    if exists:
                        try:
                            size = path.stat().st_size
                            index_files_status[name]["size"] = f"{size / 1024:.1f}KB"
                            logger.info(f"索引檔案存在: {path.name}, 大小: {size / 1024:.1f}KB")
                        except Exception as e:
                            logger.error(f"取得索引檔案大小時發生錯誤 {path.name}: {str(e)}")
                            index_files_status[name]["error"] = str(e)
                    else:
                        logger.warning(f"索引檔案不存在: {path.name}")
                
                # 合併檔案狀態
                file_status["index"] = index_files_status
                
                # 檢查所有必要檔案是否存在
                has_required_files = (
                    all(status["exists"] for status in file_status.values() if isinstance(status, dict) and "exists" in status) and
                    all(status["exists"] for status in index_files_status.values())
                )
                
                # 收集缺少的檔案
                missing_files = []
                for name, status in file_status.items():
                    if isinstance(status, dict) and "exists" in status and not status["exists"]:
                        missing_files.append(name)
                for name, status in index_files_status.items():
                    if not status["exists"]:
                        missing_files.append(f"index/{name}")
                
                collection_info = {
                    "name": collection.name,
                    "status": "ready" if has_required_files else "incomplete",
                    "file_status": file_status,
                    "use_pq": use_pq,
                    "stats": {
                        "total_chunks": collection.chunk_stats.get("total_chunks", 0),
                        "total_questions": collection.chunk_stats.get("total_questions", 0),
                        "last_processed_files": collection.chunk_stats.get("last_processed_files", []),
                        "index_built_at": collection.chunk_stats.get("index_built_at")
                    }
                }
                
                if not has_required_files:
                    collection_info["missing_files"] = missing_files
                    logger.warning(f"Collection '{collection.name}' 缺少檔案: {', '.join(missing_files)}")
                else:
                    logger.info(f"Collection '{collection.name}' 狀態完整 (使用 PQ: {use_pq})")
                
                result.append(collection_info)
                
            except Exception as e:
                logger.error(f"檢查 collection '{collection.name}' 時發生錯誤: {str(e)}", exc_info=True)
                result.append({
                    "name": collection.name,
                    "status": "error",
                    "error": str(e)
                })
        
        logger.info(f"共發現 {len(result)} 個 collections")
        return result
        
    except Exception as e:
        logger.error(f"列出 collections 時發生錯誤: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康檢查介面，回傳系統關鍵元件的狀態"""
    try:
        manager = CollectionManager()
        collections_dir = manager.base_dir
        collections_exist = collections_dir.exists()
        logger.info(f"Collections 目錄狀態: {'存在' if collections_exist else '不存在'}")
        
        if not collections_exist:
            logger.warning(f"Collections 目錄不存在: {collections_dir.absolute()}")
        
        # 檢查目錄權限
        collections_writable = False
        if collections_exist:
            try:
                test_file = collections_dir / ".test_write"
                test_file.touch()
                test_file.unlink()
                collections_writable = True
                logger.info("Collections 目錄可寫入")
            except Exception as e:
                logger.error(f"Collections 目錄權限檢查失敗: {str(e)}", exc_info=True)
        
        # 檢查目錄內容
        if collections_exist:
            try:
                dir_contents = list(collections_dir.iterdir())
                logger.info(f"Collections 目錄內容: {[p.name for p in dir_contents]}")
            except Exception as e:
                logger.error(f"讀取 collections 目錄內容時發生錯誤: {str(e)}", exc_info=True)
        
        # 檢查環境變數
        env_vars = {
            "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
            "GOOGLE_CLOUD_PROJECT": bool(os.getenv("GOOGLE_CLOUD_PROJECT")),
            "GOOGLE_APPLICATION_CREDENTIALS": bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
        }
        logger.info(f"環境變數狀態: {env_vars}")
        
        status = "healthy" if collections_exist and collections_writable else "degraded"
        logger.info(f"系統狀態: {status}")
        
        return {
            "status": status,
            "timestamp": time.time(),
            "collections": {
                "directory_exists": collections_exist,
                "directory_writable": collections_writable,
                "path": str(collections_dir.absolute())
            },
            "environment": env_vars
        }
    except Exception as e:
        logger.error(f"健康檢查時發生錯誤: {str(e)}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }

class AskRequest(BaseModel):
    collection: str = Field(..., description="要搜尋的 collection 名稱")
    question: str = Field(..., min_length=1, max_length=500, description="使用者問題")
    top_k: int = Field(2, ge=1, le=5, description="搜尋結果數量")

class AskResponse(BaseModel):
    """問答回應模型"""
    answer: str = Field(..., description="LLM 生成的回答")
    timing: Dict[str, float] = Field(..., description="時間統計資訊")

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """使用 LLM 處理搜尋結果並生成回答"""
    try:
        total_start_time = time.time()
        logger.info(f"收到問答請求: collection={request.collection}, question={request.question}")
        
        # 1. 執行搜尋
        engine = get_search_engine(request.collection)
        logger.info(f"開始搜尋 collection '{request.collection}'")
        
        # 1.1 取得 embedding
        embedding_start_time = time.time()
        embedding, embedding_time = get_embedding(request.question)
        logger.info(f"Embedding 完成，耗時: {embedding_time:.2f}秒")
        
        # 1.2 執行 DiskANN 搜尋
        diskann_start_time = time.time()
        results = engine.search(
            query=request.question,
            k=request.top_k,
            beam_width=8,
            embedding_fn=lambda _: embedding  # 直接使用已計算的 embedding
        )
        diskann_time = time.time() - diskann_start_time
        logger.info(f"DiskANN 搜尋完成，耗時: {diskann_time:.2f}秒")
        
        search_time = time.time() - total_start_time
        logger.info(f"搜尋總耗時: {search_time:.2f}秒")
        logger.debug(f"搜尋結果: {results}")
        
        # 2. 準備提示詞
        if not results.get("results"):
            logger.warning("搜尋結果為空")
            return {
                "answer": "抱歉，我找不到相關的資訊來回答這個問題。",
                "timing": {
                    "embedding_time": embedding_time,
                    "diskann_time": diskann_time,
                    "search_time": search_time,
                    "llm_time": 0,
                    "total_time": search_time
                }
            }
        
        # 準備搜尋結果文本，使用 FAQ 格式
        try:
            context_parts = []
            for i, result in enumerate(results["results"], 1):
                metadata = result.get("metadata", {})
                if metadata.get("source_type") == "faq":
                    # 使用 FAQ 的結構
                    question = metadata.get("question", "")
                    answer = metadata.get("answer", "")
                    if question and answer:
                        context_parts.append(f"FAQ {i}:\n問題：{question}\n答案：{answer}")
                else:
                    # 如果不是 FAQ 格式，使用原始文本
                    text = result.get("text", "")
                    if text:
                        context_parts.append(f"來源 {i}:\n{text}")
            
            context = "\n\n".join(context_parts)
            logger.info(f"準備了 {len(results['results'])} 個搜尋結果作為上下文")
            logger.debug(f"上下文內容: {context}")
        except Exception as e:
            logger.error(f"處理搜尋結果時發生錯誤: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"處理搜尋結果時發生錯誤: {str(e)}")
        
        # 3. 使用 LLM 生成回答
        llm_start_time = time.time()
        prompt = f"""你是一個專業的客服助手，請根據以下參考資料回答使用者的問題。
如果參考資料不足以回答問題，或問題與參考資料無關，請直接回答「抱歉，我無法根據現有資料回答這個問題」。

參考資料：
{context}

使用者問題：{request.question}

請注意：
1. 如果參考資料是 FAQ 格式，請特別注意問題和答案的對應關係
2. 回答時要簡潔明確，直接給出解決方案
3. 如果有多個相關答案，請整合成一個完整的回答
4. 不需要包含「根據參考資料」等開場白
5. 如果參考資料不足以回答問題，請直接說不知道"""

        try:
            logger.info("開始呼叫 LLM 生成回答")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "你是一個專業的客服助手，根據提供的 FAQ 資料回答問題。回答要簡潔明確，直接給出解決方案。如果資料不足以回答，請直接說不知道。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            answer = response.choices[0].message.content.strip()
            logger.info("LLM 回答生成完成")
            logger.debug(f"LLM 原始回答: {answer}")
            
            # 如果回答包含「無法回答」或「不知道」等關鍵詞，統一回答
            if any(keyword in answer.lower() for keyword in ["無法回答", "不知道", "沒有相關資訊", "找不到"]):
                answer = "抱歉，我無法根據現有資料回答這個問題。"
                logger.info("回答被判定為無法回答")
                
        except Exception as e:
            logger.error(f"LLM 生成回答時發生錯誤: {str(e)}", exc_info=True)
            answer = "抱歉，系統處理您的問題時發生錯誤。"
        
        llm_time = time.time() - llm_start_time
        logger.info(f"LLM 處理耗時: {llm_time:.2f}秒")
        
        # 4. 準備回應
        return {
            "answer": answer,
            "timing": {
                "embedding_time": embedding_time,
                "diskann_time": diskann_time,
                "search_time": search_time,
                "llm_time": llm_time,
                "total_time": time.time() - total_start_time
            }
        }
        
    except Exception as e:
        logger.error(f"處理問答請求時發生錯誤: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 