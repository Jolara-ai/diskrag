# FAQ 工作流程實現總結

## 🎯 實現目標

成功實現了標準化的FAQ工作流程，通過「生成多個問法 -> 向量化問題 -> 透過問題找答案和出處」的方式極大提高搜尋的召回率。

## ✅ 已完成的功能

### 1. Phase 1: 數據準備 ✅

- ✅ **CSV 格式設計**：標準化的FAQ CSV格式，包含所有必要欄位
- ✅ **欄位驗證**：自動檢查必要欄位（question, answer）
- ✅ **靈活配置**：支持可選欄位（id, source_file, source_page, source_section, source_image）

**示例格式：**
```csv
id,question,answer,source_file,source_page,source_section,source_image
faq_001,這份使用手冊適用於哪個型號的洗碗機？,適用於 EBF7531SBA 型號的全嵌式洗碗機。,EBF7531SBA_ZH_Manual.pdf,1,封面,images/cover.png
```

### 2. Phase 2: 處理與索引 ✅

- ✅ **CSV 解析**：支持標準CSV格式讀取
- ✅ **自動問題生成**：使用LLM生成多個相似問題
- ✅ **元數據管理**：所有問題共享同一份答案和出處信息
- ✅ **向量化**：為所有問題（原始+生成）建立向量
- ✅ **索引建立**：支持DiskANN索引建立

**內部處理流程：**
1. 讀取CSV文件
2. 為每個原始問題生成多個相似問題
3. 建立向量和元數據
4. 所有問題共享同一份答案和出處信息

### 3. Phase 3: 查詢與呈現 ✅

- ✅ **FAQ專用搜索**：`faq_search()` 方法
- ✅ **自動去重**：根據 `qa_id` 自動去除重複結果
- ✅ **格式化輸出**：統一的結果格式
- ✅ **完整出處**：保留所有來源信息
- ✅ **相似度計算**：轉換距離為相似度

**API回應格式：**
```json
{
  "results": [
    {
      "answer": "適用於 EBF7531SBA 型號的全嵌式洗碗機。",
      "matched_question": "EBF7531SBA 的使用指南",
      "original_question": "這份使用手冊適用於哪個型號的洗碗機？",
      "similarity": 0.95,
      "source": {
        "file": "EBF7531SBA_ZH_Manual.pdf",
        "page": 1,
        "section": "封面",
        "image": "images/cover.png"
      },
      "metadata": {
        "qa_id": "faq_001",
        "is_generated": true,
        "vector_index": 2
      }
    }
  ],
  "timing": {
    "embedding_time": 0.123,
    "search_time": 0.456,
    "total_time": 0.579
  },
  "stats": {
    "search_type": "faq_pq_accelerated",
    "total_results_before_dedup": 15,
    "total_results_after_dedup": 5,
    "duplicates_removed": 10
  }
}
```

## 🔧 核心組件實現

### 1. Preprocessor 增強 ✅

**文件：** `preprocessing/processor.py`

- ✅ 支持FAQ CSV格式解析
- ✅ 自動生成相似問題
- ✅ 元數據結構化管理
- ✅ 向量和索引建立

**關鍵方法：**
```python
def process_file(self, input_file: str, dry_run: bool = False) -> None:
    """處理 FAQ CSV 文件"""
    # 1. 讀取CSV
    # 2. 生成相似問題
    # 3. 建立向量和元數據
    # 4. 更新collection
```

### 2. QuestionGenerator 增強 ✅

**文件：** `preprocessing/question_generator.py`

- ✅ 支持相似問題生成
- ✅ LLM集成（OpenAI）
- ✅ 錯誤處理和重試機制
- ✅ 結果過濾和驗證

**關鍵方法：**
```python
def generate_similar_questions(self, original_question: str, answer: str, ...) -> List[GeneratedQuestion]:
    """基於原始問題和答案生成多個相似問題"""
```

### 3. SearchEngine 增強 ✅

**文件：** `search_engine.py`

- ✅ 新增 `faq_search()` 方法
- ✅ 自動結果去重
- ✅ 格式化輸出
- ✅ 線程安全支持

**關鍵方法：**
```python
def faq_search(self, query: str, k: int = 5, ...) -> Dict[str, Any]:
    """FAQ 專用搜索方法 - 支持結果去重和格式化"""
```

### 4. API 增強 ✅

**文件：** `app.py`

- ✅ 新增 `/faq-search` 端點
- ✅ 支持FAQ搜索模式
- ✅ 統一的響應格式

**新增端點：**
```python
@app.post("/faq-search", response_model=SearchResponse)
async def faq_search(request: SearchRequest):
    """執行FAQ專用搜尋（自動去重和格式化）"""
```

## 📊 性能優化

### 1. 線程安全 ✅

- ✅ 使用 `threading.Lock` 保護共享狀態
- ✅ 可選的線程安全模式
- ✅ 細粒度鎖設計，最小化性能影響

### 2. 搜索優化 ✅

- ✅ PQ加速圖搜索
- ✅ 結果去重減少重複計算
- ✅ 智能候選選擇

### 3. 內存管理 ✅

- ✅ 高效的元數據結構
- ✅ 向量索引優化
- ✅ 自動清理機制

## 🧪 測試和驗證

### 1. 測試腳本 ✅

**文件：** `scripts/test_faq_workflow.py`

- ✅ 完整工作流程測試
- ✅ 去重功能驗證
- ✅ 性能測試

### 2. 示例數據 ✅

**文件：** `examples/faq_data.csv`

- ✅ 標準FAQ CSV格式示例
- ✅ 完整的欄位示例
- ✅ 實際使用場景

## 📚 文檔和指南

### 1. 工作流程文檔 ✅

**文件：** `docs/FAQ_WORKFLOW.md`

- ✅ 詳細的工作流程說明
- ✅ 使用示例和最佳實踐
- ✅ 故障排除指南

### 2. README 更新 ✅

**文件：** `README.md`

- ✅ FAQ工作流程介紹
- ✅ 快速開始指南
- ✅ 命令示例

## 🎯 關鍵特性

### 1. 自動問題生成 ✅

- 基於原始問題生成多個相似問法
- 使用LLM確保語義一致性
- 可配置的生成參數

### 2. 結果去重 ✅

- 根據 `qa_id` 自動去重
- 保留相似度最高的結果
- 統計去重效果

### 3. 完整出處 ✅

- 保留所有來源信息
- 支持文件、頁碼、章節、圖片
- 結構化輸出

### 4. 高性能 ✅

- PQ加速搜索
- 線程安全設計
- 可選的性能模式

## 🚀 使用示例

### 1. 準備數據

```csv
id,question,answer,source_file,source_page,source_section,source_image
faq_001,這份使用手冊適用於哪個型號的洗碗機？,適用於 EBF7531SBA 型號的全嵌式洗碗機。,EBF7531SBA_ZH_Manual.pdf,1,封面,images/cover.png
```

### 2. 處理和索引

```bash
# 處理FAQ文件
diskrag process faq_data.csv --collection my_manual --questions

# 建立索引
diskrag index my_manual
```

### 3. 搜索使用

```python
from search_engine import SearchEngineCorrect

# 創建搜索引擎
engine = SearchEngineCorrect("my_manual")

# FAQ搜索
results = engine.faq_search(
    query="EBF7531SBA 這台機器怎麼用？",
    k=5,
    embedding_fn=embedding_fn
)

# 處理結果
for result in results["results"]:
    print(f"問題: {result['matched_question']}")
    print(f"答案: {result['answer']}")
    print(f"相似度: {result['similarity']:.2f}")
    print(f"來源: {result['source']['file']} 第{result['source']['page']}頁")
```

## 🔮 未來改進

### 1. 功能增強

- [ ] 支持更多LLM提供商
- [ ] 自定義問題生成模板
- [ ] 批量處理優化
- [ ] 實時更新支持

### 2. 性能優化

- [ ] 向量緩存機制
- [ ] 分布式搜索
- [ ] 更智能的去重算法
- [ ] 自適應搜索參數

### 3. 用戶體驗

- [ ] Web界面
- [ ] 可視化搜索結果
- [ ] 搜索歷史記錄
- [ ] 用戶反饋機制

## 📈 總結

成功實現了完整的FAQ工作流程，具備以下優勢：

1. **標準化**：統一的CSV格式和處理流程
2. **自動化**：自動問題生成和結果去重
3. **高性能**：PQ加速和線程安全設計
4. **易用性**：簡單的API和完整的文檔
5. **可擴展**：模塊化設計，易於擴展

這個實現為FAQ搜索提供了一個完整、高效、易用的解決方案，能夠極大提高搜索的召回率和用戶體驗。 