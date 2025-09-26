# FAQ 工作流程與格式設計

## 概述

本工作流程旨在標準化FAQ處理，通過「生成多個問法 -> 向量化問題 -> 透過問題找答案和出處」的方式極大提高搜尋的召回率。

## Phase 1: 數據準備 (工程師的工作)

### CSV 檔案格式

工程師只需要填寫一個簡單、直觀的 CSV 檔案：

```csv
id,question,answer,source_file,source_page,source_section,source_image
faq_001,這份使用手冊適用於哪個型號的洗碗機？,適用於 EBF7531SBA 型號的全嵌式洗碗機。,EBF7531SBA_ZH_Manual.pdf,1,封面,images/cover.png
faq_002,如何購買原裝配件？,應訪問 https://.../accessories/,EBF7531SBA_ZH_Manual.pdf,2,內容,
faq_003,8歲以下的青少年可以使用嗎？,不可以，未滿 8 歲的青少年不得使用本機。,EBF7531SBA_ZH_Manual.pdf,3,安全資訊,
```

### 欄位說明

- **id** (可選但建議): 每個問答對的唯一標識符。如果留空，系統可以自動生成。
- **question** (必需): 標準、最典型的問題。
- **answer** (必需): 對應的答案。
- **source_file** (可選): 來源檔案名稱，如 `EBF7531SBA_ZH_Manual.pdf`。
- **source_page** (可選): 來源頁碼。
- **source_section** (可選): 來源章節標題，如「安全資訊」。
- **source_image** (可選): 相關圖片的路徑或 URL。

## Phase 2: 處理與索引

### 命令執行

```bash
# 處理FAQ CSV文件
diskrag process faq_data.csv --collection my_manual --questions

# 建立索引
diskrag index --collection my_manual
```

### 內部處理流程

1. **讀取 CSV**: 解析 `faq_data.csv`
2. **生成相似問題**: 對於每個原始問題，系統會生成多個相似問題：
   - 原始問題：`這份使用手冊適用於哪個型號的洗碗機？`
   - 生成問題：
     - `這本說明書對應哪款洗碗機？`
     - `EBF7531SBA 洗碗機的手冊是哪一份？`
     - `洗碗機 EBF7531SBA 的使用指南`

3. **建立向量與元數據**: 所有問題的向量都共享同一份元數據，包含完整的答案和出處。

### metadata.parquet 結構

| text (被向量化的內容) | text_hash | vector_index | metadata (JSON) |
|----------------------|-----------|--------------|-----------------|
| `這份手冊適用哪個型號？` | hash1 | 0 | `{"qa_id": "faq_001", "is_generated": false, "answer": "適用於 EBF7531SBA...", "source_file": "...", ...}` |
| `這本說明書對應哪款洗碗機？` | hash2 | 1 | `{"qa_id": "faq_001", "is_generated": true, "answer": "適用於 EBF7531SBA...", "source_file": "...", ...}` |
| `EBF7531SBA 手冊是哪份？` | hash3 | 2 | `{"qa_id": "faq_001", "is_generated": true, "answer": "適用於 EBF7531SBA...", "source_file": "...", ...}` |

## Phase 3: 查詢與呈現

### 查詢流程

1. **用戶查詢**: `"EBF7531SBA 這台機器怎麼用？"`
2. **向量化查詢**: 將用戶查詢轉換為向量
3. **DiskANN 搜尋**: 在索引中找到 top-k 個最相似的問題向量
4. **結果處理與去重**: 根據 `qa_id` 去重，只保留相似度最高的結果
5. **呈現結果**: 展示去重後的答案列表

### API 回應範例

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
    },
    {
      "answer": "在基本設定中，選擇「水質硬度」選項，並從等級 1 到 10 中選擇...",
      "matched_question": "如何設定水質硬度？",
      "original_question": "如何設定水質硬度？",
      "similarity": 0.85,
      "source": {
        "file": "EBF7531SBA_ZH_Manual.pdf",
        "page": 15,
        "section": "基本設定",
        "image": "images/water_hardness.png"
      },
      "metadata": {
        "qa_id": "faq_004",
        "is_generated": false,
        "vector_index": 10
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
    "nodes_visited": 45,
    "k": 5,
    "L_search": 20,
    "total_results_before_dedup": 15,
    "total_results_after_dedup": 5,
    "duplicates_removed": 10
  }
}
```

## 使用範例

### 1. 準備FAQ數據

創建 `faq_data.csv` 文件：

```csv
id,question,answer,source_file,source_page,source_section,source_image
faq_001,這份使用手冊適用於哪個型號的洗碗機？,適用於 EBF7531SBA 型號的全嵌式洗碗機。,EBF7531SBA_ZH_Manual.pdf,1,封面,images/cover.png
faq_002,如何購買原裝配件？,應訪問 https://www.bosch-home.com/accessories/ 或聯繫當地授權經銷商。,EBF7531SBA_ZH_Manual.pdf,2,配件資訊,
```

### 2. 處理和索引

```bash
# 處理FAQ文件
diskrag process faq_data.csv --collection my_manual --questions

# 建立索引
diskrag index --collection my_manual
```

### 3. 搜索使用

```python
from search_engine import SearchEngineCorrect

# 創建搜索引擎
engine = SearchEngineCorrect("my_manual")

# 定義embedding函數
def embedding_fn(text):
    # 這裡應該使用實際的embedding模型
    return np.random.randn(1536).astype(np.float32)

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
    print("---")
```

## 技術實現

### 核心組件

1. **Preprocessor**: 處理CSV文件，生成相似問題
2. **QuestionGenerator**: 使用LLM生成相似問題
3. **SearchEngine**: 支持FAQ專用搜索方法
4. **CollectionManager**: 管理向量和元數據

### 關鍵特性

- ✅ **自動問題生成**: 基於原始問題生成多個相似問法
- ✅ **結果去重**: 根據 `qa_id` 自動去重
- ✅ **完整出處**: 保留所有來源信息
- ✅ **高性能**: 使用PQ加速搜索
- ✅ **線程安全**: 支持並發請求

### 性能優化

- 使用PQ加速圖搜索
- 結果去重減少重複計算
- 線程安全的統計數據
- 可選的線程安全模式

## 最佳實踐

1. **問題設計**: 確保原始問題清晰、具體
2. **答案質量**: 提供完整、準確的答案
3. **來源信息**: 盡可能提供完整的出處信息
4. **測試驗證**: 使用多種問法測試搜索效果
5. **性能監控**: 定期檢查搜索性能和準確性

## 故障排除

### 常見問題

1. **CSV格式錯誤**: 確保必要欄位存在且格式正確
2. **問題生成失敗**: 檢查LLM配置和網絡連接
3. **搜索結果為空**: 檢查索引是否正確建立
4. **性能問題**: 考慮調整搜索參數或使用PQ加速

### 調試方法

```python
# 啟用調試模式
results = engine.search_with_debug(
    query="測試問題",
    k=5,
    embedding_fn=embedding_fn,
    debug_mode=True
)
``` 