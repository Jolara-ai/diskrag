# DiskRAG

DiskRAG 是一個基於 DiskANN 的向量檢索系統，用於處理和搜尋文件。

## 功能特點

- 支援多種文件格式：
  - Markdown (.md) 手冊檔案
  - FAQ 格式的 CSV 檔案
  - 文章格式的 CSV 檔案
- 自動文字分塊和向量化
- 基於 DiskANN 的高效向量搜尋
- 支援多個 collection 管理
- 自動問題產生
- 完整的命令列工具
- FastAPI 介面

## 安裝

```bash
# 複製儲存庫
git clone https://github.com/joho-ai/diskrag.git
cd diskrag

# 安裝相依套件
poetry install
```

## 設定

1. 建立設定檔：

```bash
# 為每個 collection 建立獨立的設定檔
poetry run python -m preprocessing.cli create-config --output faq_config.yaml
poetry run python -m preprocessing.cli create-config --output manual_config.yaml
```

2. 編輯設定檔，設定：
   - Collection 名稱（每個設定檔對應一個 collection）
   - Embedding 提供者和模型
   - 問題產生參數（FAQ 專用）
   - 文字分塊參數
   - 索引參數

範例：FAQ Collection 設定（`faq_config.yaml`）：
```yaml
collection: "faq_collection"  # FAQ 專用的 collection 名稱

embedding:
  provider: "openai"  # 或 "vertexai"
  model: "text-embedding-3-small"  # 或其他支援的模型
  max_retries: 3
  retry_delay: 2

question_generation:
  enabled: true  # FAQ 建議啟用問題產生
  provider: "openai"
  model: "gpt-4o-mini"  # 根據需求選擇
  max_questions: 5  # 每個原始問題產生的最大問題數
  temperature: 0.7
  max_retries: 3
  retry_delay: 2

chunk:
  size: 300  # 最大文字長度
  overlap: 50  # 重疊長度
  min_size: 50  # 最小文字長度

output:
  format: "parquet"
  compression: "snappy"
```

範例：手冊 Collection 設定（`manual_config.yaml`）：
```yaml
collection: "manual_collection"  # 手冊專用的 collection 名稱

embedding:
  provider: "openai"
  model: "text-embedding-3-small"
  max_retries: 3
  retry_delay: 2

chunk:
  size: 500  # 手冊檔案可以使用較大的分塊大小
  overlap: 100
  min_size: 100

output:
  format: "parquet"
  compression: "snappy"
```

> **設定說明**：
> 1. 每個 collection 需要獨立的設定檔
> 2. FAQ collection 建議啟用問題產生功能
> 3. 手冊 collection 不需要問題產生設定
> 4. 可以根據不同 collection 的需求調整分塊參數
> 5. 設定檔名稱建議與 collection 用途相關（如 `faq_config.yaml`、`manual_config.yaml`）

## 檔案格式

### FAQ CSV 格式

FAQ 格式的 CSV 檔案必須包含以下欄位：

- 必要欄位：
  - `question`: 問題文字
  - `answer`: 答案文字
- 選用欄位：
  - `note`: 備註資訊

範例：
```csv
question,answer,note
如何使用系統？,請參考使用者手冊第1章。,適用於新使用者
系統支援哪些格式？,支援 Markdown 和 Word 格式。,更新於2024年
```

### 文章 CSV 格式

文章格式的 CSV 檔案必須包含以下欄位：

- 必要欄位：
  - `title`: 文章標題
  - `paragraph_text`: 段落文字
- 選用欄位：
  - `section`: 章節資訊

範例：
```csv
title,paragraph_text,section
系統概述,這是一個基於 DiskANN 的向量檢索系統。,第一章
安裝指南,請按照以下步驟安裝系統...,第二章
```

## 使用

### 1. 處理文件

#### FAQ CSV 檔案處理

FAQ 檔案處理分為兩個階段：

1. **試執行模式**（產生問題）：
```bash
# 產生問題並儲存到 *_post.csv 檔案
poetry run python -m preprocessing.cli process --type csv --input ./docs/faq.csv --config ./faq_config.yaml --dry-run
```

2. **正式處理**（建立向量和索引）：
```bash
# 使用產生問題後的檔案建立向量和索引
poetry run python -m preprocessing.cli process --type csv --input ./docs/faq_post.csv --config ./faq_config.yaml
```

> **注意**：FAQ 處理必須先使用 `--dry-run` 模式產生問題，然後使用產生的 `*_post.csv` 檔案進行正式處理。FAQ 的筆數建議至少要有50筆資料。

#### 手冊檔案處理

```bash
# 處理 Markdown 檔案
poetry run python -m preprocessing.cli process --type md --input path/to/manual.md --config config.yaml
```

> **TODO**: 未來版本將支援：
> - Word (.docx) 檔案處理
> - 更多文件格式的支援

### 2. 管理 Collections

列出所有 collections：

```bash
poetry run python -m preprocessing.cli list
```

顯示 collection 詳細資訊：

```bash
poetry run python -m preprocessing.cli show --name your_collection_name
```

刪除 collection：

```bash
poetry run python -m preprocessing.cli delete --name your_collection_name
```

> **更新 Collection 的步驟**：
> 1. 先刪除現有的 collection：
>    ```bash
>    poetry run python -m preprocessing.cli delete --name your_collection_name
>    ```
> 2. 使用新的設定檔重新建立 collection：
>    ```bash
>    # 處理 FAQ 檔案
>    poetry run python -m preprocessing.cli process --type csv --input ./docs/faq_post.csv --config ./configs/faq_config.yaml
>    
>    # 處理手冊檔案
>    poetry run python -m preprocessing.cli process --type md --input ./docs/manual.md --config ./configs/manual_config.yaml
>    ```
> 3. 重新建立索引：
>    ```bash
>    poetry run python -m preprocessing.cli build-index your_collection_name --config ./configs/your_config.yaml
>    ```

> **TODO**: 未來版本將支援：
> - 一鍵重建 collection（rebuild 命令）
> - 增量更新 collection
> - 自動備份和還原功能
> - 批次處理多個檔案
> - 設定管理

### 3. 建立索引

使用預設參數：

```bash
poetry run python -m preprocessing.cli build-index your_collection_name
```

自訂參數：

```bash
poetry run python -m preprocessing.cli build-index your_collection_name --R 64 --threads 4
```

索引參數說明：
- `R`：圖的度數，控制每個節點的鄰居數量（預設：32）
- `threads`：使用的執行緒數（預設：1）

### 4. 搜尋

使用命令列搜尋：

```bash
poetry run python search_cli.py your_collection_name
```

使用 FastAPI 服務：

```bash
# 啟動服務
poetry run uvicorn app:app --reload

# 發送搜尋請求
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"collection": "your_collection_name", "query": "你的問題"}'

# 問答請求（使用 LLM 生成回答）
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "collection": "your_collection_name",
       "question": "你的問題",
       "top_k": 2
     }'
```

API 端點說明：

1. `/health` (GET)
   - 用途：檢查系統健康狀態
   - 回傳：系統狀態、環境變數狀態等資訊

2. `/search` (POST)
   - 用途：執行語義搜尋
   - 請求格式：
     ```json
     {
       "collection": "collection 名稱",
       "query": "搜尋查詢文字",
       "top_k": 5  // 選用，預設為 5，範圍 1-20
     }
     ```
   - 回傳：搜尋結果列表和時間統計

3. `/ask` (POST)
   - 用途：使用 LLM 處理搜尋結果並生成回答
   - 特點：
     - 自動搜尋相關內容
     - 使用 GPT-4o-mini 模型生成回答
     - 特別優化處理 FAQ 格式的資料
     - 如果找不到相關資訊，會明確表示無法回答
   - 請求格式：
     ```json
     {
       "collection": "collection 名稱",
       "question": "使用者問題",
       "top_k": 2  // 選用，預設為 2，範圍 1-5
     }
     ```
   - 回傳格式：
     ```json
     {
       "answer": "LLM 生成的回答",
       "timing": {
         "search_time": 1.38,  // 搜尋耗時（秒）
         "llm_time": 0.96,     // LLM 處理耗時（秒）
         "total_time": 2.34    // 總耗時（秒）
       }
     }
     ```
   - 注意事項：
     - 回答會根據搜尋結果的相關性自動調整
     - 如果搜尋結果為空或不足以回答問題，會回傳「抱歉，我無法根據現有資料回答這個問題」
     - 回答會特別注意 FAQ 格式的資料，優先使用結構化的問題和答案
     - 回答會保持簡潔明確，直接給出解決方案

4. `/collections` (GET)
   - 用途：列出所有可用的 collections
   - 回傳：collection 列表，包含狀態和檔案資訊

## Docker 使用說明

> **重要提醒**：使用 Docker 前，請確保已經完成以下步驟：
> 1. 已經建立並處理好 collection（使用上述 Python 命令）
> 2. 已經建立索引（使用 `build-index` 命令）
> 3. 已經設定好環境變數（OpenAI API Key 等）

### 1. 環境準備

1. 確認必要的目錄結構：
```bash
# 檢查目錄是否存在
ls -la collections/  # 存放向量和索引的目錄
ls -la configs/      # 存放設定檔的目錄
```

2. 建立 `.env` 檔案：
```bash
# OpenAI API 設定
OPENAI_API_KEY=your-api-key-here

# 服務連接埠（選用）
PORT=8000
```

### 2. 啟動 API 服務

```bash
# 建立並啟動 API 服務（背景執行）
docker-compose build api
docker-compose up -d api

# 檢查服務狀態
docker-compose ps

# 查看服務日誌
docker-compose logs -f api
```

### 3. 使用 API

服務啟動後，可以透過 HTTP 請求使用搜尋功能：

```bash
# 健康檢查
curl http://localhost:8000/health

# 搜尋請求
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"collection": "your_collection_name", "query": "你的問題"}'

# 問答請求（使用 LLM 生成回答）
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "collection": "your_collection_name",
       "question": "你的問題",
       "top_k": 2
     }'
```

### 4. 停止服務

```bash
# 停止 API 服務
docker-compose down
```

## TODO

### 功能增強
- [ ] Word (.docx) 檔案處理
- [ ] 更多文件格式的支援
- [ ] 一鍵重建 collection（rebuild 命令）
- [ ] 增量更新 collection
- [ ] 自動備份和還原功能
- [ ] 批次處理多個檔案
- [ ] 設定管理

### Docker 相關
- [ ] 提供更多 Docker 環境下的故障排除指南
- [ ] 說明如何在不同環境中部署 API 服務
- [ ] 提供 Docker 環境下的效能優化建議
- [ ] 說明如何設定 Docker 資源限制
- [ ] 提供 Docker 環境下的監控方案

### 文件處理
- [ ] FAQ 檔案處理流程說明
- [ ] 手冊檔案處理流程說明
- [ ] 檔案格式驗證工具
- [ ] 批次處理工具

### Collection 管理
- [ ] Collection 建立流程說明
- [ ] Collection 更新流程說明
- [ ] Collection 備份和還原說明
- [ ] Collection 狀態檢查工具

## 專案結構

```
diskrag/
├── app.py              # FastAPI 介面
├── assets/            # 靜態資源
├── build_index.py     # 索引建立核心
├── build_index_cli.py # 索引建立命令列介面
├── collections/       # Collection 資料目錄
├── configs/          # 設定檔目錄
├── data/             # 原始資料目錄
│   └── manual/      # 手冊資料
├── docs/            # 文件目錄
├── logs/           # 日誌目錄
├── preprocessing/  # 預處理模組
│   ├── __init__.py
│   ├── cli.py              # 命令列介面
│   ├── collection.py       # 集合管理
│   ├── config.py          # 設定類別
│   ├── embedding.py       # 向量產生
│   ├── processor.py       # FAQ 處理器
│   └── question_generator.py  # 問題產生
├── pydiskann/     # DiskANN 實作
├── scripts/       # 工具腳本
├── search_cli.py  # 命令列搜尋介面
├── search_engine.py # 搜尋核心
├── tests/         # 測試目錄
├── verify_data.py # 資料驗證工具
├── Dockerfile    # Docker 設定
├── poetry.lock   # 相依套件版本鎖定
└── pyproject.toml # 專案設定

```

## 開發說明

- 使用 `black` 和 `isort` 進行程式碼格式化
- 使用 `mypy` 進行型別檢查
- 使用 `pytest` 進行單元測試
- 所有設定參數都應透過 `config.yaml` 檔案設定
- 使用 `CollectionManager` 管理集合的建立、更新和刪除
- 使用 `Preprocessor` 處理 FAQ 檔案
- 使用 `DocumentProcessor` 處理手冊檔案
- 向量產生使用 `EmbeddingGenerator`
- 問題產生使用 `QuestionGenerator`（如果啟用）

## 注意事項

- 確保 OpenAI API Key 已正確設定
- 文件處理支援斷點續傳
- 搜尋查詢限制在 500 字元以內
- 建議定期備份向量資料和索引
- 每個 collection 可以包含不同的文件集合
- collection 名稱在建立後不能修改
- 可以同時維護多個 collection 用於不同的用途
- 優先使用 `config.yaml` 中的設定，命令列參數可以覆蓋設定檔中的設定

## References

This project is inspired by the design and algorithm presented in the following research paper:

**DiskANN: Fast Accurate Billion-Point Nearest Neighbor Search on a Single Node**  
Microsoft Research  
🔗 https://www.microsoft.com/en-us/research/publication/diskann-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node/


## 授權條款

MIT
