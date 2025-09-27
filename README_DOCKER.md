# DiskRAG Docker 快速使用指南

DiskRAG 是一個基於 DiskANN 的語義搜尋系統，現在支援 Docker 部署，讓您可以快速建立 FAQ 搜尋 API。

## 🚀 快速開始

### 1. 準備環境

確保您的系統已安裝：
- Docker
- Docker Compose

### 2. 設置環境變數

```bash
# 複製環境變數範例文件
cp env.example .env

# 編輯 .env 文件，填入您的 OpenAI API 金鑰
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. 準備 FAQ 數據

將您的 FAQ CSV 文件放在 `data/` 目錄下。CSV 格式應包含：
- `question`: 問題
- `answer`: 答案
- `source_file`: 來源文件（可選）
- `source_page`: 來源頁面（可選）
- `source_section`: 來源章節（可選）

範例：
```csv
id,question,answer,source_file,source_page,source_section
1,如何使用系統？,請參考使用手冊第1章,manual.pdf,1,介紹
2,如何重置密碼？,請聯繫管理員或使用忘記密碼功能,manual.pdf,5,帳戶管理
```

### 4. 處理 FAQ 數據

```bash
# 基本處理（不生成相似問題）
./scripts/docker_process_faq.sh my_collection data/faq_data.csv

# 處理並生成相似問題
./scripts/docker_process_faq.sh my_collection data/faq_data.csv --questions
```

### 5. 啟動 API 服務

```bash
# 啟動服務
./scripts/docker_start.sh

# 或重新構建並啟動
./scripts/docker_start.sh --build
```

### 6. 使用 API

API 服務啟動後，您可以：

- 訪問 API 文檔：http://localhost:8000/docs
- 健康檢查：http://localhost:8000/health
- 查看所有 collections：http://localhost:8000/collections

#### 搜尋 FAQ

```bash
curl -X POST "http://localhost:8000/faq-search" \
  -H "Content-Type: application/json" \
  -d '{
    "collection": "my_collection",
    "query": "如何使用系統？",
    "top_k": 5
  }'
```

#### 智能問答

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "collection": "my_collection",
    "question": "如何使用系統？",
    "top_k": 2
  }'
```

## 📁 目錄結構

```
your-project/
├── collections/          # FAQ 集合數據（自動生成）
│   └── my_collection/    # 每個 collection 的數據
│       ├── collection_info.json  # 集合配置信息
│       ├── metadata.parquet      # 元數據文件
│       ├── vectors.npy           # 向量文件
│       └── index/                # 索引文件
│           ├── index.dat
│           └── meta.json
├── data/                # 原始 CSV 文件
│   └── faq_data.csv     # 您的 FAQ 數據
├── logs/                # 日誌文件
├── docker-compose.yml   # Docker 配置
├── .env                 # 環境變數
└── scripts/
    ├── docker_start.sh      # 啟動腳本
    └── docker_process_faq.sh # FAQ 處理腳本
```

## 🔧 常用命令

### 服務管理

```bash
# 啟動 API 服務
docker compose up -d api

# 查看服務狀態
docker compose ps

# 查看日誌
docker compose logs -f api

# 停止服務
docker compose down
```

### FAQ 處理

```bash
# 處理 FAQ 文件
docker compose --profile faq-processing run --rm process-faq data/faq.csv --collection my_collection

# 建立索引
docker compose --profile faq-processing run --rm index my_collection

# 列出所有 collections
docker compose --profile faq-processing run --rm list
```

### 使用腳本

```bash
# 啟動服務
./scripts/docker_start.sh

# 處理 FAQ
./scripts/docker_process_faq.sh my_collection data/faq.csv --questions
```

## 🎯 完整工作流程示例

### 1. 準備數據

```bash
# 創建 data 目錄
mkdir -p data

# 創建示例 FAQ 文件
cat > data/example_faq.csv << EOF
id,question,answer,source_file,source_page,source_section
1,如何使用系統？,請參考使用手冊第1章,manual.pdf,1,介紹
2,如何重置密碼？,請聯繫管理員或使用忘記密碼功能,manual.pdf,5,帳戶管理
3,系統支持哪些格式？,支持CSV、Markdown、Word文檔格式,manual.pdf,10,文件格式
EOF
```

### 2. 處理 FAQ 數據

```bash
# 處理 FAQ 並生成相似問題
./scripts/docker_process_faq.sh example_collection data/example_faq.csv --questions
```

### 3. 啟動 API 服務

```bash
# 啟動服務
./scripts/docker_start.sh
```

### 4. 測試搜索

```bash
# 測試 FAQ 搜索
curl -X POST "http://localhost:8000/faq-search" \
  -H "Content-Type: application/json" \
  -d '{
    "collection": "example_collection",
    "query": "如何使用系統？",
    "top_k": 5
  }'

# 測試智能問答
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "collection": "example_collection",
    "question": "如何使用系統？",
    "top_k": 2
  }'
```

## 🐛 故障排除

### 常見問題

1. **API 服務無法啟動**
   - 檢查 `.env` 文件中的 `OPENAI_API_KEY` 是否正確
   - 查看日誌：`docker compose logs api`

2. **FAQ 處理失敗**
   - 檢查 CSV 文件格式是否正確
   - 確保文件路徑正確
   - 查看日誌：`docker compose logs process-faq`

3. **索引建立失敗**
   - 確保 FAQ 處理成功完成
   - 檢查 collections 目錄權限

4. **FAQ 搜索返回空結果**
   - 確保 FAQ 數據已正確處理
   - 檢查 collection 名稱是否正確
   - 查看 collections 列表：`docker compose --profile faq-processing run --rm list`

### 重新開始

如果需要重新開始：

```bash
# 停止所有服務
docker compose down

# 刪除 collections 目錄（會刪除所有數據）
rm -rf collections

# 重新處理 FAQ
./scripts/docker_process_faq.sh my_collection data/faq.csv

# 重新啟動服務
./scripts/docker_start.sh
```

### 調試命令

```bash
# 查看 collections 狀態
docker compose --profile faq-processing run --rm list

# 檢查 collection 詳細信息
curl -X GET "http://localhost:8000/collections" | jq .

# 測試普通搜索（不帶去重）
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"collection": "my_collection", "query": "測試", "top_k": 5}'

# 測試 FAQ 搜索（帶去重）
curl -X POST "http://localhost:8000/faq-search" \
  -H "Content-Type: application/json" \
  -d '{"collection": "my_collection", "query": "測試", "top_k": 5}'
```

## 📚 API 端點

### 搜尋端點

- `POST /faq-search` - FAQ 搜尋（推薦，自動去重）
- `POST /search` - 一般搜尋（不去重）
- `POST /ask` - 智能問答（使用 LLM 生成回答）

### 管理端點

- `GET /collections` - 列出所有 collections
- `GET /health` - 健康檢查
- `GET /docs` - API 文檔

### 請求格式

#### FAQ 搜索
```json
{
  "collection": "my_collection",
  "query": "如何使用系統？",
  "top_k": 5
}
```

#### 智能問答
```json
{
  "collection": "my_collection",
  "question": "如何使用系統？",
  "top_k": 2
}
```

### 響應格式

#### FAQ 搜索響應
```json
{
  "results": [
    {
      "text": "如何使用系統？",
      "distance": 0.123,
      "metadata": {
        "qa_id": "faq_001",
        "answer": "請參考使用手冊第1章",
        "source_file": "manual.pdf",
        "source_page": 1,
        "source_section": "介紹"
      }
    }
  ],
  "timing": {
    "embedding_time": 0.1,
    "search_time": 0.05,
    "total_time": 0.15
  },
  "stats": {
    "search_type": "exact",
    "total_results_before_dedup": 15,
    "final_results_after_dedup": 5
  }
}
```

## 🔒 安全注意事項

1. 不要將 `.env` 文件提交到版本控制
2. 定期更新 OpenAI API 金鑰
3. 在生產環境中使用 HTTPS
4. 限制 API 訪問權限

## 📞 支援

如果遇到問題，請檢查：
1. Docker 和 Docker Compose 版本
2. 網絡連接
3. OpenAI API 配額
4. 系統資源（記憶體、磁盤空間）

## 🎉 成功指標

當您看到以下結果時，表示設置成功：

1. **FAQ 處理成功**：
   ```
   ✅ FAQ文件處理完成
   ✅ 索引建立完成
   ```

2. **API 服務正常**：
   ```
   ✅ API 服務已成功啟動
   API 地址: http://localhost:8000
   ```

3. **搜索返回結果**：
   ```json
   {
     "results": [...],
     "stats": {
       "final_results_after_dedup": 5
     }
   }
   ```
