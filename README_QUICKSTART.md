# DiskRAG 快速開始指南

## 🚀 5分鐘快速開始

### 1. 環境設置 (已完成)
✅ 虛擬環境已建立
✅ 依賴套件已安裝
✅ 目錄結構已建立
✅ 設定檔已建立

### 2. 設定 API 金鑰
```bash
# 複製環境變數範例
cp .env.example .env

# 編輯 .env 文件，填入您的 OpenAI API 金鑰
# OPENAI_API_KEY=your-api-key-here
```

### 3. 使用 FAQ 工作流程 (推薦)

#### 準備 FAQ 數據
```bash
# 使用範例 FAQ 文件
./scripts/process_faq.sh my_manual examples/faq_data.csv

# 或使用自己的 CSV 文件
./scripts/process_faq.sh my_collection data/my_faq.csv
```

#### 搜索測試
```bash
# 測試搜索
./scripts/search_faq.sh my_manual "EBF7531SBA 這台機器怎麼用？"
```

#### 啟動 API 服務
```bash
# 啟動 FastAPI 服務
./scripts/run_api.sh
```

### 4. 傳統工作流程

#### 處理文件
```bash
# 處理 FAQ 文件
python diskrag.py process data/example.csv --collection faq

# 處理 Markdown 文件
python diskrag.py process data/manual.md --collection manual
```

#### 建立索引
```bash
python diskrag.py index faq
```

#### 搜索
```bash
python diskrag.py search faq "DiskANN 解決了什麼問題?"
```

## 📁 目錄結構

```
diskrag/
├── data/                    # 數據文件
│   └── example.csv         # 範例文件
├── examples/               # 範例文件
│   └── faq_data.csv       # FAQ 範例
├── collections/            # 向量集合
├── logs/                   # 日誌文件
├── scripts/                # 腳本文件
│   ├── install.sh         # 安裝腳本
│   ├── process_faq.sh     # FAQ 處理腳本
│   ├── search_faq.sh      # FAQ 搜索腳本
│   └── run_api.sh         # API 服務腳本
├── config.yaml            # 設定檔
├── .env.example           # 環境變數範例
└── README_QUICKSTART.md   # 本文件
```

## 🔧 常用命令

### FAQ 工作流程 (推薦)
```bash
# 處理 FAQ 文件
./scripts/process_faq.sh <collection_name> <csv_file>

# 搜索 FAQ
./scripts/search_faq.sh <collection_name> <query>

# 啟動 API 服務
./scripts/run_api.sh
```

### 傳統工作流程
```bash
# 處理文件
python diskrag.py process <file> --collection <name>

# 建立索引
python diskrag.py index <collection_name>

# 搜索
python diskrag.py search <collection_name> <query>

# 列出所有 collections
python diskrag.py list
```

## 📊 FAQ CSV 格式

```csv
id,question,answer,source_file,source_page,source_section,source_image
faq_001,這份使用手冊適用於哪個型號的洗碗機？,適用於 EBF7531SBA 型號的全嵌式洗碗機。,EBF7531SBA_ZH_Manual.pdf,1,封面,images/cover.png
faq_002,如何購買原裝配件？,應訪問 https://www.bosch-home.com/accessories/ 或聯繫當地授權經銷商。,EBF7531SBA_ZH_Manual.pdf,2,配件資訊,
```

## 🌐 API 使用

### 啟動服務
```bash
./scripts/run_api.sh
```

### API 端點
- **FAQ 搜索**: `POST /faq-search`
- **普通搜索**: `POST /search`
- **健康檢查**: `GET /health`
- **Collections**: `GET /collections`

### 使用示例
```bash
# FAQ 搜索
curl -X POST 'http://localhost:8000/faq-search' \
  -H 'Content-Type: application/json' \
  -d '{
    "collection": "my_manual",
    "query": "EBF7531SBA 這台機器怎麼用？",
    "top_k": 5
  }'
```

## 🆘 故障排除

### 常見問題

1. **環境變數未設置**
   ```bash
   # 設置環境變數
   export OPENAI_API_KEY='your-api-key'
   ```

2. **虛擬環境未激活**
   ```bash
   # 激活虛擬環境
   source venv/bin/activate  # Linux/macOS
   source venv/Scripts/activate  # Windows
   ```

3. **Docker 未安裝**
   - 安裝 Docker: https://docs.docker.com/get-docker/
   - 安裝 Docker Compose: https://docs.docker.com/compose/install/

### 獲取幫助
- 查看完整文檔: `README.md`
- 查看工作流程文檔: `docs/FAQ_WORKFLOW.md`
- 運行測試: `python scripts/test_faq_workflow.py`
