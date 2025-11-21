# DiskRAG

基於 DiskANN 的向量搜尋系統，用於建立知識庫並進行搜尋。

> 
> ```bash
> # 安裝 uv
> curl -LsSf https://astral.sh/uv/install.sh | sh
> 
> # 使用 uv 安裝 Python 3.11 並建立虛擬環境
> uv venv --python 3.11
> ```

## 快速開始

### 方式一：使用 Makefile（推薦）

```bash
# 1. 先安裝環境
make install

# 2. 快速體驗（處理資料、建立索引、搜尋測試）
# 注意：Makefile 會自動載入虛擬環境，無需手動載入
make demo

# 或手動執行其他操作（Makefile 會自動載入環境）
make process-faq ARGS='my_faq data/faq.csv'  # 處理 FAQ
make search-faq ARGS='my_faq "你的問題"'      # 搜尋
make run-api                    # 啟動 API 服務
```

> 💡 **提示**：如果直接使用腳本或命令（不使用 Makefile），需要先載入虛擬環境：
> ```bash
> source venv/bin/activate
> ```

### 方式二：使用腳本

```bash
# 1. 先安裝環境
./scripts/install.sh

# 2. 載入虛擬環境（重要！）
source venv/bin/activate

# 3. 快速體驗
./scripts/demo.sh

# 或手動執行其他操作（需先載入環境）
./scripts/process_faq.sh my_faq data/faq.csv
./scripts/search_faq.sh my_faq "你的問題"
```

### 方式三：使用命令

```bash
# 先載入虛擬環境（重要！）
source venv/bin/activate

# 執行命令
diskrag process data/file.csv --collection my_collection
diskrag index my_collection
diskrag search my_collection "你的問題"
```

## 安裝

### 系統需求

- Python 3.11
- pip
- OpenAI API 金鑰

### 安裝 Python 3.11

如果系統上沒有 Python 3.11，請使用 `uv` 安裝：

```bash
# 1. 安裝 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 重新載入 shell（或重新開啟終端）
source ~/.zshrc  # 或 source ~/.bash_profile

# 3. 使用 uv 安裝 Python 3.11 並建立虛擬環境
uv venv --python 3.11
```


### 使用 Makefile

```bash
make install
```

### 使用腳本

```bash
chmod +x scripts/*.sh
./scripts/install.sh
```

> 💡 **Python 版本建議**: 建議使用 Python 3.11，可避免所有依賴套件的兼容性問題

## 設定

1. 複製環境變數範例：
```bash
cp env.example .env
```

2. 編輯 `.env` 檔案，填入 OpenAI API 金鑰：
```
OPENAI_API_KEY=your-api-key-here
```

取得 API 金鑰：造訪 [OpenAI Platform](https://platform.openai.com/api-keys)

## 資料格式

### FAQ CSV 格式（最簡單）

```csv
question,answer
什麼是 DiskANN？,DiskANN 是一個可擴展的近似最近鄰搜尋演算法...
DiskANN 解決了什麼問題？,DiskANN 解決了大規模向量搜尋中的記憶體限制問題...
```

### FAQ CSV 格式（完整，含出處）

```csv
id,question,answer,source_file,source_page,source_section
faq_001,這份使用手冊適用於哪個型號？,適用於 EBF7531SBA 型號...,manual.pdf,1,封面
```

### 其他格式

- Markdown: `.md`, `.markdown`

## 常用命令

### Makefile 命令

```bash
make help              # 顯示所有可用命令
make demo              # 快速體驗
make install           # 安裝
make process-faq ARGS='collection_name csv_file'  # 處理 FAQ
make search-faq ARGS='collection_name query'     # 搜尋
make run-api           # 啟動 API 服務
make test              # 執行測試
make clean             # 清理編譯產物
```

### diskrag 命令

> ⚠️ **重要**：執行以下命令前，請先載入虛擬環境：
> ```bash
> source venv/bin/activate
> ```

```bash
# 處理檔案
diskrag process data/file.csv --collection my_collection

# 建立索引
diskrag index my_collection

# 搜尋
diskrag search my_collection "你的問題"

# 列出所有 collections
diskrag list

# 查看幫助
diskrag --help
```

## API 服務

> ⚠️ **重要**：啟動服務前，請先載入虛擬環境：
> ```bash
> source venv/bin/activate
> ```

啟動服務：

```bash
# 方式一：使用 Makefile（會自動載入環境）
make run-api

# 方式二：手動啟動（需先載入環境）
source venv/bin/activate
python app.py
```

服務將在 `http://localhost:8000` 啟動

### API 端點

- `POST /faq-search` - FAQ 搜尋
- `POST /search` - 一般搜尋
- `GET /health` - 健康檢查
- `GET /collections` - 列出所有 collections

### 使用範例

```bash
curl -X POST 'http://localhost:8000/faq-search' \
  -H 'Content-Type: application/json' \
  -d '{
    "collection": "my_collection",
    "query": "你的問題",
    "top_k": 5
  }'
```

## 故障排除

### 環境變數未設定

```bash
export OPENAI_API_KEY='your-api-key'
```

### 虛擬環境未啟用

```bash
source venv/bin/activate
```

### CSV 檔案格式錯誤

- 確保 CSV 檔案包含 `question` 和 `answer` 欄位
- 檢查檔案編碼為 UTF-8

### 索引建立失敗

```bash
# 檢查 collection 是否存在
diskrag list

# 強制重建索引
diskrag index my_collection --force-rebuild
```

## 進階功能

### 合併 Collections

```bash
diskrag merge collection1 collection2 --target merged_collection
```

### 修復索引

```bash
diskrag doctor my_collection
```

### 處理整個目錄

```bash
diskrag process-dir data --prefix docs --recursive
```

### 高品質索引

```bash
diskrag index my_collection --target-quality high
```

## 配置

主要配置項在 `config.yaml`：

- **索引品質**: `fast`, `balanced` (預設), `high`
- **Embedding 模型**: `text-embedding-3-small` (預設), `text-embedding-3-large`
- **分塊大小**: 預設 300 字元

詳細配置說明請參考 [configs/README.md](configs/README.md)



## 常見問題

**Q: 如何取得 OpenAI API 金鑰？**  
A: 造訪 [OpenAI Platform](https://platform.openai.com/api-keys) 建立 API 金鑰

**Q: 支援哪些檔案格式？**  
A: CSV (FAQ)、Markdown (.md, .markdown)

## 許可證

MIT License
