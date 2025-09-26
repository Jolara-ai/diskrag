#!/bin/bash

# DiskRAG 一鍵安裝腳本

set -e

echo "╔══════════════════════════════════════╗"
echo "║      DiskRAG 一鍵安裝腳本           ║"
echo "╚══════════════════════════════════════╝"
echo

# 檢查 Python 版本
echo "檢查 Python 版本..."
if ! command -v python3 &> /dev/null; then
    echo "錯誤: 未找到 Python 3"
    echo "請先安裝 Python 3.8 或更新版本"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "找到 Python $PYTHON_VERSION"

# 建立虛擬環境
echo
echo "建立虛擬環境..."
python3 -m venv venv

# 啟用虛擬環境
echo "啟用虛擬環境..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Unix-like
    source venv/bin/activate
fi

# 升級 pip
echo
echo "升級 pip..."
pip install --upgrade pip

# 安裝依賴
echo
echo "安裝依賴套件..."
pip install -r requirements.txt

# 建立必要目錄
echo
echo "建立目錄結構..."
mkdir -p data collections logs examples

# 建立預設設定檔
if [ ! -f "config.yaml" ]; then
    echo "建立預設設定檔..."
    cat > config.yaml << 'EOF'
# DiskRAG 設定檔
collection: "default"

embedding:
  provider: "openai"
  model: "text-embedding-3-small"
  max_retries: 3
  retry_delay: 2

question_generation:
  enabled: true
  provider: "openai"
  model: "gpt-4o-mini"
  max_questions: 5
  temperature: 0.7
  max_retries: 3
  retry_delay: 2

chunk:
  size: 300
  overlap: 50
  min_size: 50

output:
  format: "parquet"
  compression: "snappy"

# 新增索引參數區塊
index:
  R: 32
  L: 64
  alpha: 1.2
EOF
fi

# 建立範例 FAQ 檔案
if [ ! -f "examples/faq_data.csv" ]; then
    echo "建立範例 FAQ 檔案..."
    cat > examples/faq_data.csv << 'EOF'
id,question,answer,source_file,source_page,source_section,source_image
faq_001,這份使用手冊適用於哪個型號的洗碗機？,適用於 EBF7531SBA 型號的全嵌式洗碗機。,EBF7531SBA_ZH_Manual.pdf,1,封面,images/cover.png
faq_002,如何購買原裝配件？,應訪問 https://www.bosch-home.com/accessories/ 或聯繫當地授權經銷商。,EBF7531SBA_ZH_Manual.pdf,2,配件資訊,
faq_003,8歲以下的青少年可以使用嗎？,不可以，未滿 8 歲的青少年不得使用本機。,EBF7531SBA_ZH_Manual.pdf,3,安全資訊,images/safety.png
faq_004,如何設定水質硬度？,在基本設定中，選擇「水質硬度」選項，並從等級 1 到 10 中選擇對應您所在地區的水質硬度。,EBF7531SBA_ZH_Manual.pdf,15,基本設定,images/water_hardness.png
faq_005,洗碗機可以洗滌哪些物品？,可以洗滌：餐具、玻璃杯、碗盤、鍋具等。不可洗滌：木製餐具、鋁製鍋具、塑膠容器等。,EBF7531SBA_ZH_Manual.pdf,8,使用說明,images/items.png
EOF
fi

# 建立範例檔案
if [ ! -f "data/example.csv" ]; then
    echo "建立範例檔案..."
    cat > data/example.csv << 'EOF'
question,answer
什麼是 DiskANN？,DiskANN 是一個可擴展的近似最近鄰搜索算法，專門設計用於處理大規模向量數據集，特別是當數據集大小超過記憶體容量時。
DiskANN 解決了什麼問題？,DiskANN 解決了大規模向量搜索中的記憶體限制問題，允許在磁碟上建立和查詢十億級別的向量索引，同時保持高精度和高效能。
DiskANN 的核心原理是什麼？,DiskANN 結合了圖形導航搜索和分層索引結構，將熱點數據保存在記憶體中，冷數據存儲在磁碟上，通過智能的數據分層來優化查詢效能。
什麼是 Vamana 圖？,Vamana 是 DiskANN 使用的圖形結構，它是一個度數受限的圖，每個節點的鄰居數量有上限，這樣可以控制記憶體使用量並提高搜索效率。
DiskANN 相比於其他 ANN 算法有什麼優勢？,DiskANN 的主要優勢包括：1) 可處理超大規模數據集 2) 記憶體使用量可控 3) 查詢延遲穩定 4) 支援動態更新 5) 在精度和效能間有良好平衡。
DiskANN 如何處理記憶體不足的問題？,DiskANN 使用分層架構，將經常訪問的節點和邊緩存在記憶體中，較少訪問的數據存儲在磁碟上，通過預取和緩存策略來減少磁碟 I/O。
EOF
fi

# 建立 .env 範例檔案
if [ ! -f ".env.example" ]; then
    echo "建立 .env 範例檔案..."
    cat > .env.example << 'EOF'
# DiskRAG 環境變數範例
# 請複製此檔案為 .env 並填入您的 API 金鑰

# OpenAI API 金鑰 (必需)
OPENAI_API_KEY=your-openai-api-key-here

# 可選：Vertex AI 專案 ID (如果使用 Google Cloud)
# VERTEX_PROJECT_ID=your-vertex-project-id
EOF
fi

# 建立 README 檔案
if [ ! -f "README_QUICKSTART.md" ]; then
    echo "建立快速開始指南..."
    cat > README_QUICKSTART.md << 'EOF'
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
EOF
fi

# 建立快捷命令
echo
echo "建立快捷命令..."
cat > diskrag << 'EOF'
#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$DIR/venv/bin/activate" 2>/dev/null || source "$DIR/venv/Scripts/activate" 2>/dev/null
python "$DIR/diskrag.py" "$@"
EOF
chmod +x diskrag

# 檢查 OPENAI_API_KEY
echo
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  注意: 未設定 OPENAI_API_KEY"
    echo
    echo "請設定環境變數:"
    echo "export OPENAI_API_KEY='your-api-key'"
    echo
    echo "或建立 .env 檔案:"
    echo "echo \"OPENAI_API_KEY=your-api-key\" > .env"
else
    echo "✓ 已設定 OPENAI_API_KEY"
fi

# 完成訊息
echo
echo "╔══════════════════════════════════════╗"
echo "║         安裝完成！                   ║"
echo "╚══════════════════════════════════════╝"
echo
echo "使用方式:"
echo "  ./diskrag process data/example.csv --collection example"
echo "  ./diskrag index example"
echo "  ./diskrag search example '什麼是 DiskRAG'"
echo
echo "或啟用虛擬環境後使用:"
echo "  source venv/bin/activate  # Unix/Linux/macOS"
echo "  venv\\Scripts\\activate     # Windows"
echo "  python diskrag.py --help"
echo