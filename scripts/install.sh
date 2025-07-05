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
mkdir -p data collections logs

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
  enabled: false
  provider: "openai"
  model: "gpt-4o-mini"
  max_questions: 5
  temperature: 0.7

chunk:
  size: 300
  overlap: 50
  min_size: 50

output:
  format: "parquet"
  compression: "snappy"
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
什麼是 DiskANN 的 R 參數？,R 參數是 Vamana 圖中每個節點的最大度數限制，它控制了圖的連接密度，較大的 R 值通常會提高搜索精度但增加記憶體使用量。
DiskANN 的索引建構過程是怎樣的？,DiskANN 的索引建構包括：1) 建立初始 Vamana 圖 2) 圖的度數修剪 3) 分層數據放置 4) 生成查詢時的記憶體佈局。
DiskANN 支援動態更新嗎？,是的，DiskANN 支援動態插入和刪除操作，但需要定期重建索引以保持最佳效能，特別是當數據變化較大時。
DiskANN 的查詢算法是如何工作的？,DiskANN 的查詢從記憶體中的起始點開始，使用貪婪搜索在 Vamana 圖中導航，當需要訪問磁碟上的節點時會批量預取以減少 I/O 次數。
DiskANN 適用於哪些應用場景？,DiskANN 適用於大規模向量搜索場景，如：1) 圖像和視頻檢索 2) 自然語言處理 3) 推薦系統 4) 向量數據庫 5) 機器學習特徵匹配。
DiskANN 的效能瓶頸在哪裡？,主要瓶頸包括：1) 磁碟 I/O 延遲 2) 記憶體和磁碟間的數據傳輸 3) 圖構建時的計算複雜度 4) 高維數據的距離計算成本。
如何調整 DiskANN 的參數以獲得最佳效能？,需要根據數據特性調整：1) R 值平衡精度和記憶體使用 2) 記憶體預算分配 3) 預取策略參數 4) 索引刷新頻率。
DiskANN 論文的主要貢獻是什麼？,主要貢獻包括：1) 提出了磁碟友好的 ANN 算法 2) 設計了 Vamana 圖結構 3) 實現了十億級別向量的高效搜索 4) 證明了磁碟基 ANN 的可行性。
DiskANN 與 HNSW 算法有什麼區別？,主要區別：1) DiskANN 專門針對磁碟存儲優化 2) 使用單層 Vamana 圖而非層次結構 3) 更好的記憶體控制 4) 針對大規模數據集的特殊優化。
DiskANN 的時間複雜度是多少？,查詢時間複雜度約為 O(log n)，其中 n 是數據集大小，但實際效能還受到磁碟 I/O 和緩存命中率的影響。
DiskANN 如何處理高維數據？,DiskANN 使用維度縮減技術和有效的距離計算方法來處理高維數據，並通過圖結構來減少實際需要計算距離的點數量。
什麼是 DiskANN 的預取策略？,預取策略是指在查詢過程中提前從磁碟讀取可能需要的數據，以減少查詢延遲，通常基於圖的拓撲結構來預測訪問模式。
DiskANN 的記憶體使用量如何控制？,通過限制記憶體中節點的數量、控制圖的度數、使用壓縮技術，以及智能的緩存替換策略來控制記憶體使用量。
DiskANN 論文中的實驗結果如何？,實驗顯示 DiskANN 在處理十億級向量時能夠達到高召回率（>95%），同時查詢延遲控制在毫秒級別，相比純記憶體方法大幅降低了記憶體需求。
DiskANN 支援哪些距離度量？,DiskANN 主要支援歐幾里得距離和內積距離，也可以擴展支援其他距離度量，但需要考慮計算效率的影響。
如何評估 DiskANN 的搜索品質？,主要使用召回率（Recall）來評估搜索品質，即返回的近鄰中真正近鄰的比例，通常以 Recall@K 的形式報告。
DiskANN 的可擴展性如何？,DiskANN 具有良好的可擴展性，可以處理從百萬到十億級別的向量數據集，記憶體使用量基本保持恆定。
DiskANN 的開源實現在哪裡？,Microsoft Research 提供了 DiskANN 的開源實現，可在 GitHub 上找到，包含了完整的索引建構和查詢代碼。
DiskANN 與傳統倒排索引有什麼不同？,DiskANN 是基於圖的向量搜索方法，適合密集向量，而倒排索引主要用於稀疏特徵和文本搜索，兩者的數據結構和算法原理完全不同。
DiskANN 在工業界有哪些應用？,DiskANN 被應用於大型科技公司的搜索引擎、推薦系統、圖像檢索服務等，特別是需要處理海量向量數據的場景。
DiskANN 的未來發展方向是什麼？,未來可能的發展包括：1) 更好的動態更新支援 2) 多模態數據處理 3) 分散式版本 4) 更高效的壓縮算法 5) GPU 加速等。
如何開始學習和使用 DiskANN？,建議先閱讀原始論文理解核心概念，然後查看開源代碼實現，在小規模數據集上進行實驗，逐步擴展到大規模應用。
DiskANN 論文發表在哪裡？,DiskANN 論文發表在 NIPS 2019（現在稱為 NeurIPS），題目是 "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node"。
DiskANN 與向量數據庫的關係是什麼？,DiskANN 是許多現代向量數據庫的核心算法之一，為向量數據庫提供了高效的大規模向量索引和搜索能力。
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
echo "互動式介面:"
echo "  python interactive.py"
echo