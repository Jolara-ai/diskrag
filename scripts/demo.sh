#!/bin/bash

# DiskRAG 快速體驗腳本
# 這個腳本會自動完成：處理範例資料 -> 建立索引 -> 搜尋測試
# 注意：請先執行 make install 或 ./scripts/install.sh 安裝環境

set -e

echo "╔══════════════════════════════════════╗"
echo "║   DiskRAG 快速體驗腳本               ║"
echo "╚══════════════════════════════════════╝"
echo ""

# 檢查環境是否已安裝
if [ ! -d "venv" ]; then
    echo "❌ 虛擬環境不存在"
    echo ""
    echo "請先執行安裝："
    echo "  make install"
    echo "  或"
    echo "  ./scripts/install.sh"
    echo ""
    exit 1
fi

# 啟用虛擬環境
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "❌ 虛擬環境不完整"
    echo "請先執行: make install 或 ./scripts/install.sh"
    exit 1
fi

# 檢查必要的模組
echo "🔍 檢查環境..."
if ! python -c "import numpy" 2>/dev/null; then
    echo "❌ NumPy 未安裝"
    echo "請先執行: make install"
    exit 1
fi

if ! python -c "import pydiskann" 2>/dev/null; then
    echo "❌ pydiskann 未安裝"
    echo "請先執行: make install"
    exit 1
fi

echo "✅ 環境檢查通過"
echo ""

# 檢查 API Key
if [ -z "$OPENAI_API_KEY" ]; then
    if [ -f ".env" ]; then
        source .env 2>/dev/null || true
    fi
    
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "⚠️  未設定 OPENAI_API_KEY"
        echo ""
        echo "請選擇以下方式之一："
        echo "1. 建立 .env 檔案並新增: OPENAI_API_KEY=your-api-key"
        echo "2. 設定環境變數: export OPENAI_API_KEY=your-api-key"
        echo ""
        read -p "是否要現在設定 API Key? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            read -p "請輸入您的 OpenAI API Key: " api_key
            echo "OPENAI_API_KEY=$api_key" > .env
            export OPENAI_API_KEY="$api_key"
            echo "✅ API Key 已儲存到 .env 檔案"
        else
            echo "❌ 無法繼續，請先設定 OPENAI_API_KEY"
            exit 1
        fi
    fi
fi

# 使用範例資料
COLLECTION_NAME="demo_collection"
EXAMPLE_FILE="examples/faq_data.csv"

# 如果範例檔案不存在，使用 data/example.csv
if [ ! -f "$EXAMPLE_FILE" ]; then
    EXAMPLE_FILE="data/example.csv"
fi

if [ ! -f "$EXAMPLE_FILE" ]; then
    echo "❌ 找不到範例檔案"
    echo "請確保存在以下檔案之一："
    echo "  - examples/faq_data.csv"
    echo "  - data/example.csv"
    exit 1
fi

echo "📊 步驟 1/3: 處理範例資料..."
echo "   檔案: $EXAMPLE_FILE"
echo "   Collection: $COLLECTION_NAME"
echo ""

# 處理 FAQ 檔案
if [ -f "scripts/process_faq.sh" ]; then
    ./scripts/process_faq.sh "$COLLECTION_NAME" "$EXAMPLE_FILE"
else
    # 直接使用 diskrag 命令
    if command -v diskrag >/dev/null 2>&1; then
        diskrag process "$EXAMPLE_FILE" --collection "$COLLECTION_NAME"
        diskrag index "$COLLECTION_NAME"
    else
        python diskrag.py process "$EXAMPLE_FILE" --collection "$COLLECTION_NAME"
        python diskrag.py index "$COLLECTION_NAME"
    fi
fi

echo ""
echo "🔍 步驟 2/3: 測試搜尋..."
echo ""

# 根據使用的檔案決定測試查詢
if [ "$EXAMPLE_FILE" = "data/example.csv" ]; then
    # DiskANN 相關查詢
    TEST_QUERIES=(
        "什麼是 DiskANN？"
        "DiskANN 如何處理記憶體不足的問題？"
        "DiskANN 的優勢是什麼？"
    )
else
    # 洗碗機相關查詢（對應 examples/faq_data.csv）
    TEST_QUERIES=(
        "這份使用手冊適用於哪個型號？"
        "如何購買原裝配件？"
        "如何設定水質硬度？"
    )
fi

for query in "${TEST_QUERIES[@]}"; do
    echo "📝 查詢: $query"
    echo "────────────────────────────────────────"
    
    if command -v diskrag >/dev/null 2>&1; then
        diskrag search "$COLLECTION_NAME" "$query" --top-k 3 2>/dev/null || true
    else
        python diskrag.py search "$COLLECTION_NAME" "$query" --top-k 3 2>/dev/null || true
    fi
    
    echo ""
    sleep 1
done

echo ""
echo "╔══════════════════════════════════════╗"
echo "║          🎉 體驗完成！                ║"
echo "╚══════════════════════════════════════╝"
echo ""
echo "✅ 您已經成功："
echo "   1. 處理了範例 FAQ 資料"
echo "   2. 建立了向量索引"
echo "   3. 進行了搜尋測試"
echo ""
echo "📚 下一步："
echo ""
echo "⚠️  重要：執行以下操作前，請先載入虛擬環境："
echo "   source venv/bin/activate"
echo ""
echo "   1. 查看完整文檔: cat README.md"
echo "   2. 處理自己的資料:"
echo "      source venv/bin/activate"
echo "      ./scripts/process_faq.sh my_collection data/my_faq.csv"
echo "   3. 搜尋測試:"
echo "      source venv/bin/activate"
echo "      ./scripts/search_faq.sh $COLLECTION_NAME '你的問題'"
echo "   4. 啟動 API 服務:"
echo "      source venv/bin/activate"
echo "      python app.py"
echo ""
echo "💡 提示: 使用 'diskrag list' 查看所有 collections（需先載入環境）"


