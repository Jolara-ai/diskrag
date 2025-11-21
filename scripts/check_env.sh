#!/bin/bash

# DiskRAG 環境檢查腳本

set -e

echo "🔍 檢查 DiskRAG 環境..."

# 檢查虛擬環境
if [ ! -d "venv" ]; then
    echo "❌ 虛擬環境不存在"
    echo "請先執行: ./scripts/install.sh"
    exit 1
fi

# 檢查 Python 依賴
if [ ! -f "venv/bin/python" ]; then
    echo "❌ 虛擬環境不完整"
    echo "請先執行: ./scripts/install.sh"
    exit 1
fi

# 檢查設定檔
if [ ! -f "config.yaml" ]; then
    echo "❌ 設定檔不存在"
    echo "請先執行: ./scripts/install.sh"
    exit 1
fi

# 檢查環境變數
if [ -z "$OPENAI_API_KEY" ]; then
    if [ -f ".env" ]; then
        export $(grep -v '^#' .env | xargs)
    fi
    
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "❌ OPENAI_API_KEY 環境變數未設定"
        echo "請設定環境變數或建立 .env 檔案"
        echo "export OPENAI_API_KEY='your-api-key'"
        exit 1
    fi
fi

# 檢查必要目錄
for dir in data collections logs; do
    if [ ! -d "$dir" ]; then
        echo "📁 建立目錄: $dir"
        mkdir -p "$dir"
    fi
done

echo "✅ 環境檢查通過！"
echo "可以使用以下命令："
echo "  ./scripts/process_faq.sh <collection_name> <csv_file>"
echo "  ./scripts/run_api.sh" 