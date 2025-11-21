#!/bin/bash

# DiskRAG FAQ 搜索腳本
# 用法: ./scripts/search_faq.sh <collection_name> <query>

set -e

# 檢查參數
if [ $# -lt 2 ]; then
    echo "❌ 用法錯誤"
    echo "用法: $0 <collection_name> <query>"
    echo ""
    echo "參數說明:"
    echo "  collection_name: 集合名稱"
    echo "  query: 搜索查詢"
    echo ""
    echo "範例:"
    echo "  $0 my_manual 'EBF7531SBA 這台機器怎麼用？'"
    exit 1
fi

COLLECTION_NAME="$1"
QUERY="$2"

echo "🔍 DiskRAG FAQ 搜索"
echo "Collection: $COLLECTION_NAME"
echo "查詢: $QUERY"
echo ""

# 檢查環境
if [ ! -f "scripts/check_env.sh" ]; then
    echo "❌ 環境檢查腳本不存在"
    exit 1
fi

# 執行環境檢查
source scripts/check_env.sh

# 啟用虛擬環境（如果尚未啟用）
if [ -z "$VIRTUAL_ENV" ]; then
    echo "🔧 啟用虛擬環境..."
    if [ -d "venv/bin" ]; then
        source venv/bin/activate
    else
        echo "❌ 虛擬環境不存在"
        echo "請先執行: ./scripts/install.sh 或 make install"
        exit 1
    fi
fi

# 執行搜索
echo "🔍 正在搜索..."
if command -v diskrag >/dev/null 2>&1; then
    diskrag search "$COLLECTION_NAME" "$QUERY" --top-k 5
else
    python diskrag.py search "$COLLECTION_NAME" "$QUERY" --top-k 5
fi

