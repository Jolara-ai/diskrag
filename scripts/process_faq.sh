#!/bin/bash

# DiskRAG FAQ 處理腳本
# 用法: ./scripts/process_faq.sh <collection_name> <csv_file> [--questions]

set -e

# 檢查參數
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "❌ 用法錯誤"
    echo "用法: $0 <collection_name> <csv_file> [--questions]"
    echo ""
    echo "參數說明:"
    echo "  collection_name: 集合名稱"
    echo "  csv_file: CSV文件路徑"
    echo "  --questions: 可選，生成相似問題（預設不生成）"
    echo ""
    echo "示例:"
    echo "  $0 my_manual data/faq_data.csv"
    echo "  $0 product_faq examples/faq_data.csv --questions"
    exit 1
fi

COLLECTION_NAME="$1"
CSV_FILE="$2"
GENERATE_QUESTIONS=false

# 檢查是否有 --questions 參數
if [ $# -eq 3 ] && [ "$3" = "--questions" ]; then
    GENERATE_QUESTIONS=true
fi

echo "🚀 DiskRAG FAQ 處理腳本"
echo "Collection: $COLLECTION_NAME"
echo "CSV 文件: $CSV_FILE"
echo "生成相似問題: $GENERATE_QUESTIONS"
echo ""

# 檢查環境
if [ ! -f "scripts/check_env.sh" ]; then
    echo "❌ 環境檢查腳本不存在"
    exit 1
fi

# 運行環境檢查
source scripts/check_env.sh

# 檢查CSV文件是否存在
if [ ! -f "$CSV_FILE" ]; then
    echo "❌ CSV 文件不存在: $CSV_FILE"
    echo "請檢查文件路徑是否正確"
    exit 1
fi

# 檢查CSV文件格式
echo "🔍 檢查CSV文件格式..."
if ! head -n 1 "$CSV_FILE" | grep -q "question.*answer"; then
    echo "⚠️  警告: CSV文件可能不是標準FAQ格式"
    echo "標準格式應包含: id,question,answer,source_file,source_page,source_section,source_image"
    echo "繼續處理..."
fi

# 激活虛擬環境
echo "🔧 激活虛擬環境..."
if [ -d "venv/bin" ]; then
    source venv/bin/activate
elif [ -d "venv/Scripts" ]; then
    source venv/Scripts/activate
else
    echo "❌ 虛擬環境不存在"
    echo "請先運行: ./scripts/install.sh"
    exit 1
fi

# 檢查Python模組
echo "🔍 檢查Python模組..."
if ! python -c "import diskrag" 2>/dev/null; then
    echo "❌ diskrag 模組未安裝"
    echo "請先運行: ./scripts/install.sh"
    exit 1
fi

# 處理FAQ文件
echo "🔄 開始處理FAQ文件..."
if [ "$GENERATE_QUESTIONS" = true ]; then
    echo "步驟1: 解析CSV並生成相似問題..."
else
    echo "步驟1: 解析CSV文件..."
fi

# 使用diskrag命令處理文件
if command -v diskrag >/dev/null 2>&1; then
    # 如果diskrag命令可用
    if [ "$GENERATE_QUESTIONS" = true ]; then
        diskrag process "$CSV_FILE" --collection "$COLLECTION_NAME" --questions
    else
        diskrag process "$CSV_FILE" --collection "$COLLECTION_NAME"
    fi
else
    # 使用Python模組
    if [ "$GENERATE_QUESTIONS" = true ]; then
        python -m diskrag process "$CSV_FILE" --collection "$COLLECTION_NAME" --questions
    else
        python -m diskrag process "$CSV_FILE" --collection "$COLLECTION_NAME"
    fi
fi

if [ $? -ne 0 ]; then
    echo "❌ FAQ文件處理失敗"
    exit 1
fi

echo "✅ FAQ文件處理完成"

# 建立索引
echo "步驟2: 建立索引..."
if command -v diskrag >/dev/null 2>&1; then
    # 檢查索引是否已存在
    if diskrag list | grep -q "^$COLLECTION_NAME"; then
        echo "🔍 發現已存在的 collection，檢查索引狀態..."
        # 這裡可以添加更詳細的索引檢查邏輯
        echo "✅ 索引已存在，跳過索引建立"
        echo "💡 如需重新建立索引，請運行: diskrag index $COLLECTION_NAME --force-rebuild"
    else
        diskrag index "$COLLECTION_NAME"
    fi
else
    # 使用Python模組
    if python -c "import diskrag; rag = diskrag.DiskRAG(); print('\\n'.join([c.name for c in rag.list_collections()]))" 2>/dev/null | grep -q "^$COLLECTION_NAME$"; then
        echo "🔍 發現已存在的 collection，檢查索引狀態..."
        echo "✅ 索引已存在，跳過索引建立"
        echo "💡 如需重新建立索引，請運行: python -m diskrag index $COLLECTION_NAME --force-rebuild"
    else
        python -m diskrag index "$COLLECTION_NAME"
    fi
fi

if [ $? -ne 0 ]; then
    echo "❌ 索引建立失敗"
    exit 1
fi

echo "✅ 索引建立完成"

# 顯示結果
echo ""
echo "🎉 FAQ處理完成！"
echo ""
echo "Collection資訊:"
echo "  名稱: $COLLECTION_NAME"
echo "  文件: $CSV_FILE"
echo ""
echo "下一步操作:"
echo "  1. 搜索測試: ./scripts/search_faq.sh $COLLECTION_NAME '你的問題'"
echo "  2. 啟動API服務: ./scripts/run_api.sh"
echo "  3. 查看collections: diskrag list"
echo ""
if [ "$GENERATE_QUESTIONS" = false ]; then
    echo "💡 提示: 如果需要生成相似問題，可以重新運行:"
    echo "  $0 $COLLECTION_NAME $CSV_FILE --questions"
    echo ""
fi
echo "API使用示例:"
echo "  curl -X POST 'http://localhost:8000/faq-search' \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"collection\": \"$COLLECTION_NAME\", \"query\": \"你的問題\", \"top_k\": 5}'" 