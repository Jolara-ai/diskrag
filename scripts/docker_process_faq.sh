#!/bin/bash

# DiskRAG Docker FAQ 處理腳本
# 用法: ./scripts/docker_process_faq.sh <collection_name> <csv_file> [--questions]

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
    echo ""
    echo "注意: 此腳本需要在 docker-compose.yml 同級目錄下執行"
    exit 1
fi

COLLECTION_NAME="$1"
CSV_FILE="$2"
GENERATE_QUESTIONS=false

# 檢查是否有 --questions 參數
if [ $# -eq 3 ] && [ "$3" = "--questions" ]; then
    GENERATE_QUESTIONS=true
fi

echo "🚀 DiskRAG Docker FAQ 處理腳本"
echo "Collection: $COLLECTION_NAME"
echo "CSV 文件: $CSV_FILE"
echo "生成相似問題: $GENERATE_QUESTIONS"
echo ""

# 檢查CSV文件是否存在
if [ ! -f "$CSV_FILE" ]; then
    echo "❌ CSV 文件不存在: $CSV_FILE"
    echo "請檢查文件路徑是否正確"
    exit 1
fi

# 檢查docker-compose.yml是否存在
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ docker-compose.yml 不存在"
    echo "請確保在正確的目錄下執行此腳本"
    exit 1
fi

# 檢查.env文件是否存在
if [ ! -f ".env" ]; then
    echo "⚠️  警告: .env 文件不存在"
    echo "請確保已設置 OPENAI_API_KEY 環境變數"
    echo "可以創建 .env 文件並添加: OPENAI_API_KEY=your_api_key_here"
fi

# 檢查CSV文件格式
echo "🔍 檢查CSV文件格式..."
if ! head -n 1 "$CSV_FILE" | grep -q "question.*answer"; then
    echo "⚠️  警告: CSV文件可能不是標準FAQ格式"
    echo "標準格式應包含: id,question,answer,source_file,source_page,source_section,source_image"
    echo "繼續處理..."
fi

# 處理FAQ文件
echo "🔄 開始處理FAQ文件..."
if [ "$GENERATE_QUESTIONS" = true ]; then
    echo "步驟1: 解析CSV並生成相似問題..."
    docker-compose --profile faq-processing run --rm process-faq "$CSV_FILE" --collection "$COLLECTION_NAME" --questions
else
    echo "步驟1: 解析CSV文件..."
    docker-compose --profile faq-processing run --rm process-faq "$CSV_FILE" --collection "$COLLECTION_NAME"
fi

if [ $? -ne 0 ]; then
    echo "❌ FAQ文件處理失敗"
    exit 1
fi

echo "✅ FAQ文件處理完成"

# 建立索引
echo "步驟2: 建立索引..."
docker-compose --profile faq-processing run --rm index "$COLLECTION_NAME"

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
echo "  1. 啟動API服務: docker-compose up -d api"
echo "  2. 查看collections: docker-compose --profile faq-processing run --rm list"
echo "  3. 測試API: curl -X POST 'http://localhost:8000/faq-search' \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"collection\": \"$COLLECTION_NAME\", \"query\": \"你的問題\", \"top_k\": 5}'"
echo ""
if [ "$GENERATE_QUESTIONS" = false ]; then
    echo "💡 提示: 如果需要生成相似問題，可以重新運行:"
    echo "  $0 $COLLECTION_NAME $CSV_FILE --questions"
    echo ""
fi
