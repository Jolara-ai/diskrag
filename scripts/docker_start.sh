#!/bin/bash

# DiskRAG Docker 啟動腳本
# 用法: ./scripts/docker_start.sh [--build]

set -e

BUILD=false

# 檢查參數
if [ $# -eq 1 ] && [ "$1" = "--build" ]; then
    BUILD=true
fi

echo "🚀 DiskRAG Docker 啟動腳本"
echo ""

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
    echo ""
fi

# 檢查collections目錄是否存在
if [ ! -d "collections" ]; then
    echo "📁 創建 collections 目錄..."
    mkdir -p collections
fi

# 檢查data目錄是否存在
if [ ! -d "data" ]; then
    echo "📁 創建 data 目錄..."
    mkdir -p data
fi

# 檢查logs目錄是否存在
if [ ! -d "logs" ]; then
    echo "📁 創建 logs 目錄..."
    mkdir -p logs
fi

# 如果需要重新構建
if [ "$BUILD" = true ]; then
    echo "🔨 重新構建 Docker 映像..."
    docker-compose build --no-cache
fi

# 啟動API服務
echo "🚀 啟動 API 服務..."
docker-compose up -d api

# 等待服務啟動
echo "⏳ 等待服務啟動..."
sleep 10

# 檢查服務狀態
echo "🔍 檢查服務狀態..."
if docker-compose ps | grep -q "Up"; then
    echo "✅ API 服務已成功啟動"
    echo ""
    echo "📋 服務資訊:"
    echo "  API 地址: http://localhost:8000"
    echo "  API 文檔: http://localhost:8000/docs"
    echo "  健康檢查: http://localhost:8000/health"
    echo ""
    echo "🔧 可用操作:"
    echo "  1. 查看服務狀態: docker-compose ps"
    echo "  2. 查看日誌: docker-compose logs -f api"
    echo "  3. 停止服務: docker-compose down"
    echo "  4. 處理FAQ: ./scripts/docker_process_faq.sh <collection_name> <csv_file>"
    echo "  5. 列出collections: docker-compose --profile faq-processing run --rm list"
    echo ""
    echo "📖 API 使用示例:"
    echo "  curl -X POST 'http://localhost:8000/faq-search' \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"collection\": \"your_collection\", \"query\": \"你的問題\", \"top_k\": 5}'"
else
    echo "❌ API 服務啟動失敗"
    echo "請檢查日誌: docker-compose logs api"
    exit 1
fi
