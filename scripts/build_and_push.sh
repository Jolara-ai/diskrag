#!/bin/bash

# DiskRAG Docker 構建和推送腳本
# 用法: ./scripts/build_and_push.sh [--push] [--tag <tag>]

set -e

PUSH=false
TAG="latest"

# 解析參數
while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH=true
            shift
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        *)
            echo "❌ 未知參數: $1"
            echo "用法: $0 [--push] [--tag <tag>]"
            exit 1
            ;;
    esac
done

echo "🔨 DiskRAG Docker 構建腳本"
echo "標籤: $TAG"
echo "推送: $PUSH"
echo ""

# 檢查 docker-compose.yml 是否存在
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ docker-compose.yml 不存在"
    exit 1
fi

# 構建映像
echo "🔨 構建 Docker 映像..."
docker-compose build --no-cache

# 標籤映像
IMAGE_NAME="diskrag:${TAG}"
echo "🏷️  標籤映像為: $IMAGE_NAME"
docker tag diskrag_api:latest $IMAGE_NAME

# 如果需要推送
if [ "$PUSH" = true ]; then
    echo "📤 推送映像到 Docker Hub..."
    # 這裡需要您設置 Docker Hub 用戶名
    # docker tag $IMAGE_NAME your-username/diskrag:${TAG}
    # docker push your-username/diskrag:${TAG}
    echo "⚠️  請手動設置 Docker Hub 用戶名並推送"
    echo "範例:"
    echo "  docker tag $IMAGE_NAME your-username/diskrag:${TAG}"
    echo "  docker push your-username/diskrag:${TAG}"
fi

echo ""
echo "✅ 構建完成！"
echo "映像名稱: $IMAGE_NAME"
echo ""
echo "📋 使用說明:"
echo "1. 本地使用:"
echo "   docker-compose up -d api"
echo ""
echo "2. 在其他機器使用:"
echo "   docker pull your-username/diskrag:${TAG}"
echo "   docker run -d -p 8000:8000 -v ./collections:/app/collections your-username/diskrag:${TAG}"
