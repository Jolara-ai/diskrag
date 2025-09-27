#!/bin/bash

# DiskRAG Docker 分发包创建脚本
# 用法: ./scripts/create_distribution.sh [--version <version>]

set -e

VERSION="1.0.0"
DIST_DIR="diskrag-docker-${VERSION}"

# 解析參數
while [[ $# -gt 0 ]]; do
    case $1 in
        --version)
            VERSION="$2"
            DIST_DIR="diskrag-docker-${VERSION}"
            shift 2
            ;;
        *)
            echo "❌ 未知參數: $1"
            echo "用法: $0 [--version <version>]"
            exit 1
            ;;
    esac
done

echo "📦 創建 DiskRAG Docker 分发包"
echo "版本: $VERSION"
echo "目錄: $DIST_DIR"
echo ""

# 清理舊的分发包
if [ -d "$DIST_DIR" ]; then
    echo "🧹 清理舊的分发包..."
    rm -rf "$DIST_DIR"
fi

# 創建分发包目錄
echo "📁 創建分发包目錄..."
mkdir -p "$DIST_DIR"
mkdir -p "$DIST_DIR/collections"
mkdir -p "$DIST_DIR/data"
mkdir -p "$DIST_DIR/logs"
mkdir -p "$DIST_DIR/scripts"

# 複製必要文件
echo "📋 複製文件..."
cp docker-compose.simple.yml "$DIST_DIR/"
cp env.example "$DIST_DIR/"
cp README_DOCKER.md "$DIST_DIR/"
cp DISTRIBUTION_README.md "$DIST_DIR/README.md"
cp scripts/docker_start.sh "$DIST_DIR/scripts/"
cp scripts/docker_process_faq.sh "$DIST_DIR/scripts/"

# 設置腳本權限
chmod +x "$DIST_DIR/scripts/"*.sh

# 創建 .gitignore
echo "📝 創建 .gitignore..."
cat > "$DIST_DIR/.gitignore" << EOF
# 環境變數
.env

# 數據目錄
collections/
data/
logs/

# 系統文件
.DS_Store
Thumbs.db

# 日誌文件
*.log
EOF

# 創建版本信息
echo "📄 創建版本信息..."
cat > "$DIST_DIR/VERSION" << EOF
DiskRAG Docker Distribution
Version: ${VERSION}
Build Date: $(date)
EOF

# 創建壓縮包
echo "🗜️  創建壓縮包..."
tar -czf "${DIST_DIR}.tar.gz" "$DIST_DIR"

# 清理臨時目錄
rm -rf "$DIST_DIR"

echo ""
echo "✅ 分发包創建完成！"
echo "📦 文件: ${DIST_DIR}.tar.gz"
echo ""
echo "📋 分发包內容:"
echo "  - docker-compose.simple.yml (Docker 配置)"
echo "  - env.example (環境變數示例)"
echo "  - README.md (使用說明)"
echo "  - scripts/ (腳本目錄)"
echo "  - collections/ (FAQ 集合目錄)"
echo "  - data/ (原始數據目錄)"
echo "  - logs/ (日誌目錄)"
echo ""
echo "📤 分發步驟:"
echo "1. 將 ${DIST_DIR}.tar.gz 發送給同事"
echo "2. 同事解壓: tar -xzf ${DIST_DIR}.tar.gz"
echo "3. 進入目錄: cd $DIST_DIR"
echo "4. 設置環境: cp env.example .env"
echo "5. 編輯 .env 文件，填入 OPENAI_API_KEY"
echo "6. 處理 FAQ: ./scripts/docker_process_faq.sh my_collection data/faq.csv"
echo "7. 啟動服務: ./scripts/docker_start.sh"
