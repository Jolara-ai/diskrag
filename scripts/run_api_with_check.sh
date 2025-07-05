#!/bin/bash

# 啟動 FastAPI 服務前檢查 collections 目錄

COLLECTIONS_DIR="collections"

# 檢查 collections 目錄是否存在且有子資料夾
if [ ! -d "$COLLECTIONS_DIR" ] || [ -z "$(find "$COLLECTIONS_DIR" -mindepth 1 -type d)" ]; then
    echo "\033[31m[錯誤]\033[0m collections 目錄下沒有任何已建立的 collection！"
    echo "請先執行 scripts/quickstart.sh 建立範例 collection，再啟動 FastAPI 服務。"
    exit 1
fi

# 啟動 FastAPI 服務
echo "\033[32m[提示]\033[0m 偵測到 collections 目錄下已有 collection，啟動 FastAPI 服務..."
docker compose --profile api up -d