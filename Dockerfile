# 使用多階段建構來減少最終映像大小
FROM python:3.11-slim

WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安裝依賴
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式碼
COPY . .

# 創建必要的目錄
RUN mkdir -p collections data logs

# 設置環境變數
ENV PYTHONUNBUFFERED=1
ENV PATH=/root/.local/bin:$PATH

# 健康檢查（簡化版）
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# 創建啟動腳本
RUN echo '#!/bin/bash\n\
if [ "$1" = "api" ]; then\n\
    echo "🚀 啟動 API 服務..."\n\
    exec uvicorn app:app --host 0.0.0.0 --port 8000\n\
elif [ "$1" = "process-faq" ]; then\n\
    echo "🔄 處理 FAQ 文件..."\n\
    shift\n\
    exec python diskrag.py process "$@"\n\
elif [ "$1" = "index" ]; then\n\
    echo "🔍 建立索引..."\n\
    shift\n\
    exec python diskrag.py index "$@"\n\
elif [ "$1" = "list" ]; then\n\
    echo "📋 列出 collections..."\n\
    exec python diskrag.py list\n\
else\n\
    echo "❌ 未知命令: $1"\n\
    echo "可用命令:"\n\
    echo "  api - 啟動 API 服務"\n\
    echo "  process-faq <csv_file> --collection <name> [--questions] - 處理 FAQ 文件"\n\
    echo "  index <collection_name> - 建立索引"\n\
    echo "  list - 列出所有 collections"\n\
    exit 1\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# 預設命令
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"]