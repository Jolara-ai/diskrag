# 使用多階段建構來減少最終映像大小
FROM python:3.11-slim

WORKDIR /app

# 安裝依賴
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式碼
COPY . .

# 創建必要的目錄
RUN mkdir -p collections data

# 設置環境變數
ENV PYTHONUNBUFFERED=1
ENV PATH=/root/.local/bin:$PATH

# 健康檢查（簡化版）
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# 預設命令
ENTRYPOINT ["python", "diskrag.py"]