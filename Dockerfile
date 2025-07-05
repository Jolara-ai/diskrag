FROM python:3.11-slim

# 設置工作目錄
WORKDIR /app

# 設置環境變數
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.7.1 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_CREATE=false 

# 安裝系統依賴 - 分層安裝以利用緩存
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Poetry - 單獨一層
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# 安裝編譯依賴 - 單獨一層
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 複製依賴文件 - 利用緩存層
COPY pyproject.toml poetry.lock* ./

# 安裝依賴 - 單獨一層
RUN poetry install --no-dev --no-interaction --no-ansi

# 複製應用代碼 - 最後一層
COPY . .

# 創建數據目錄並設置權限
RUN mkdir -p /app/data/manual /app/data/chunks /app/data/vectors /app/collections && \
    chmod -R 755 /app/data /app/collections

# 設置非 root 用戶
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# 暴露端口
EXPOSE 8000

# 啟動命令
CMD ["poetry", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"] 