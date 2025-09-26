# DiskRAG 簡化使用指南

## 🚀 快速開始

### 1. 環境安裝
```bash
# 一鍵安裝
./scripts/install.sh
```

### 2. 設置API金鑰
```bash
# 複製環境變數示例
cp .env.example .env

# 編輯 .env 文件，填入您的 OpenAI API 金鑰
# OPENAI_API_KEY=your-api-key-here
```

### 3. 使用FAQ工作流程 (推薦)

#### 處理FAQ數據
```bash
# 使用示例FAQ文件
./scripts/process_faq.sh my_manual examples/faq_data.csv

# 或使用自己的CSV文件
./scripts/process_faq.sh my_collection data/my_faq.csv
```

#### 搜索測試
```bash
# 測試搜索
./scripts/search_faq.sh my_manual "EBF7531SBA 這台機器怎麼用？"
```

#### 啟動API服務
```bash
# 啟動FastAPI服務
./scripts/run_api.sh
```

## 📁 腳本說明

### 核心腳本

1. **`install.sh`** - 一鍵安裝腳本
   - 安裝Python依賴
   - 創建虛擬環境
   - 建立目錄結構
   - 創建配置文件

2. **`check_env.sh`** - 環境檢查腳本
   - 檢查虛擬環境
   - 檢查配置文件
   - 檢查環境變數
   - 創建必要目錄

3. **`process_faq.sh`** - FAQ處理腳本
   - 用法: `./scripts/process_faq.sh <collection_name> <csv_file>`
   - 自動生成相似問題
   - 建立向量和索引
   - 完整的錯誤處理

4. **`search_faq.sh`** - FAQ搜索腳本
   - 用法: `./scripts/search_faq.sh <collection_name> <query>`
   - 自動去重和格式化
   - 顯示完整結果

5. **`run_api.sh`** - API服務啟動腳本
   - 檢查環境變數
   - 檢查Docker和Docker Compose
   - 啟動FastAPI服務
   - 顯示使用示例

### 輔助腳本

- **`quickstart.sh`** - 快速開始腳本
- **`test_faq_workflow.py`** - FAQ工作流程測試腳本

## 📊 FAQ CSV格式

```csv
id,question,answer,source_file,source_page,source_section,source_image
faq_001,這份使用手冊適用於哪個型號的洗碗機？,適用於 EBF7531SBA 型號的全嵌式洗碗機。,EBF7531SBA_ZH_Manual.pdf,1,封面,images/cover.png
faq_002,如何購買原裝配件？,應訪問 https://www.bosch-home.com/accessories/ 或聯繫當地授權經銷商。,EBF7531SBA_ZH_Manual.pdf,2,配件資訊,
faq_003,8歲以下的青少年可以使用嗎？,不可以，未滿 8 歲的青少年不得使用本機。,EBF7531SBA_ZH_Manual.pdf,3,安全資訊,images/safety.png
```

## 🌐 API使用

### 啟動服務
```bash
./scripts/run_api.sh
```

### API端點

- **FAQ搜索**: `POST /faq-search`
- **普通搜索**: `POST /search`
- **健康檢查**: `GET /health`
- **Collections**: `GET /collections`

### 使用示例

```bash
# FAQ搜索
curl -X POST 'http://localhost:8000/faq-search' \
  -H 'Content-Type: application/json' \
  -d '{
    "collection": "my_manual",
    "query": "EBF7531SBA 這台機器怎麼用？",
    "top_k": 5
  }'

# 普通搜索
curl -X POST 'http://localhost:8000/search' \
  -H 'Content-Type: application/json' \
  -d '{
    "collection": "my_manual",
    "query": "你的問題",
    "top_k": 5
  }'

# 查看所有collections
curl 'http://localhost:8000/collections'
```

## 🔧 常用命令

### FAQ工作流程 (推薦)
```bash
# 處理FAQ文件
./scripts/process_faq.sh <collection_name> <csv_file>

# 搜索FAQ
./scripts/search_faq.sh <collection_name> <query>

# 啟動API服務
./scripts/run_api.sh
```

### 傳統工作流程
```bash
# 處理文件
python diskrag.py process <file> --collection <name>

# 建立索引
python diskrag.py index <collection_name>

# 搜索
python diskrag.py search <collection_name> <query>

# 列出所有collections
python diskrag.py list
```

## 🆘 故障排除

### 常見問題

1. **環境變數未設置**
   ```bash
   # 設置環境變數
   export OPENAI_API_KEY='your-api-key'
   ```

2. **虛擬環境未激活**
   ```bash
   # 激活虛擬環境
   source venv/bin/activate  # Linux/macOS
   source venv/Scripts/activate  # Windows
   ```

3. **Docker未安裝**
   - 安裝Docker: https://docs.docker.com/get-docker/
   - 安裝Docker Compose: https://docs.docker.com/compose/install/

4. **CSV文件格式錯誤**
   - 確保包含必要的列: `question`, `answer`
   - 檢查CSV文件編碼是否為UTF-8
   - 確保沒有特殊字符

### 獲取幫助

- 查看完整文檔: `README.md`
- 查看工作流程文檔: `docs/FAQ_WORKFLOW.md`
- 運行測試: `python scripts/test_faq_workflow.py`
- 查看快速開始指南: `README_QUICKSTART.md`

## 📈 性能優化

### 搜索性能
- 使用PQ加速搜索
- 結果自動去重
- 線程安全設計

### 內存管理
- 高效的元數據結構
- 向量索引優化
- 自動清理機制

## 🎯 最佳實踐

1. **數據準備**
   - 使用標準FAQ CSV格式
   - 確保問題清晰、具體
   - 提供完整、準確的答案

2. **環境管理**
   - 定期更新依賴
   - 監控磁盤空間
   - 備份重要數據

3. **API使用**
   - 使用適當的top_k值
   - 監控API調用頻率
   - 處理錯誤響應

4. **性能監控**
   - 定期檢查搜索性能
   - 監控內存使用
   - 優化搜索參數 