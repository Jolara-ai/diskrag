# DiskRAG 配置說明

## 配置文件結構

DiskRAG 使用單一的配置文件 `config.yaml` 來管理所有設定。

### 主要配置文件

- **`config.yaml`** (根目錄) - 主要的用戶配置文件
- **`config.yaml.example`** (根目錄) - 完整的配置模板，包含詳細註釋

### 範例配置文件

- **`configs/faq_config.example.yaml`** - FAQ 處理的範例配置

## 快速開始

1. **複製配置模板**：
   ```bash
   cp config.yaml.example config.yaml
   ```

2. **修改基本設定**：
   ```yaml
   collection: "my_collection"  # 修改為您的 collection 名稱
   ```

3. **設定 API 金鑰**：
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## 配置選項說明

### 基本設定

- `collection`: collection 名稱，用於識別和管理數據集

### Embedding 設定

- `provider`: embedding 提供商 ("openai" 或 "vertex")
- `model`: embedding 模型名稱
- `api_key`: API 金鑰（建議通過環境變數設置）
- `max_retries`: API 調用失敗時的最大重試次數
- `retry_delay`: 重試之間的等待時間（秒）

### 問題生成設定

- `enabled`: 是否啟用問題生成（FAQ 專用）
- `provider`: LLM 提供商
- `model`: LLM 模型名稱
- `max_questions`: 每個問答對生成的最大問題數
- `temperature`: 生成溫度，控制創造性（0.0-1.0）

### 分塊設定

- `size`: 每個文本塊的最大字符數
- `overlap`: 相鄰文本塊之間的重疊字符數
- `min_size`: 文本塊的最小字符數

### 輸出設定

- `format`: 輸出文件格式（目前只支持 "parquet"）
- `compression`: 壓縮方式（"snappy", "gzip", "none"）

### 索引設定

- `R`: Vamana 圖的最大度數
- `L`: 圖建立時的搜索列表大小
- `alpha`: 圖建立時的剪枝參數

## 環境變數支持

以下配置可以通過環境變數設置：

- `OPENAI_API_KEY`: OpenAI API 金鑰
- `VERTEX_PROJECT_ID`: Vertex AI 專案 ID
- `COLLECTION_NAME`: collection 名稱

## 範例配置

### 基本配置
```yaml
collection: "my_docs"
embedding:
  provider: "openai"
  model: "text-embedding-3-small"
question_generation:
  enabled: false
```

### FAQ 配置
```yaml
collection: "faq_database"
question_generation:
  enabled: true
  max_questions: 5
  temperature: 0.7
```

### 高精度配置
```yaml
collection: "high_precision"
embedding:
  model: "text-embedding-3-large"
chunk:
  size: 500
  overlap: 100
index:
  R: 64
  L: 128
```

## 注意事項

1. **API 金鑰安全**：不要將 API 金鑰直接寫在配置文件中，建議使用環境變數
2. **配置驗證**：DiskRAG 會在啟動時驗證配置文件的有效性
3. **備份配置**：修改配置前建議備份原始文件
4. **範例文件**：參考 `config.yaml.example` 了解所有可用的配置選項 