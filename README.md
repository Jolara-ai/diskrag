# DiskRAG - 5分鐘快速開始

DiskRAG 是一個基於 DiskANN 的高性能向量搜索系統，讓您能夠快速建立自己的知識庫並進行智能搜索。

## 🚀 5分鐘快速開始

### 1. 安裝 (1分鐘)

**Linux/macOS:**
```bash
# 設置腳本執行權限
chmod +x scripts/*.sh

# 執行安裝腳本
./scripts/install.sh
```

**Windows (PowerShell):**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\scripts\install.ps1
```

**Windows (Command Prompt):**
```cmd
scripts\install.bat
```

### 2. 配置 (1分鐘)

```bash
# 複製環境變數範例
cp env.example .env

# 編輯 .env 文件，填入您的 OpenAI API 金鑰
# OPENAI_API_KEY=your-api-key-here
```

### 3. 使用 (3分鐘)

#### 基本使用
```bash
# 數據導入
diskrag process data/example.csv --collection my_faq

# 建立索引
diskrag index my_faq

# 立即查詢
diskrag search my_faq "DiskANN 的原理是什麼？"
```

#### FAQ 工作流程 (推薦)
```bash
# 1. 準備FAQ CSV文件 (見下方格式)
# 2. 處理FAQ文件 (自動生成相似問題)
./scripts/process_faq.sh my_manual data/example.csv

# 3. 建立索引
diskrag index my_manual

# 4. FAQ搜索 (自動去重和格式化)
./scripts/search_faq.sh my_manual "EBF7531SBA 這台機器怎麼用？"
```

🎉 **完成！** 您已經成功建立了一個智能搜索系統。

## 📚 支援的數據格式

### FAQ CSV 格式 (推薦)

```csv
id,question,answer,source_file,source_page,source_section,source_image
faq_001,這份使用手冊適用於哪個型號的洗碗機？,適用於 EBF7531SBA 型號的全嵌式洗碗機。,EBF7531SBA_ZH_Manual.pdf,1,封面,images/cover.png
faq_002,如何購買原裝配件？,應訪問 https://www.bosch-home.com/accessories/ 或聯繫當地授權經銷商。,EBF7531SBA_ZH_Manual.pdf,2,配件資訊,
faq_003,8歲以下的青少年可以使用嗎？,不可以，未滿 8 歲的青少年不得使用本機。,EBF7531SBA_ZH_Manual.pdf,3,安全資訊,images/safety.png
```

**FAQ 工作流程優勢：**
- ✅ **自動問題生成**：基於原始問題生成多個相似問法
- ✅ **結果去重**：自動去除重複答案
- ✅ **完整出處**：保留所有來源信息
- ✅ **高召回率**：通過多種問法提高搜索準確性

### 其他格式
- **CSV 文件**：FAQ 格式 (question, answer) 或文章格式 (title, paragraph_text)
- **Markdown 文件**：.md, .markdown
- **Word 文件**：.docx, .doc

## 🔧 常用命令

### 基本操作
```bash
# 處理單個文件
diskrag process data/my_file.csv --collection my_collection

# 處理整個目錄
diskrag process-dir data --prefix docs

# 建立索引
diskrag index my_collection

# 搜索
diskrag search my_collection "您的問題"

# FAQ搜索 (自動去重和格式化)
diskrag search my_collection "您的問題" --faq

# 列出所有 collections
diskrag list

# 刪除 collection
diskrag delete my_collection
```

### 高級操作
```bash
# 合併多個 collections
diskrag merge collection1 collection2 --target merged_collection

# 修復損壞的索引
diskrag doctor my_collection

# 高品質索引建立
diskrag index my_collection --target-quality high
```

## 🎯 FAQ 工作流程詳解

### Phase 1: 數據準備

工程師只需要填寫一個簡單的 CSV 文件：

```csv
id,question,answer,source_file,source_page,source_section,source_image
faq_001,這份使用手冊適用於哪個型號的洗碗機？,適用於 EBF7531SBA 型號的全嵌式洗碗機。,EBF7531SBA_ZH_Manual.pdf,1,封面,images/cover.png
```
欄位說明：

id (可選但建議): 每個問答對的唯一標識符。如果留空，系統可以自動生成。
question (必需): 標準、最典型的問題。
answer (必需): 對應的答案。
source_file (可選): 來源檔案名稱，如 EBF7531SBA_ZH_Manual.pdf。
source_page (可選): 來源頁碼。
source_section (可選): 來源章節標題，如「安全資訊」。
source_image (可選): 相關圖片的路徑或 URL。


### Phase 2: 處理與索引

```bash
# 處理FAQ文件 (自動生成相似問題)
diskrag process faq_data.csv --collection my_manual --questions

# 建立索引
diskrag index my_manual
```

**內部處理流程：**
1. 讀取 CSV 文件
2. 為每個原始問題生成多個相似問題
3. 建立向量和元數據
4. 所有問題共享同一份答案和出處信息

### Phase 3: 查詢與呈現

```bash
# FAQ搜索 (自動去重和格式化)
diskrag search my_manual "EBF7531SBA 這台機器怎麼用？" --faq
```

**查詢流程：**
1. 向量化用戶查詢
2. 在索引中找到最相似的問題
3. 根據 `qa_id` 自動去重
4. 返回格式化的結果

### API 回應範例

```json
{
  "results": [
    {
      "answer": "適用於 EBF7531SBA 型號的全嵌式洗碗機。",
      "matched_question": "EBF7531SBA 的使用指南",
      "original_question": "這份使用手冊適用於哪個型號的洗碗機？",
      "similarity": 0.95,
      "source": {
        "file": "EBF7531SBA_ZH_Manual.pdf",
        "page": 1,
        "section": "封面",
        "image": "images/cover.png"
      }
    }
  ],
  "timing": {
    "embedding_time": 0.123,
    "search_time": 0.456,
    "total_time": 0.579
  },
  "stats": {
    "search_type": "faq_pq_accelerated",
    "total_results_before_dedup": 15,
    "total_results_after_dedup": 5,
    "duplicates_removed": 10
  }
}
```

## ⚙️ 配置選項

### 品質等級
- `fast`: 快速建立，適合大規模數據
- `balanced`: 平衡精度和速度 (預設)
- `high`: 高精度，適合對準確度要求高的場景

### 環境變數
- `OPENAI_API_KEY`: OpenAI API 金鑰 (必需)
- `VERTEX_PROJECT_ID`: Google Vertex AI 專案 ID (可選)

## 📖 範例

### FAQ 數據處理
```bash
# 1. 準備 FAQ CSV 文件
echo "question,answer" > faq.csv
echo "什麼是 DiskANN？,DiskANN 是一個可擴展的近似最近鄰搜索算法..." >> faq.csv

# 2. 處理並建立索引
diskrag process faq.csv --collection faq_db --questions
diskrag index faq_db

# 3. 搜索
diskrag search faq_db "DiskANN 是什麼？"
```

### 文檔處理
```bash
# 1. 處理 Markdown 文件
diskrag process data/example.md --collection manual

# 2. 建立索引
diskrag index manual

# 3. 搜索
diskrag search manual "如何配置系統？"
```

## 🆘 常見問題

**Q: 如何獲取 OpenAI API 金鑰？**
A: 訪問 [OpenAI Platform](https://platform.openai.com/api-keys) 創建 API 金鑰

**Q: 支援哪些文件格式？**
A: CSV (FAQ/文章格式)、Markdown (.md)、Word (.docx)

**Q: 如何提高搜索準確度？**
A: 使用 `--target-quality high` 建立高品質索引

**Q: 可以處理多大規模的數據？**
A: DiskRAG 基於 DiskANN，可以處理百萬級別的向量數據

## 🔗 更多資源

- [配置說明](configs/README.md) - 詳細的配置選項
- [工具腳本](scripts/tools/README.md) - 開發和調試工具
- [API 服務](app.py) - 啟動 Web API 服務

---

**DiskRAG** - 讓知識搜索變得簡單高效 🚀

歡迎提交Issue和Pull Request！

## �� 許可證

MIT License
