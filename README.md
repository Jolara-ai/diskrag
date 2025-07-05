# DiskRAG - 快速入門指南

DiskRAG 是一套基於 DiskANN 的高效向量搜尋系統，讓你能輕鬆處理、索引與搜尋各類文件。

---


## 一鍵安裝

你可以直接執行安裝腳本，或手動安裝：

### 方法一：自動安裝（推薦）
```bash
chmod +x scripts/install.sh  # 若遇到權限問題請先執行
bash scripts/install.sh
```

### 方法二：手動安裝
```bash
# 1. 建立虛擬環境
python3 -m venv venv
source venv/bin/activate  # Windows 請用 venv\Scripts\activate

# 2. 安裝相依套件
pip install --upgrade pip
pip install -r requirements.txt

# 3. 建立必要目錄
mkdir -p data collections logs

# 4. 設定 OpenAI API 金鑰
export OPENAI_API_KEY='your-api-key'
```

---
## 快速開始

```bash
chmod +x scripts/quickstart.sh
bash scripts/quickstart.sh
```


### 1. 處理檔案

```bash
# 處理 FAQ CSV 檔案
python diskrag.py process data/faq.csv --collection faq

# 處理 FAQ 並自動產生相似問題
python diskrag.py process data/faq.csv --collection faq --questions

# 處理 Markdown 檔案
python diskrag.py process data/manual.md --collection manual
```

### 2. 建立索引
```bash
python diskrag.py index faq
```

### 3. 搜尋
```bash
python diskrag.py search faq "如何使用系統"
```

### 4. 管理 Collections
```bash
# 列出所有 collections
python diskrag.py list

# 刪除 collection
python diskrag.py delete faq
```

---

## 檔案格式範例

### FAQ CSV 格式
```csv
question,answer
如何開始？,請先登入系統。
支援哪些格式？,支援 CSV、Markdown 和 Word。
```

### 文章 CSV 格式
```csv
title,paragraph_text,section
系統簡介,這是一個搜尋系統,第一章
安裝指南,請按照步驟安裝,第二章
```

---

## Docker 用法

```bash
chmod +x scripts/run_api_with_check.sh
bash scripts/run_api_with_check.sh
```

---

## 常見問題

**Q: 如何更新現有的 collection？**
A: 刪除後重新建立：
```bash
python diskrag.py delete faq
python diskrag.py process data/faq_new.csv --collection faq
python diskrag.py index faq
```

**Q: 支援哪些檔案格式？**
A: CSV (.csv)、Markdown (.md, .markdown)、Word (.docx, .doc)

**Q: 如何調整文字分塊大小？**
A: 修改 config.yaml 中的 chunk.size 參數（預設 300 字）

**Q: OpenAI API 金鑰怎麼設定？**
A: 請在終端機執行：
```bash
export OPENAI_API_KEY='your-api-key'
```
或建立 .env 檔案：
```bash
echo "OPENAI_API_KEY=your-api-key" > .env
```

---

## 進階說明

- 你也可以用 `bash scripts/quickstart.sh` 互動式體驗全流程。
- 進階參數與自訂設定請參考 config.yaml。
- 支援互動式介面：
  ```bash
  python interactive.py
  ```

---

## 🧹 刪除虛擬環境的方法

請根據你的作業系統執行：

✅ **在 macOS / Linux / WSL：**
```bash
rm -rf venv
```

✅ **在 Windows（PowerShell）：**
```powershell
Remove-Item -Recurse -Force venv
```

✅ **在 Windows（CMD）：**
```cmd
rmdir /s /q venv
```

🔐 **小提醒：**
- 刪除前不需「停用」虛擬環境，它只是個資料夾。
- 如果你改用 `.venv` 或其他資料夾名稱，也只要改對刪除的資料夾名稱即可。

## 授權

MIT