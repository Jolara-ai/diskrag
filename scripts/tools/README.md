# 工具腳本目錄

這個目錄包含了 DiskRAG 項目的各種工具腳本，主要用於數據準備、驗證和調試。

## 工具列表

### 數據準備工具

- **batch_process.py** - 批量處理腳本，用於處理大量數據文件
- **build_index.py** - 索引構建工具，用於創建和優化向量索引
- **code_extractor.py** - 代碼提取工具，用於從文檔中提取代碼片段
- **prepare_sift_collection.py** - SIFT 數據集準備工具

### 驗證和調試工具

- **gist_academic_validation.py** - 學術數據驗證工具
- **pq_debug_test.py** - PQ (Product Quantization) 調試和測試工具

## 使用方法

這些工具腳本通常用於特定的數據處理任務或調試目的。每個腳本都有其特定的用途和參數。

### 運行示例

```bash
# 批量處理數據
python scripts/tools/batch_process.py

# 構建索引
python scripts/tools/build_index.py

# 驗證學術數據
python scripts/tools/gist_academic_validation.py
```

## 注意事項

- 這些工具腳本主要用於開發和調試目的
- 運行前請確保已安裝所有必要的依賴
- 某些工具可能需要特定的數據格式或配置
- 建議在運行前備份重要數據

## 貢獻

如果您需要添加新的工具腳本，請：
1. 將腳本放在此目錄中
2. 更新此 README 文件
3. 添加適當的文檔和註釋 