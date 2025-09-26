# DiskRAG Windows 安裝指南

本目錄包含適用於 Windows 系統的 DiskRAG 安裝腳本。

## 安裝腳本說明

### 1. PowerShell 腳本 (推薦)
- **檔案**: `install.ps1`
- **特點**: 功能最完整，支援彩色輸出，錯誤處理較好
- **使用方式**: 
  ```powershell
  .\install.ps1
  ```

### 2. 批次檔案
- **檔案**: `install.bat`
- **特點**: 簡單易用，適合不熟悉 PowerShell 的用戶
- **使用方式**: 
  ```cmd
  install.bat
  ```

## 安裝前準備

### 1. 安裝 Python
- 下載並安裝 Python 3.8 或更新版本
- 下載地址: https://www.python.org/downloads/
- 安裝時請勾選 "Add Python to PATH"

### 2. PowerShell 執行政策 (僅適用 PowerShell 腳本)
如果使用 PowerShell 腳本遇到執行政策問題，請執行：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 安裝步驟

### 方法一：使用 PowerShell 腳本 (推薦)

1. 開啟 PowerShell (以管理員身份執行)
2. 切換到 DiskRAG 專案目錄
3. 執行安裝腳本：
   ```powershell
   .\scripts\install.ps1
   ```

### 方法二：使用批次檔案

1. 開啟命令提示字元 (CMD)
2. 切換到 DiskRAG 專案目錄
3. 執行安裝腳本：
   ```cmd
   scripts\install.bat
   ```

## 安裝完成後

安裝腳本會自動完成以下工作：

1. ✅ 檢查 Python 版本
2. ✅ 建立虛擬環境
3. ✅ 安裝依賴套件
4. ✅ 建立必要目錄 (`data`, `collections`, `logs`)
5. ✅ 建立預設設定檔 (`config.yaml`)
6. ✅ 建立範例資料 (`data/example.csv`)
7. ✅ 建立啟動腳本 (`diskrag.bat`, `diskrag.ps1`)

## 使用方式

### 方法一：使用批次檔案
```cmd
.\diskrag.bat process data\example.csv --collection example
.\diskrag.bat index example
.\diskrag.bat search example "什麼是 DiskRAG"
```

### 方法二：使用 PowerShell 腳本
```powershell
.\diskrag.ps1 process data\example.csv --collection example
.\diskrag.ps1 index example
.\diskrag.ps1 search example "什麼是 DiskRAG"
```

### 方法三：手動啟用虛擬環境
```cmd
venv\Scripts\activate.bat
python diskrag.py --help
```

## 設定 API Key

安裝完成後，請設定您的 OpenAI API Key：

### 方法一：設定環境變數
```cmd
set OPENAI_API_KEY=your-api-key
```

### 方法二：建立 .env 檔案
```cmd
echo OPENAI_API_KEY=your-api-key > .env
```

## 常見問題

### Q: 遇到 "無法載入檔案" 錯誤
**A**: 這是 PowerShell 執行政策限制，請執行：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q: Python 未找到
**A**: 請確認 Python 已正確安裝並加入 PATH，或重新安裝 Python 時勾選 "Add Python to PATH"

### Q: 虛擬環境啟動失敗
**A**: 請確認 Python 版本為 3.8 或更新版本，並重新執行安裝腳本

### Q: 依賴套件安裝失敗
**A**: 請檢查網路連線，或嘗試使用國內鏡像源：
```cmd
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

## 檔案結構

安裝完成後，您的專案結構應該如下：
```
diskrag/
├── venv/                    # Python 虛擬環境
├── data/                    # 資料目錄
│   └── example.csv         # 範例資料
├── collections/            # 集合目錄
├── logs/                   # 日誌目錄
├── config.yaml            # 設定檔
├── diskrag.bat            # Windows 批次啟動腳本
├── diskrag.ps1            # PowerShell 啟動腳本
└── diskrag.py             # 主程式
```

## 支援

如果遇到問題，請：
1. 檢查 Python 版本是否為 3.8+
2. 確認網路連線正常
3. 查看錯誤訊息並參考常見問題
4. 在 GitHub Issues 中回報問題 