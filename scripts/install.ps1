# DiskRAG Windows 安裝腳本 (PowerShell)

# 設定錯誤處理
$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "      DiskRAG Windows 安裝程式" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# 檢查 PowerShell 執行政策
Write-Host "檢查 PowerShell 執行政策..." -ForegroundColor Yellow
$executionPolicy = Get-ExecutionPolicy
if ($executionPolicy -eq "Restricted") {
    Write-Host "警告: PowerShell 執行政策為 Restricted" -ForegroundColor Red
    Write-Host "請以管理員身份執行以下命令:" -ForegroundColor Yellow
    Write-Host "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Green
    Write-Host "然後重新執行此腳本" -ForegroundColor Yellow
    exit 1
}

# 檢查 Python 版本
Write-Host "檢查 Python 版本..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python 未找到"
    }
    Write-Host "找到 $pythonVersion" -ForegroundColor Green
    
    # 檢查 Python 版本是否 >= 3.8
    $versionMatch = $pythonVersion -match "Python (\d+)\.(\d+)"
    if ($versionMatch) {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 8)) {
            Write-Host "錯誤: 需要 Python 3.8 或更高版本" -ForegroundColor Red
            exit 1
        }
    }
} catch {
    Write-Host "錯誤: 未找到 Python" -ForegroundColor Red
    Write-Host "請先安裝 Python 3.8 或更高版本" -ForegroundColor Yellow
    Write-Host "下載: https://www.python.org/downloads/" -ForegroundColor Cyan
    exit 1
}

# 建立虛擬環境
Write-Host ""
Write-Host "建立虛擬環境..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "虛擬環境已存在，正在移除..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "venv"
}
python -m venv venv

# 啟用虛擬環境
Write-Host "啟用虛擬環境..." -ForegroundColor Yellow
$activateScript = "venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
} else {
    Write-Host "錯誤: 找不到虛擬環境啟用腳本" -ForegroundColor Red
    exit 1
}

# 升級 pip
Write-Host ""
Write-Host "升級 pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# 安裝依賴
Write-Host ""
Write-Host "安裝依賴套件..." -ForegroundColor Yellow
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
} else {
    Write-Host "警告: 未找到 requirements.txt" -ForegroundColor Yellow
}

# 建立必要目錄
Write-Host ""
Write-Host "建立目錄結構..." -ForegroundColor Yellow
$directories = @("data", "collections", "logs")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "已建立目錄: $dir" -ForegroundColor Green
    }
}

# 建立預設設定檔
if (-not (Test-Path "config.yaml")) {
    Write-Host "建立預設設定檔..." -ForegroundColor Yellow
    $configContent = @"
# DiskRAG 設定檔
collection: "default"

embedding:
  provider: "openai"
  model: "text-embedding-3-small"
  max_retries: 3
  retry_delay: 2

question_generation:
  enabled: false
  provider: "openai"
  model: "gpt-4o-mini"
  max_questions: 5
  temperature: 0.7

chunk:
  size: 300
  overlap: 50
  min_size: 50

output:
  format: "parquet"
  compression: "snappy"

# 索引參數
index:
  R: 32
  L: 64
  alpha: 1.2
"@
    $configContent | Out-File -FilePath "config.yaml" -Encoding UTF8
    Write-Host "已建立 config.yaml" -ForegroundColor Green
}

# 建立範例檔案
if (-not (Test-Path "data\example.csv")) {
    Write-Host "建立範例檔案..." -ForegroundColor Yellow
    $exampleContent = @"
question,answer
什麼是 DiskANN？,DiskANN 是一個可擴展的近似最近鄰搜索算法，專門設計用於處理大規模向量數據集，特別是當數據集大小超過記憶體容量時。
DiskANN 解決了什麼問題？,DiskANN 解決了大規模向量搜索中的記憶體限制問題，允許在磁碟上建立和查詢十億級別的向量索引，同時保持高精度和高效能。
DiskANN 的核心原理是什麼？,DiskANN 結合了圖形導航搜索和分層索引結構，將熱點數據保存在記憶體中，冷數據存儲在磁碟上，通過智能的數據分層來優化查詢效能。
什麼是 Vamana 圖？,Vamana 是 DiskANN 使用的圖形結構，它是一個度數受限的圖，每個節點的鄰居數量有上限，這樣可以控制記憶體使用量並提高搜索效率。
DiskANN 相比於其他 ANN 算法有什麼優勢？,DiskANN 的主要優勢包括：1) 可處理超大規模數據集 2) 記憶體使用量可控 3) 查詢延遲穩定 4) 支援動態更新 5) 在精度和效能間有良好平衡。
"@
    $exampleContent | Out-File -FilePath "data\example.csv" -Encoding UTF8
    Write-Host "已建立 data\example.csv" -ForegroundColor Green
}

# 建立 Windows 批次檔案
Write-Host ""
Write-Host "建立 Windows 批次檔案..." -ForegroundColor Yellow
$batchContent = @"
@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python diskrag.py %*
"@
$batchContent | Out-File -FilePath "diskrag.bat" -Encoding ASCII
Write-Host "已建立 diskrag.bat" -ForegroundColor Green

# 建立 PowerShell 腳本
$psScriptContent = @"
# DiskRAG PowerShell 啟動腳本
`$scriptDir = Split-Path -Parent `$MyInvocation.MyCommand.Path
Set-Location `$scriptDir
& "`$scriptDir\venv\Scripts\Activate.ps1"
python diskrag.py `$args
"@
$psScriptContent | Out-File -FilePath "diskrag.ps1" -Encoding UTF8
Write-Host "已建立 diskrag.ps1" -ForegroundColor Green

# 檢查 OPENAI_API_KEY
Write-Host ""
$apiKey = [Environment]::GetEnvironmentVariable("OPENAI_API_KEY", "User")
if ([string]::IsNullOrEmpty($apiKey)) {
    Write-Host "警告: 未設定 OPENAI_API_KEY" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "請設定環境變數:" -ForegroundColor Yellow
    Write-Host "[Environment]::SetEnvironmentVariable('OPENAI_API_KEY', 'your-api-key', 'User')" -ForegroundColor Green
    Write-Host ""
    Write-Host "或建立 .env 檔案:" -ForegroundColor Yellow
    Write-Host "New-Item -Path '.env' -ItemType File -Value 'OPENAI_API_KEY=your-api-key'" -ForegroundColor Green
} else {
    Write-Host "✓ 已設定 OPENAI_API_KEY" -ForegroundColor Green
}

# 完成訊息
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "          安裝完成！" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "使用方式:" -ForegroundColor Yellow
Write-Host "  .\diskrag.bat process data\example.csv --collection example" -ForegroundColor Green
Write-Host "  .\diskrag.bat index example" -ForegroundColor Green
Write-Host "  .\diskrag.bat search example '什麼是 DiskRAG'" -ForegroundColor Green
Write-Host ""
Write-Host "或使用 PowerShell 腳本:" -ForegroundColor Yellow
Write-Host "  .\diskrag.ps1 process data\example.csv --collection example" -ForegroundColor Green
Write-Host ""
Write-Host "或手動啟用虛擬環境:" -ForegroundColor Yellow
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "  python diskrag.py --help" -ForegroundColor Green
Write-Host ""
Write-Host "注意: 如果遇到執行政策問題，請執行:" -ForegroundColor Yellow
Write-Host "  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Green
Write-Host "" 