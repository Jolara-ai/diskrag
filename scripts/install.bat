@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ╔══════════════════════════════════════╗
echo ║      DiskRAG Windows 安裝腳本       ║
echo ╚══════════════════════════════════════╝
echo.

REM 檢查 Python 版本
echo 檢查 Python 版本...
python --version >nul 2>&1
if errorlevel 1 (
    echo 錯誤: 未找到 Python
    echo 請先安裝 Python 3.8 或更新版本
    echo 下載地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo 找到 Python !PYTHON_VERSION!

REM 建立虛擬環境
echo.
echo 建立虛擬環境...
if exist "venv" (
    echo 虛擬環境已存在，正在移除...
    rmdir /s /q "venv"
)
python -m venv venv

REM 啟用虛擬環境
echo 啟用虛擬環境...
call venv\Scripts\activate.bat

REM 升級 pip
echo.
echo 升級 pip...
python -m pip install --upgrade pip

REM 安裝依賴
echo.
echo 安裝依賴套件...
if exist "requirements.txt" (
    pip install -r requirements.txt
) else (
    echo 警告: 未找到 requirements.txt 檔案
)

REM 建立必要目錄
echo.
echo 建立目錄結構...
if not exist "data" mkdir data
if not exist "collections" mkdir collections
if not exist "logs" mkdir logs

REM 建立預設設定檔
if not exist "config.yaml" (
    echo 建立預設設定檔...
    (
        echo # DiskRAG 設定檔
        echo collection: "default"
        echo.
        echo embedding:
        echo   provider: "openai"
        echo   model: "text-embedding-3-small"
        echo   max_retries: 3
        echo   retry_delay: 2
        echo.
        echo question_generation:
        echo   enabled: false
        echo   provider: "openai"
        echo   model: "gpt-4o-mini"
        echo   max_questions: 5
        echo   temperature: 0.7
        echo.
        echo chunk:
        echo   size: 300
        echo   overlap: 50
        echo   min_size: 50
        echo.
        echo output:
        echo   format: "parquet"
        echo   compression: "snappy"
        echo.
        echo # 新增索引參數區塊
        echo index:
        echo   R: 32
        echo   L: 64
        echo   alpha: 1.2
    ) > config.yaml
    echo 已建立 config.yaml
)

REM 建立範例檔案
if not exist "data\example.csv" (
    echo 建立範例檔案...
    (
        echo question,answer
        echo 什麼是 DiskANN？,DiskANN 是一個可擴展的近似最近鄰搜索算法，專門設計用於處理大規模向量數據集，特別是當數據集大小超過記憶體容量時。
        echo DiskANN 解決了什麼問題？,DiskANN 解決了大規模向量搜索中的記憶體限制問題，允許在磁碟上建立和查詢十億級別的向量索引，同時保持高精度和高效能。
        echo DiskANN 的核心原理是什麼？,DiskANN 結合了圖形導航搜索和分層索引結構，將熱點數據保存在記憶體中，冷數據存儲在磁碟上，通過智能的數據分層來優化查詢效能。
        echo 什麼是 Vamana 圖？,Vamana 是 DiskANN 使用的圖形結構，它是一個度數受限的圖，每個節點的鄰居數量有上限，這樣可以控制記憶體使用量並提高搜索效率。
        echo DiskANN 相比於其他 ANN 算法有什麼優勢？,DiskANN 的主要優勢包括：1^) 可處理超大規模數據集 2^) 記憶體使用量可控 3^) 查詢延遲穩定 4^) 支援動態更新 5^) 在精度和效能間有良好平衡。
    ) > data\example.csv
    echo 已建立 data\example.csv
)

REM 建立 Windows 批次檔案
echo.
echo 建立 Windows 批次檔案...
(
    echo @echo off
    echo cd /d "%%~dp0"
    echo call venv\Scripts\activate.bat
    echo python diskrag.py %%*
) > diskrag.bat
echo 已建立 diskrag.bat

REM 檢查 OPENAI_API_KEY
echo.
set "API_KEY=%OPENAI_API_KEY%"
if "%API_KEY%"=="" (
    echo ⚠️  注意: 未設定 OPENAI_API_KEY
    echo.
    echo 請設定環境變數:
    echo set OPENAI_API_KEY=your-api-key
    echo.
    echo 或建立 .env 檔案:
    echo echo OPENAI_API_KEY=your-api-key ^> .env
) else (
    echo ✓ 已設定 OPENAI_API_KEY
)

REM 完成訊息
echo.
echo ╔══════════════════════════════════════╗
echo ║         安裝完成！                   ║
echo ╚══════════════════════════════════════╝
echo.
echo 使用方式:
echo   .\diskrag.bat process data\example.csv --collection example
echo   .\diskrag.bat index example
echo   .\diskrag.bat search example "什麼是 DiskRAG"
echo.
echo 或手動啟用虛擬環境後使用:
echo   venv\Scripts\activate.bat
echo   python diskrag.py --help
echo.
pause 