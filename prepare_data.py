#!/usr/bin/env python3
"""
準備 DiskANN 搜索所需的資料檔案

使用方法：
1. 複製此腳本到您的專案目錄
2. 修改配置參數（見下方 CONFIG 部分）
3. 運行腳本：python prepare_data.py

配置說明：
- manual_dir: 手冊檔案目錄，支援 .md 和 .docx 檔案
- chunks_dir: 文字塊和元資料輸出目錄
- vectors_dir: 向量資料輸出目錄
- embedding_model: OpenAI embedding 模型名稱
- batch_size: 批量處理大小
- min_text_length: 最小文字長度
- max_text_length: 最大文字長度
"""

import os
import sys
from pathlib import Path

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from prepare_md_chunks import DocumentProcessor, MIN_TEXT_LENGTH, MAX_TEXT_LENGTH

# 配置參數
CONFIG = {
    # 輸入輸出路徑
    "manual_dir": "data/manual",  # 手冊檔案目錄
    "chunks_dir": "data/chunks",  # 文字塊輸出目錄
    "vectors_dir": "data/vectors",  # 向量資料輸出目錄
    
    # 模型參數
    "embedding_model": "text-embedding-3-small",  # OpenAI embedding 模型
    "batch_size": 50,  # 批量處理大小
    
    # 文字處理參數
    "min_text_length": MIN_TEXT_LENGTH,  # 最小文字長度
    "max_text_length": MAX_TEXT_LENGTH,  # 最大文字長度
}

def check_environment():
    """檢查環境配置"""
    # 檢查 OpenAI API 金鑰
    if not os.getenv("OPENAI_API_KEY"):
        print("錯誤: 未設定 OPENAI_API_KEY 環境變數")
        print("請設定環境變數：")
        print("export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)
    
    # 檢查輸入目錄
    manual_dir = Path(CONFIG["manual_dir"])
    if not manual_dir.exists():
        print(f"錯誤: 手冊目錄不存在: {manual_dir}")
        print("請創建目錄或修改 CONFIG['manual_dir'] 為正確的路徑")
        sys.exit(1)
    
    # 檢查輸入檔案
    files = list(manual_dir.glob("*"))
    if not files:
        print(f"錯誤: 手冊目錄為空: {manual_dir}")
        print("請將手冊檔案（.md 或 .docx）放入該目錄")
        sys.exit(1)
    
    # 顯示找到的檔案
    print("\n找到以下檔案：")
    for file in sorted(files, key=lambda x: x.name.lower()):
        if file.suffix in ['.md', '.docx']:
            print(f"- {file.name}")
    
    # 確認繼續
    response = input("\n是否繼續處理這些檔案？(y/n): ")
    if response.lower() != 'y':
        print("已取消處理")
        sys.exit(0)

def main():
    """主函數"""
    print("=== DiskANN 資料準備工具 ===")
    
    # 檢查環境
    check_environment()
    
    # 創建處理器
    processor = DocumentProcessor(
        manual_dir=CONFIG["manual_dir"],
        chunks_dir=CONFIG["chunks_dir"],
        vectors_dir=CONFIG["vectors_dir"],
        embedding_model=CONFIG["embedding_model"],
        batch_size=CONFIG["batch_size"]
    )
    
    # 處理文檔
    print("\n開始處理文檔...")
    try:
        processor.process_all_documents()
        print("\n處理完成！")
        print(f"輸出目錄：")
        print(f"- 文字塊和元資料：{CONFIG['chunks_dir']}")
        print(f"- 向量資料：{CONFIG['vectors_dir']}")
    except Exception as e:
        print(f"\n處理過程中出錯：{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 