#!/bin/bash

# DiskRAG 快速開始腳本

set -e

echo "=== DiskRAG 快速開始 ==="
echo

# 檢查環境
if [ ! -f "scripts/check_env.sh" ]; then
    echo "❌ 環境檢查腳本不存在"
    echo "請先執行: ./scripts/install.sh"
    exit 1
fi

# 執行環境檢查
source scripts/check_env.sh

# 建立必要目錄
echo "建立目錄結構..."
mkdir -p data collections examples

# 檢查是否有範例 FAQ 檔案
if [ ! -f "examples/faq_data.csv" ]; then
    echo "建立範例 FAQ 檔案..."
    cat > examples/faq_data.csv << 'EOF'
id,question,answer,source_file,source_page,source_section,source_image
faq_001,這份使用手冊適用於哪個型號的洗碗機？,適用於 EBF7531SBA 型號的全嵌式洗碗機。,EBF7531SBA_ZH_Manual.pdf,1,封面,images/cover.png
faq_002,如何購買原裝配件？,應造訪 https://www.bosch-home.com/accessories/ 或聯絡當地授權經銷商。,EBF7531SBA_ZH_Manual.pdf,2,配件資訊,
faq_003,8歲以下的青少年可以使用嗎？,不可以，未滿 8 歲的青少年不得使用本機。,EBF7531SBA_ZH_Manual.pdf,3,安全資訊,images/safety.png
faq_004,如何設定水質硬度？,在基本設定中，選擇「水質硬度」選項，並從等級 1 到 10 中選擇對應您所在地區的水質硬度。,EBF7531SBA_ZH_Manual.pdf,15,基本設定,images/water_hardness.png
faq_005,洗碗機可以洗滌哪些物品？,可以洗滌：餐具、玻璃杯、碗盤、鍋具等。不可洗滌：木製餐具、鋁製鍋具、塑膠容器等。,EBF7531SBA_ZH_Manual.pdf,8,使用說明,images/items.png
EOF
fi

# 檢查是否有範例檔案
if [ ! -f "data/example.csv" ]; then
    echo "建立範例檔案..."
    cat > data/example.csv << 'EOF'
question,answer
什麼是 DiskANN？,DiskANN 是一個可擴展的近似最近鄰搜尋演算法，專門設計用於處理大規模向量資料集，特別是當資料集大小超過記憶體容量時。
DiskANN 解決了什麼問題？,DiskANN 解決了大規模向量搜尋中的記憶體限制問題，允許在磁碟上建立和查詢十億級別的向量索引，同時保持高精度和高效能。
DiskANN 的核心原理是什麼？,DiskANN 結合了圖形導航搜尋和分層索引結構，將熱點資料保存在記憶體中，冷資料儲存在磁碟上，透過智能的資料分層來優化查詢效能。
什麼是 Vamana 圖？,Vamana 是 DiskANN 使用的圖形結構，它是一個度數受限的圖，每個節點的鄰居數量有上限，這樣可以控制記憶體使用量並提高搜尋效率。
DiskANN 相比於其他 ANN 演算法有什麼優勢？,DiskANN 的主要優勢包括：1) 可處理超大規模資料集 2) 記憶體使用量可控 3) 查詢延遲穩定 4) 支援動態更新 5) 在精度和效能間有良好平衡。
DiskANN 如何處理記憶體不足的問題？,DiskANN 使用分層架構，將經常造訪的節點和邊快取在記憶體中，較少造訪的資料儲存在磁碟上，透過預取和快取策略來減少磁碟 I/O。
EOF
fi

# 顯示使用說明
echo
echo "=== 使用範例 ==="
echo
echo "🎯 FAQ 工作流程 (推薦):"
echo "1. 處理 FAQ 檔案:"
echo "   ./scripts/process_faq.sh my_manual examples/faq_data.csv"
echo
echo "2. 搜索測試:"
echo "   ./scripts/search_faq.sh my_manual 'EBF7531SBA 這台機器怎麼用？'"
echo
echo "3. 啟動 API 服務:"
echo "   ./scripts/run_api.sh"
echo
echo "📚 傳統工作流程:"
echo "1. 處理 FAQ 檔案:"
echo "   python diskrag.py process data/example.csv --collection faq"
echo
echo "2. 處理 Markdown 檔案:"
echo "   python diskrag.py process data/manual.md --collection manual"
echo
echo "3. 建立索引:"
echo "   python diskrag.py index faq"
echo
echo "4. 搜尋:"
echo "   python diskrag.py search faq 'DiskANN 解決了什麼問題?'"
echo
echo "5. 列出所有 collections:"
echo "   python diskrag.py list"
echo

# 詢問是否執行範例
read -p "是否執行 FAQ 工作流程範例？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo
    echo "執行 FAQ 工作流程範例..."
    
    # 檢查是否有 process_faq.sh 腳本
    if [ -f "scripts/process_faq.sh" ]; then
        echo "步驟1: 處理 FAQ 檔案..."
        ./scripts/process_faq.sh demo_manual examples/faq_data.csv
        
        echo
        echo "步驟2: 搜索測試..."
        ./scripts/search_faq.sh demo_manual "EBF7531SBA 這台機器怎麼用？"
        
        echo
        echo "✅ FAQ 工作流程範例完成！"
        echo ""
        echo "下一步："
        echo "  - 啟動 API 服務: ./scripts/run_api.sh"
        echo "  - 查看 collections: python diskrag.py list"
        echo "  - 查看完整文檔: cat README.md"
    else
        echo "❌ process_faq.sh 腳本不存在，使用傳統工作流程..."
        
        # 使用傳統工作流程
        echo "步驟1: 處理 FAQ 檔案..."
        python diskrag.py process data/example.csv --collection demo_faq
        
        echo
        echo "步驟2: 建立索引..."
        python diskrag.py index demo_faq
        
        echo
        echo "步驟3: 搜索測試..."
        python diskrag.py search demo_faq "DiskANN 解決了什麼問題?"
        
        echo
        echo "✅ 傳統工作流程範例完成！"
    fi
else
    echo
    echo "跳過範例執行。"
    echo ""
    echo "您可以手動執行以下命令："
    echo "  ./scripts/process_faq.sh my_manual examples/faq_data.csv"
    echo "  ./scripts/search_faq.sh my_manual 'EBF7531SBA 這台機器怎麼用？'"
    echo "  ./scripts/run_api.sh"
fi

echo
echo "🎉 快速開始完成！"
echo ""
echo "📖 更多資訊："
echo "  - 完整文檔: README.md"
echo "  - 使用說明: README.md"
echo "  - FAQ 工作流程: docs/FAQ_WORKFLOW.md"
echo "  - API 文檔: http://localhost:8000/docs (啟動服務後)"