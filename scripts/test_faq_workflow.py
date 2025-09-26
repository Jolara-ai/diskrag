#!/usr/bin/env python3
"""
FAQ 工作流程測試腳本

這個腳本用於測試完整的FAQ工作流程：
1. 讀取FAQ CSV文件
2. 生成相似問題
3. 建立向量和索引
4. 執行FAQ搜索
5. 驗證結果去重和格式化
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# 添加項目根目錄到Python路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from preprocessing.processor import Preprocessor
from preprocessing.config import PreprocessingConfig, EmbeddingConfig, QuestionGenerationConfig
from search_engine import SearchEngineCorrect

def create_mock_embedding(dimension: int = 1536):
    """創建模擬的embedding函數"""
    def mock_embedding(text: str) -> np.ndarray:
        # 使用文本的簡單哈希作為隨機種子，確保相同文本產生相同向量
        import hashlib
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        np.random.seed(seed)
        return np.random.randn(dimension).astype(np.float32)
    return mock_embedding

def test_faq_workflow():
    """測試完整的FAQ工作流程"""
    print("🔍 開始FAQ工作流程測試...")
    
    # 1. 準備測試數據
    test_csv_content = """id,question,answer,source_file,source_page,source_section,source_image
faq_001,這份使用手冊適用於哪個型號的洗碗機？,適用於 EBF7531SBA 型號的全嵌式洗碗機。,EBF7531SBA_ZH_Manual.pdf,1,封面,images/cover.png
faq_002,如何購買原裝配件？,應訪問 https://www.bosch-home.com/accessories/ 或聯繫當地授權經銷商。,EBF7531SBA_ZH_Manual.pdf,2,配件資訊,
faq_003,8歲以下的青少年可以使用嗎？,不可以，未滿 8 歲的青少年不得使用本機。,EBF7531SBA_ZH_Manual.pdf,3,安全資訊,images/safety.png"""
    
    # 創建測試CSV文件
    test_csv_path = project_root / "test_faq_data.csv"
    with open(test_csv_path, "w", encoding="utf-8") as f:
        f.write(test_csv_content)
    
    print(f"✅ 創建測試CSV文件: {test_csv_path}")
    
    # 2. 配置預處理器
    config = PreprocessingConfig(
        collection="test_faq_collection",
        embedding=EmbeddingConfig(
            provider="mock",  # 使用模擬embedding
            model="mock-model"
        ),
        question_generation=QuestionGenerationConfig(
            enabled=True,
            max_questions=3,
            temperature=0.7
        )
    )
    
    # 3. 創建預處理器並處理文件
    print("🔄 處理FAQ文件...")
    try:
        preprocessor = Preprocessor(config)
        preprocessor.process_file(str(test_csv_path), dry_run=False)
        print("✅ FAQ文件處理完成")
    except Exception as e:
        print(f"❌ FAQ文件處理失敗: {e}")
        return False
    
    # 4. 創建搜索引擎
    print("🔍 創建搜索引擎...")
    try:
        engine = SearchEngineCorrect("test_faq_collection", use_thread_safe_stats=False)
        print("✅ 搜索引擎創建成功")
    except Exception as e:
        print(f"❌ 搜索引擎創建失敗: {e}")
        return False
    
    # 5. 測試FAQ搜索
    print("🔍 測試FAQ搜索...")
    try:
        mock_embedding_fn = create_mock_embedding(engine.info.dimension)
        
        # 測試查詢
        test_queries = [
            "EBF7531SBA 這台機器怎麼用？",
            "洗碗機型號是什麼？",
            "如何買配件？",
            "小孩能用嗎？"
        ]
        
        for query in test_queries:
            print(f"\n🔍 測試查詢: {query}")
            results = engine.faq_search(
                query=query,
                k=3,
                embedding_fn=mock_embedding_fn
            )
            
            print(f"  找到 {len(results['results'])} 個結果")
            for i, result in enumerate(results['results'], 1):
                print(f"  {i}. 問題: {result['matched_question'][:50]}...")
                print(f"     答案: {result['answer'][:50]}...")
                print(f"     相似度: {result['similarity']:.2f}")
                print(f"     來源: {result['source']['file']} 第{result['source']['page']}頁")
        
        print("✅ FAQ搜索測試完成")
        
    except Exception as e:
        print(f"❌ FAQ搜索測試失敗: {e}")
        return False
    
    # 6. 清理測試文件
    try:
        os.remove(test_csv_path)
        print("✅ 清理測試文件完成")
    except:
        pass
    
    print("\n🎉 FAQ工作流程測試完成！")
    return True

def test_faq_dedup():
    """測試FAQ結果去重功能"""
    print("\n🔍 測試FAQ結果去重功能...")
    
    # 創建搜索引擎
    try:
        engine = SearchEngineCorrect("test_faq_collection", use_thread_safe_stats=False)
        mock_embedding_fn = create_mock_embedding(engine.info.dimension)
        
        # 執行搜索
        results = engine.faq_search(
            query="EBF7531SBA 洗碗機",
            k=5,
            embedding_fn=mock_embedding_fn
        )
        
        # 檢查去重結果
        qa_ids = set()
        for result in results['results']:
            qa_id = result['metadata']['qa_id']
            if qa_id in qa_ids:
                print(f"❌ 發現重複的qa_id: {qa_id}")
                return False
            qa_ids.add(qa_id)
        
        print(f"✅ 去重測試通過，共 {len(results['results'])} 個唯一結果")
        return True
        
    except Exception as e:
        print(f"❌ 去重測試失敗: {e}")
        return False

if __name__ == "__main__":
    print("🚀 開始FAQ工作流程測試...")
    
    # 執行測試
    success = test_faq_workflow()
    
    if success:
        # 執行去重測試
        dedup_success = test_faq_dedup()
        if dedup_success:
            print("\n🎉 所有測試通過！")
            sys.exit(0)
        else:
            print("\n❌ 去重測試失敗")
            sys.exit(1)
    else:
        print("\n❌ FAQ工作流程測試失敗")
        sys.exit(1) 