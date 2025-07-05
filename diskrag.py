#!/usr/bin/env python3
"""
DiskRAG - 簡化的主程式入口
"""
import argparse
import logging
import sys
from pathlib import Path
import yaml
import numpy as np
from typing import Optional, List, Dict, Any
import time

# 預處理相關
from preprocessing.collection import CollectionManager
from preprocessing.config import load_config, PreprocessingConfig, EmbeddingConfig, QuestionGenerationConfig, ChunkConfig, OutputConfig
from preprocessing.processor import Preprocessor
from preprocessing.embedding import EmbeddingGenerator
from prepare_md_chunks import DocumentProcessor

# 索引相關
from pydiskann.vamana_graph import build_vamana
from pydiskann.pq.pq_model import SimplePQ
from pydiskann.io.diskann_persist import DiskANNPersist

# 搜尋相關
from search_engine import SearchEngine
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DiskRAG:
    """統一的 DiskRAG 操作介面"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.manager = CollectionManager()
        self.client = OpenAI()
        
    def process(self, input_path: str, collection: Optional[str] = None, 
                generate_questions: bool = False) -> None:
        """處理檔案（自動判斷類型）"""
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"找不到檔案: {input_path}")
            
        # 載入或建立設定
        if self.config_path.exists():
            config = load_config(str(self.config_path))
            if collection:
                config.collection = collection
        else:
            config = self._create_default_config(collection or "default_collection")
            
        # 根據檔案類型處理
        if input_file.suffix.lower() == '.csv':
            logger.info(f"處理 CSV 檔案: {input_file}")
            self._process_csv(input_file, config, generate_questions)
        elif input_file.suffix.lower() in ['.md', '.markdown']:
            logger.info(f"處理 Markdown 檔案: {input_file}")
            self._process_markdown(input_file, config)
        elif input_file.suffix.lower() in ['.docx', '.doc']:
            logger.info(f"處理 Word 檔案: {input_file}")
            self._process_docx(input_file, config)
        else:
            raise ValueError(f"不支援的檔案類型: {input_file.suffix}")
            
    def _process_csv(self, input_file: Path, config: PreprocessingConfig, 
                     generate_questions: bool) -> None:
        """處理 CSV 檔案"""
        config.question_generation.enabled = generate_questions
        processor = Preprocessor(config)
        
        # 如果需要生成問題，先執行 dry-run
        if generate_questions and "_post" not in input_file.stem:
            logger.info("生成問題中...")
            processor.process_file(str(input_file), dry_run=True)
            post_file = input_file.parent / f"{input_file.stem}_post{input_file.suffix}"
            if post_file.exists():
                logger.info(f"使用生成問題後的檔案: {post_file}")
                processor.process_file(str(post_file), dry_run=False)
            else:
                logger.warning("問題生成失敗，使用原始檔案")
                processor.process_file(str(input_file), dry_run=False)
        else:
            processor.process_file(str(input_file), dry_run=False)
            
    def _process_markdown(self, input_file: Path, config: PreprocessingConfig) -> None:
        """處理 Markdown 檔案"""
        processor = DocumentProcessor(
            collection_name=config.collection,
            manual_dir=str(input_file.parent),
            config_path=self.config_path
        )
        chunks = processor.process_markdown(input_file)
        if chunks:
            self._save_chunks(chunks, processor, config)
            
    def _process_docx(self, input_file: Path, config: PreprocessingConfig) -> None:
        """處理 Word 檔案"""
        processor = DocumentProcessor(
            collection_name=config.collection,
            manual_dir=str(input_file.parent),
            config_path=self.config_path
        )
        chunks = processor.process_docx(input_file)
        if chunks:
            self._save_chunks(chunks, processor, config)
            
    def _save_chunks(self, chunks: List[Any], processor: DocumentProcessor, 
                     config: PreprocessingConfig) -> None:
        """儲存文字塊"""
        texts = [chunk.text for chunk in chunks]
        metadata_list = [{
            "id": chunk.id,
            "text": chunk.text,
            "image": chunk.image,
            "section": chunk.section,
            "manual": chunk.manual,
            "source_type": "manual",
            "source_id": str(chunk.id),
            "is_question": False
        } for chunk in chunks]
        
        # 生成向量
        logger.info(f"為 {len(texts)} 個文字塊生成向量...")
        embedding_results, valid_indices = processor.embedding_generator.generate_embeddings(texts)
        
        if embedding_results:
            vectors = np.array([r.vector for r in embedding_results])
            valid_texts = [r.text for r in embedding_results]
            valid_metadata = [metadata_list[i] for i in valid_indices]
            
            # 更新 collection
            processor.collection_manager.update_collection(
                collection_name=config.collection,
                vectors=vectors,
                texts=valid_texts,
                metadata_list=valid_metadata
            )
            logger.info(f"成功處理 {len(valid_texts)} 個文字塊")
            
    def build_index(self, collection: str) -> None:
        """建立索引"""
        info = self.manager.get_collection_info(collection)
        if not info:
            raise ValueError(f"找不到 collection: {collection}")
            
        vectors_path = self.manager.get_vectors_path(collection)
        vectors = np.load(str(vectors_path))
        
        logger.info(f"建立索引 - 向量數: {len(vectors)}, 維度: {vectors.shape[1]}")
        
        # 檢查向量數量是否足夠進行聚類
        min_samples_needed = 16  # KMeans 預設的 n_clusters
        if len(vectors) < min_samples_needed:
            raise ValueError(f"向量數量({len(vectors)})不足，至少需要 {min_samples_needed} 個向量才能建立索引")
        
        # 建立 PQ 模型 - 添加進度條
        logger.info("訓練 PQ 模型...")
        pq_model = SimplePQ(n_subvectors=8)
        pq_model.fit(vectors, show_progress=True)
        
        logger.info("對向量進行 PQ 編碼...")
        pq_codes = pq_model.encode(vectors)
        logger.info(f"PQ 編碼完成，編碼形狀: {pq_codes.shape}")
        
        # 建立 Vamana 圖 - 添加進度條和完整參數
        logger.info("建立 Vamana 圖...")
        graph = build_vamana(
            vectors, 
            R=8,           # 圖的度數
            L=16,           # 搜索列表大小 (通常是 R 的 2 倍)
            alpha=1.2,      # 剪枝參數
            show_progress=True 
        )
        logger.info("Vamana 圖建立完成")
        
        # 儲存索引
        index_dir = self.manager.get_index_dir(collection)
        index_dir.mkdir(parents=True, exist_ok=True)
        
        persist = DiskANNPersist(dim=vectors.shape[1], R=32)
        
        # 保存各個組件
        logger.info("保存索引文件...")
        persist.save_index(str(index_dir / "index.dat"), graph)
        persist.save_pq_codebook(str(index_dir / "pq_model.pkl"), pq_model)
        persist.save_pq_codes(str(index_dir / "pq_codes.bin"), pq_codes)
        
        # 保存元數據
        from datetime import datetime
        persist.save_meta(str(index_dir / "meta.json"), {
            "D": vectors.shape[1],
            "R": 32,
            "N": len(vectors),
            "n_subvectors": 8,
            "pq_centroids": 16,  # 添加 PQ 中心點數量
            "build_time": datetime.now().isoformat()  # 使用 ISO 格式時間
        })
        
        # 更新 collection 信息
        info.chunk_stats.update({
            "index_built_at": datetime.now().isoformat(),
            "index_params": {
                "R": 32,
                "L": 64,
                "alpha": 1.2,
                "pq_subquantizers": 8,
                "pq_centroids": 16,
                "threads": 1
            }
        })
        self.manager.save_collection_info(collection, info)
        
        logger.info(f"索引建立完成: {index_dir}")
        logger.info("更新 collection 信息完成")
        
    def search(self, collection: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜尋"""
        engine = SearchEngine(collection)
        
        # 生成查詢向量
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_vector = np.array(response.data[0].embedding)
        
        # 執行搜尋
        results = engine.search(
            query=query,
            k=top_k,
            embedding_fn=lambda x: query_vector
        )
        
        return results["results"]
        
    def list_collections(self) -> None:
        """列出所有 collections"""
        collections = self.manager.list_collections()
        if not collections:
            print("沒有任何 collection")
            return
            
        print("\n可用的 Collections:")
        print("-" * 60)
        for col in collections:
            print(f"名稱: {col.name}")
            print(f"向量數: {col.num_vectors}")
            print(f"建立時間: {col.created_at}")
            print("-" * 60)
            
    def delete_collection(self, collection: str) -> None:
        """刪除 collection"""
        self.manager.delete_collection(collection)
        logger.info(f"已刪除 collection: {collection}")
        
    def _create_default_config(self, collection: str) -> PreprocessingConfig:
        """建立預設設定"""
        return PreprocessingConfig(
            collection=collection,
            embedding=EmbeddingConfig(
                provider="openai",
                model="text-embedding-3-small",
                max_retries=3,
                retry_delay=2
            ),
            question_generation=QuestionGenerationConfig(
                enabled=False,
                provider="openai",
                model="gpt-4o-mini",
                max_questions=5,
                temperature=0.7
            ),
            chunk=ChunkConfig(
                size=300,
                overlap=50,
                min_size=50
            ),
            output=OutputConfig(
                format="parquet",
                compression="snappy"
            )
        )

def main():
    parser = argparse.ArgumentParser(
        description='DiskRAG - 向量搜尋系統',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
範例:
  # 處理檔案
  python diskrag.py process data/faq.csv --collection faq
  python diskrag.py process data/manual.md --collection manual
  
  # 建立索引
  python diskrag.py index faq
  
  # 搜尋
  python diskrag.py search faq "如何使用系統"
  
  # 管理 collections
  python diskrag.py list
  python diskrag.py delete faq
        '''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # process 命令
    process_parser = subparsers.add_parser('process', help='處理檔案')
    process_parser.add_argument('file', help='要處理的檔案')
    process_parser.add_argument('--collection', '-c', help='collection 名稱')
    process_parser.add_argument('--questions', '-q', action='store_true',
                              help='生成相似問題 (僅 CSV)')
    
    # index 命令
    index_parser = subparsers.add_parser('index', help='建立索引')
    index_parser.add_argument('collection', help='collection 名稱')
    
    # search 命令
    search_parser = subparsers.add_parser('search', help='搜尋')
    search_parser.add_argument('collection', help='collection 名稱')
    search_parser.add_argument('query', help='搜尋查詢')
    search_parser.add_argument('--top-k', '-k', type=int, default=2,
                             help='回傳結果數 (預設: 5)')
    
    # list 命令
    subparsers.add_parser('list', help='列出所有 collections')
    
    # delete 命令
    delete_parser = subparsers.add_parser('delete', help='刪除 collection')
    delete_parser.add_argument('collection', help='collection 名稱')
    
    # config 參數
    parser.add_argument('--config', default='config.yaml',
                       help='設定檔路徑 (預設: config.yaml)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    # 建立 DiskRAG 實例
    rag = DiskRAG(args.config)
    
    try:
        if args.command == 'process':
            rag.process(args.file, args.collection, args.questions)
        elif args.command == 'index':
            rag.build_index(args.collection)
        elif args.command == 'search':
            results = rag.search(args.collection, args.query, args.top_k)
            print(f"\n搜尋結果 (共 {len(results)} 筆):")
            print("-" * 80)
            for i, result in enumerate(results, 1):
                print(f"\n{i}. 相似度: {1 - result['distance']:.2%}")
                print(f"   {result['text'][:200]}...")
                if result.get('metadata'):
                    print(f"   來源: {result['metadata'].get('manual', 'N/A')}")
        elif args.command == 'list':
            rag.list_collections()
        elif args.command == 'delete':
            rag.delete_collection(args.collection)
            
    except Exception as e:
        logger.error(f"執行失敗: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()