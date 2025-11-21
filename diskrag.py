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
import polars as pl
from typing import Optional, List, Dict, Any
import time
import os

# 載入環境變數
try:
    from dotenv import load_dotenv
    # 載入 .env 文件
    load_dotenv()
except ImportError:
    # 如果沒有安裝 python-dotenv，嘗試手動載入 .env 文件
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# 預處理相關
from preprocessing.collection import CollectionManager
from preprocessing.config import load_config, PreprocessingConfig, EmbeddingConfig, QuestionGenerationConfig, ChunkConfig, OutputConfig
from preprocessing.processor import Preprocessor
from preprocessing.chunker import DocumentProcessor

# 索引相關
from pydiskann.vamana_graph import build_vamana
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
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.global_config = yaml.safe_load(f)
        else:
            self.global_config = {}
        self.manager = CollectionManager()
        
        # 檢查 OpenAI API Key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY 環境變數未設置。請：\n"
                "1. 在 .env 文件中設置 OPENAI_API_KEY=your-api-key\n"
                "2. 或設置環境變數：export OPENAI_API_KEY=your-api-key"
            )
        
        self.client = OpenAI(api_key=api_key)
        
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

        if collection is None:
            collection = config.collection

        # 根據檔案類型處理
        if input_file.suffix.lower() == '.csv':
            logger.info(f"處理 CSV 檔案: {input_file}")
            self._process_csv(input_file, config, generate_questions)
        elif input_file.suffix.lower() in ['.md', '.markdown']:
            logger.info(f"處理 Markdown 檔案: {input_file}")
            self._process_document(input_file, config, 'md')
        else:
            raise ValueError(f"不支援的檔案類型: {input_file.suffix}。目前支援: .csv, .md, .markdown")
            
    def _process_csv(self, input_file: Path, config: PreprocessingConfig, 
                     generate_questions: bool) -> None:
        """處理 CSV 檔案"""
        config.question_generation.enabled = generate_questions
        processor = Preprocessor(config)
        
        processor.process_file(str(input_file))
            
    def _process_document(self, input_file: Path, config: PreprocessingConfig, doc_type: str) -> None:
        """處理 Markdown 檔案"""
        processor = DocumentProcessor(
            collection_name=config.collection,
            manual_dir=str(input_file.parent),
            config_path=str(self.config_path)
        )
        chunks = processor.process_markdown(input_file)
        
        if chunks:
            self._save_chunks(chunks, processor, config, input_file)
            
    def _save_chunks(self, chunks: List[Any], processor: DocumentProcessor, 
                     config: PreprocessingConfig, input_file: Path = None) -> None:
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
            
            # Ensure collection exists before updating
            info = self.manager.get_collection_info(config.collection)
            if not info:
                dim = vectors.shape[1]
                self.manager.create_collection(
                    collection_name=config.collection,
                    config=config.to_dict(),
                    dimension=dim,
                    source_files=[str(input_file.name) if input_file else "unknown"]
                )

            self.manager.update_collection(
                collection_name=config.collection,
                vectors=vectors,
                texts=valid_texts,
                metadata_list=valid_metadata
            )
            logger.info(f"成功處理 {len(valid_texts)} 個文字塊")
            
    
    def build_index(self, collection: str, target_quality: str = "balanced", 
                    verbose: bool = False, force_rebuild: bool = False) -> None:
        """
        為 collection 建立索引
        """
        try:
            from scripts.tools.build_index import build_index as build_index_func
            
            build_index_func(
                collection_name=collection,
                target_quality=target_quality,
                verbose=verbose,
                force_rebuild=force_rebuild
            )
        except Exception as e:
            logger.error(f"為 collection '{collection}' 建立索引時失敗: {e}")
            raise
        
    def search(self, collection: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜尋"""
        engine = SearchEngine(collection)
        
        # 生成查詢向量
        response = self.client.embeddings.create(
            model=self.global_config.get('embedding', {}).get('model', 'text-embedding-3-small'),
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
        for col in sorted(collections, key=lambda c: c.name):
            print(f"  - {col.name} (向量數: {col.num_vectors})")
        print("-" * 60)
            
    def delete_collection(self, collection: str) -> None:
        """刪除 collection"""
        confirm = input(f"確定要永久刪除 collection '{collection}' 及其所有資料嗎？(y/N): ")
        if confirm.lower() == 'y':
            self.manager.delete_collection(collection)
            logger.info(f"已刪除 collection: {collection}")
        else:
            print("取消刪除。")

    def process_directory(self, directory: str, collection_prefix: str = None,
                         recursive: bool = False, pattern: str = "*") -> Dict[str, Any]:
        """處理目錄中的所有支援檔案"""
        dir_path = Path(directory)
        if not dir_path.exists():
            raise ValueError(f"目錄不存在: {directory}")
            
        supported_extensions = {'.csv', '.md', '.markdown'}
        
        # 收集檔案
        if recursive:
            files = []
            for ext in supported_extensions:
                files.extend(dir_path.rglob(f"{pattern}{ext}"))
        else:
            files = []
            for ext in supported_extensions:
                files.extend(dir_path.glob(f"{pattern}{ext}"))
                
        if not files:
            logger.warning(f"在 {directory} 中沒有找到支援的檔案")
            return {"processed": 0, "failed": 0, "files": []}
            
        logger.info(f"找到 {len(files)} 個檔案")
        
        # 處理檔案
        results = {
            "processed": 0,
            "failed": 0,
            "files": []
        }
        
        for file in sorted(files):
            # 決定 collection 名稱
            if collection_prefix:
                collection = f"{collection_prefix}_{file.stem}"
            else:
                collection = file.stem
                
            logger.info(f"處理 {file.name} -> collection: {collection}")
            
            try:
                # 處理檔案
                self.process(str(file), collection)
                
                # 建立索引
                logger.info(f"建立索引: {collection}")
                self.build_index(collection)
                
                results["processed"] += 1
                results["files"].append({
                    "file": str(file),
                    "collection": collection,
                    "status": "success"
                })
                
            except Exception as e:
                logger.error(f"處理 {file} 時失敗: {str(e)}")
                results["failed"] += 1
                results["files"].append({
                    "file": str(file),
                    "collection": collection,
                    "status": "failed",
                    "error": str(e)
                })
                
        return results

    def merge_collections(self, collections: List[str], target_collection: str) -> None:
        """合併多個 collections 到一個"""
        logger.info(f"合併 {len(collections)} 個 collections 到 {target_collection}")
        
        all_vectors = []
        all_texts = []
        
        # 收集所有數據
        for collection in collections:
            info = self.manager.get_collection_info(collection)
            if not info:
                logger.warning(f"找不到 collection: {collection}")
                continue
                
            vectors_path = self.manager.get_vectors_path(collection)
            metadata_path = self.manager.get_metadata_path(collection)
            
            if not vectors_path.exists() or not metadata_path.exists():
                logger.warning(f"collection {collection} 的數據文件不存在")
                continue
                
            vectors = np.load(str(vectors_path))
            texts_df = pl.read_parquet(str(metadata_path))
            
            all_vectors.append(vectors)
            all_texts.append(texts_df)
            
            logger.info(f"已載入 {collection}: {len(vectors)} 個向量")
        
        if not all_vectors:
            raise ValueError("沒有有效的 collections 可以合併")
        
        # 合併數據
        merged_vectors = np.vstack(all_vectors)
        merged_texts = pl.concat(all_texts)
        
        # 創建新的 collection
        dimension = merged_vectors.shape[1]
        config = self._create_default_config(target_collection)
        
        self.manager.create_collection(
            collection_name=target_collection,
            config=config.to_dict(),
            dimension=dimension,
            source_files=collections
        )
        
        # 保存合併的數據
        self.manager.save_vectors(target_collection, merged_vectors)
        # 保存合併的元數據
        metadata_path = self.manager.get_metadata_path(target_collection)
        merged_texts.write_parquet(str(metadata_path))
        
        logger.info(f"成功合併到 {target_collection}: {len(merged_vectors)} 個向量")

    def doctor_collection(self, collection: str) -> bool:
        """修復指定集合的 PQ 模型"""
        logger.info(f"🔧 開始修復集合 '{collection}' 的 PQ 模型...")
        
        try:
            info = self.manager.get_collection_info(collection)
            if not info:
                logger.error(f"❌ 找不到集合: {collection}")
                return False
            
            # 載入原始向量數據
            vectors_path = self.manager.get_vectors_path(collection)
            
            if not vectors_path.exists():
                logger.error(f"❌ 向量文件不存在: {vectors_path}")
                return False
                
            vectors = np.load(str(vectors_path))
            
            # 檢查向量數據是否為空或損壞
            if vectors.size == 0:
                logger.error(f"❌ 向量數據為空！文件: {vectors_path}")
                return False
            
            # 檢查向量數量是否與集合信息一致
            if info and len(vectors) != info.num_vectors:
                logger.warning(f"⚠️  向量數量不匹配: 文件中有 {len(vectors)} 個，集合信息顯示 {info.num_vectors} 個")
                
                # 嘗試從索引文件中恢復向量
                logger.info("🔧 嘗試從索引文件中恢復向量...")
                try:
                    from pydiskann.io.diskann_persist import MMapNodeReader
                    index_dir = self.manager.get_index_dir(collection)
                    reader = MMapNodeReader(str(index_dir / "index.dat"), dim=info.dimension)
                    
                    # 讀取所有向量
                    recovered_vectors = []
                    for i in range(info.num_vectors):
                        vec, _ = reader.get_node(i)
                        recovered_vectors.append(vec)
                    
                    vectors = np.array(recovered_vectors, dtype=np.float32)
                    reader.close()
                    
                    logger.info(f"✅ 成功從索引恢復 {len(vectors)} 個向量")
                    
                    # 保存恢復的向量
                    np.save(str(vectors_path), vectors)
                    logger.info(f"💾 已保存恢復的向量到: {vectors_path}")
                    
                except Exception as e:
                    logger.error(f"❌ 無法從索引恢復向量: {e}")
                    return False
            
            # 確保數據類型為 float32
            if vectors.dtype != np.float32:
                logger.info(f"🔄 轉換數據類型從 {vectors.dtype} 到 float32")
                vectors = vectors.astype(np.float32)
            
            logger.info(f"📊 向量數據統計:")
            logger.info(f"  - 形狀: {vectors.shape}")
            logger.info(f"  - 數據類型: {vectors.dtype}")
            logger.info(f"  - 向量數量: {len(vectors)}")
            
            # 重新訓練 PQ 模型
            logger.info("🔄 重新訓練 PQ 模型...")
            from pydiskann.pq.fast_pq import DiskANNPQ
            
            pq = DiskANNPQ(
                dimension=info.dimension,
                num_subvectors=info.pq_config.get('num_subvectors', 8),
                num_centroids=info.pq_config.get('num_centroids', 256)
            )
            
            pq.train(vectors)
            
            # 保存新的 PQ 模型
            pq_path = self.manager.get_pq_path(collection)
            pq.save(str(pq_path))
            
            logger.info(f"✅ PQ 模型修復完成！已保存到: {pq_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 修復 PQ 模型時發生錯誤: {e}")
            return False
            
    def _create_default_config(self, collection: str) -> PreprocessingConfig:
        """建立預設設定"""
        return PreprocessingConfig(
            collection=collection,
            embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
            question_generation=QuestionGenerationConfig(enabled=False, provider="openai", model="gpt-4o-mini"),
            chunk=ChunkConfig(),
            output=OutputConfig()
        )

def main():
    parser = argparse.ArgumentParser(
        description='DiskRAG - 一個基於磁碟的 RAG 系統 CLI 工具',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='可用命令', required=True)

    # --- Process Command ---
    process_parser = subparsers.add_parser('process', help='處理來源檔案並生成向量')
    process_parser.add_argument('file', help='要處理的檔案路徑 (.csv, .md, .markdown)')
    process_parser.add_argument('--collection', '-c', help='指定 collection 名稱 (預設: 從檔名或設定檔中獲取)')
    process_parser.add_argument('--questions', '-q', action='store_true', help='為 FAQ (CSV) 生成相似問題')

    # --- Index Command ---
    index_parser = subparsers.add_parser('index', help='為 collection 建立索引')
    index_parser.add_argument('collection', help='要建立索引的 collection 名稱')
    index_parser.add_argument('--target-quality', choices=['fast', 'balanced', 'high'], 
                             default='balanced', help='目標品質級別 (預設: balanced)')
    index_parser.add_argument('--force-rebuild', action='store_true', 
                             help='強制重建索引（忽略已存在的索引）')

    # --- Search Command ---
    search_parser = subparsers.add_parser('search', help='在 collection 中搜尋')
    search_parser.add_argument('collection', help='要搜尋的 collection 名稱')
    search_parser.add_argument('query', help='搜尋的查詢語句')
    search_parser.add_argument('--top-k', '-k', type=int, default=5, help='回傳結果數量 (預設: 5)')

    # --- Process Directory Command ---
    process_dir_parser = subparsers.add_parser('process-dir', help='處理整個目錄的檔案')
    process_dir_parser.add_argument('directory', help='要處理的目錄路徑')
    process_dir_parser.add_argument('--prefix', '-p', help='collection 名稱前綴')
    process_dir_parser.add_argument('--recursive', '-r', action='store_true', help='遞迴處理子目錄')
    process_dir_parser.add_argument('--pattern', default='*', help='檔案匹配模式 (預設: *)')

    # --- Merge Collections Command ---
    merge_parser = subparsers.add_parser('merge', help='合併多個 collections')
    merge_parser.add_argument('collections', nargs='+', help='要合併的 collection 名稱')
    merge_parser.add_argument('--target', '-t', required=True, help='目標 collection 名稱')

    # --- Doctor Command ---
    doctor_parser = subparsers.add_parser('doctor', help='修復 collection 的 PQ 模型')
    doctor_parser.add_argument('collection', help='要修復的 collection 名稱')

    # --- Manage Commands ---
    subparsers.add_parser('list', help='列出所有 collections')
    delete_parser = subparsers.add_parser('delete', help='刪除一個 collection')
    delete_parser.add_argument('collection', help='要刪除的 collection 名稱')

  
    parser.add_argument('--config', default='config.yaml', help='設定檔路徑 (預設: config.yaml)')
    parser.add_argument('--verbose', '-v', action='store_true', help='顯示詳細日誌')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    rag = DiskRAG(args.config)

    try:
        if args.command == 'process':
            rag.process(args.file, args.collection, args.questions)
            print(f"\n處理完成！請記得執行 'python diskrag.py index {args.collection or Path(args.file).stem}' 來建立索引。")

        elif args.command == 'index':
            rag.build_index(
                args.collection, 
                target_quality=getattr(args, 'target_quality', 'balanced'),
                verbose=args.verbose,
                force_rebuild=args.force_rebuild
            )

        elif args.command == 'search':
            results = rag.search(args.collection, args.query, args.top_k)
            print(f"\n搜尋 \"{args.query}\" 的結果 (共 {len(results)} 筆):")
            print("-" * 80)
            for i, result in enumerate(results, 1):
                similarity = 1 - result['distance'] # 假設距離是 0-2 之間
                print(f"[{i}] 相似度: {similarity:.2%}")
                
                # 檢查是否為 FAQ 類型，如果是則顯示答案
                metadata = result.get('metadata', {})
                if isinstance(metadata, str):
                    try:
                        import json
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                
                # 檢查是否為 FAQ（支援嵌套 metadata）
                is_faq = False
                answer = None
                question = None
                
                # 方法1: 檢查頂層是否有 answer 欄位（FAQ 的標誌）
                if metadata.get('answer'):
                    is_faq = True
                    answer = metadata.get('answer')
                    question = metadata.get('original_question') or result.get('text', '')
                # 方法2: 檢查頂層的 type
                elif metadata.get('type') == 'faq':
                    is_faq = True
                    answer = metadata.get('answer')
                    question = metadata.get('original_question') or result.get('text', '')
                # 方法3: 檢查是否有嵌套的 metadata（從 parquet 讀取時可能是字串）
                else:
                    nested_meta_str = metadata.get('metadata')
                    if nested_meta_str:
                        if isinstance(nested_meta_str, str):
                            try:
                                nested_meta = json.loads(nested_meta_str)
                            except:
                                nested_meta = {}
                        else:
                            nested_meta = nested_meta_str
                        
                        if isinstance(nested_meta, dict) and nested_meta.get('type') == 'faq':
                            is_faq = True
                            # 答案在頂層，問題可能在頂層或嵌套層
                            answer = metadata.get('answer')
                            question = metadata.get('original_question') or nested_meta.get('original_question') or result.get('text', '')
                
                if is_faq and answer:
                    # FAQ 搜尋結果：顯示答案
                    print(f"    問題: {question}")
                    print(f"    答案: {answer}")
                else:
                    # 一般搜尋結果：顯示文字內容
                    text_content = result['text'].strip()
                    print(f"    內容: {text_content}")
                
                # 顯示來源資訊
                if metadata:
                    source = metadata.get('manual') or metadata.get('source_type') or metadata.get('source_file')
                    if source:
                        print(f"    來源: {source}")
            print("-" * 80)

        elif args.command == 'list':
            rag.list_collections()

        elif args.command == 'process-dir':
            results = rag.process_directory(
                args.directory,
                collection_prefix=args.prefix,
                recursive=args.recursive,
                pattern=args.pattern
            )
            print(f"\n目錄處理完成！")
            print(f"成功處理: {results['processed']} 個檔案")
            print(f"處理失敗: {results['failed']} 個檔案")
            if results['files']:
                print("\n詳細結果:")
                for file_info in results['files']:
                    status = "✅" if file_info['status'] == 'success' else "❌"
                    print(f"  {status} {file_info['file']} -> {file_info['collection']}")
                    if file_info['status'] == 'failed':
                        print(f"    錯誤: {file_info['error']}")

        elif args.command == 'merge':
            rag.merge_collections(args.collections, args.target)
            print(f"\n合併完成！請執行 'python diskrag.py index {args.target}' 來建立索引。")

        elif args.command == 'doctor':
            success = rag.doctor_collection(args.collection)
            if success:
                print(f"\n✅ 修復完成！collection '{args.collection}' 的 PQ 模型已修復。")
            else:
                print(f"\n❌ 修復失敗！請檢查錯誤訊息。")

        elif args.command == 'delete':
            rag.delete_collection(args.collection)


    except Exception as e:
        logger.error(f"執行命令 '{args.command}' 時發生錯誤: {e}")
        # if args.verbose:
        #     import traceback
        #     traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()