import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
from .collection import CollectionManager
from .config import load_config, save_config, PreprocessingConfig
from .processor import Preprocessor
import json
import yaml
from build_index import build_index  # 導入 build_index 函數
import numpy as np

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False) -> None:
    """設置日誌級別"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def process_file(args: argparse.Namespace) -> None:
    """處理文件並更新 collection
    
    支持的文件類型：
    - csv: FAQ 格式的 CSV 文件，使用 Preprocessor 處理
    - md/docx: 手冊文件，使用 DocumentProcessor 處理
    """
    try:
        # 加載配置
        config = load_config(args.config)
        input_path = Path(args.input)
        
        if not input_path.exists():
            raise FileNotFoundError(f"找不到輸入文件: {args.input}")
        
        # 檢查文件類型
        if args.type == "csv":
            if input_path.suffix.lower() != '.csv':
                raise ValueError(f"FAQ 類型必須是 CSV 文件，收到: {input_path.suffix}")
            # 使用 Preprocessor 處理 FAQ 文件
            processor = Preprocessor(config)
            processor.process_file(str(input_path), dry_run=args.dry_run)
            
        elif args.type in ["md", "docx"]:
            if args.dry_run:
                logger.warning("dry-run 模式僅支持 CSV 文件類型")
                return
                
            if input_path.suffix.lower() not in ['.md', '.docx']:
                raise ValueError(f"不支持的文件類型: {input_path.suffix}")
            if args.type == "md" and input_path.suffix.lower() != '.md':
                raise ValueError(f"Markdown 類型必須是 .md 文件，收到: {input_path.suffix}")
            if args.type == "docx" and input_path.suffix.lower() != '.docx':
                raise ValueError(f"Word 類型必須是 .docx 文件，收到: {input_path.suffix}")
                
            # 使用 DocumentProcessor 處理手冊文件
            from prepare_md_chunks import DocumentProcessor
            processor = DocumentProcessor(
                collection_name=config.collection,
                manual_dir=str(input_path.parent),
                config_path=args.config
            )
            
            # 處理單個文件
            chunks = []
            try:
                if args.type == "md":
                    chunks = processor.process_markdown(input_path)
                else:  # docx
                    chunks = processor.process_docx(input_path)
            except Exception as e:
                logger.error(f"處理文件時出錯: {str(e)}")
                raise ValueError(f"文件處理失敗: {str(e)}")
                
            if not chunks:
                logger.warning(f"文件 {input_path.name} 沒有產生任何有效的文字塊")
                return
                
            # 準備文本和元數據
            try:
                texts = [chunk.text for chunk in chunks]
                metadata_list = [{
                    "id": chunk.id,
                    "text": chunk.text,
                    "image": chunk.image,
                    "section": chunk.section,
                    "manual": chunk.manual,
                    "source_type": "manual",
                    "source_id": str(chunk.id),
                    "metadata": json.dumps({
                        "manual": chunk.manual,
                        "section": chunk.section,
                        "image": chunk.image
                    }, ensure_ascii=False),
                    "is_question": False
                } for chunk in chunks]
            except Exception as e:
                logger.error(f"準備文本和元數據時出錯: {str(e)}")
                raise ValueError(f"數據準備失敗: {str(e)}")
            
            # 生成向量
            logger.info(f"為 {len(texts)} 個文本生成向量...")
            try:
                embedding_results, valid_indices = processor.embedding_generator.generate_embeddings(texts)
            except Exception as e:
                logger.error(f"生成向量時出錯: {str(e)}")
                raise ValueError(f"向量生成失敗: {str(e)}")
            
            if not embedding_results:
                logger.error("沒有生成任何有效的向量")
                return
                
            # 只保留有效的文本和元數據
            try:
                vectors = np.array([r.vector for r in embedding_results])
                valid_texts = [r.text for r in embedding_results]
                valid_metadata = [metadata_list[i] for i in valid_indices]
            except Exception as e:
                logger.error(f"處理向量結果時出錯: {str(e)}")
                raise ValueError(f"處理向量數據時出錯: {str(e)}")
            
            # 更新 collection
            try:
                processor.collection_manager.update_collection(
                    collection_name=config.collection,
                    vectors=vectors,
                    texts=valid_texts,
                    metadata_list=valid_metadata
                )
            except Exception as e:
                logger.error(f"更新 collection 時出錯: {str(e)}")
                raise ValueError(f"更新 collection 失敗: {str(e)}")
            
            # 更新統計信息
            try:
                info = processor.collection_manager.get_collection_info(config.collection)
                if info:
                    info.chunk_stats.update({
                        "total_chunks": len(valid_texts),
                        "total_questions": 0,
                        "last_processed_files": [input_path.name]
                    })
                    processor.collection_manager.save_collection_info(config.collection, info)
            except Exception as e:
                logger.error(f"更新統計信息時出錯: {str(e)}")
                # 不拋出異常，因為這不是關鍵操作
            
            logger.info(f"成功處理文件 {input_path.name}，生成了 {len(valid_texts)} 個文字塊")
            
        else:
            raise ValueError(f"不支持的文件類型: {args.type}")
        
    except Exception as e:
        logger.error(f"處理文件時出錯: {str(e)}")
        raise

def list_collections(args: argparse.Namespace) -> None:
    """列出所有 collections"""
    try:
        manager = CollectionManager()
        collections = manager.list_collections()
        
        if not collections:
            print("目前沒有 collections")
            return
            
        print("\n現有的 collections:")
        print("-" * 80)
        for collection in collections:
            print(f"名稱: {collection.name}")
            print(f"模型: {collection.config['embedding']['model']}")
            print(f"向量數: {collection.num_vectors}")
            print(f"創建時間: {collection.created_at}")
            print(f"更新時間: {collection.updated_at}")
            print(f"來源文件: {', '.join(collection.source_files)}")
            print(f"統計信息: {collection.chunk_stats}")
            print("-" * 80)
            
    except Exception as e:
        logger.error(f"列出 collections 時出錯: {str(e)}")
        sys.exit(1)

def delete_collection(args: argparse.Namespace) -> None:
    """刪除 collection"""
    try:
        manager = CollectionManager()
        manager.delete_collection(args.name)
        logger.info(f"成功刪除 collection {args.name}")
        
    except Exception as e:
        logger.error(f"刪除 collection 時出錯: {str(e)}")
        sys.exit(1)

def rebuild_collection(args: argparse.Namespace) -> None:
    """重建 collection
    
    支持的文件類型：
    - faq: FAQ 格式的 CSV 文件，使用 Preprocessor 處理
    - md/docx: 手冊文件，使用 DocumentProcessor 處理
    """
    try:
        # 加載配置
        config = load_config(args.config)
        
        if args.type == "faq":
            # 使用 Preprocessor 重建 FAQ collection
            processor = Preprocessor(config)
            processor.rebuild_collection(args.input)
        elif args.type in ["md", "docx"]:
            # 使用 DocumentProcessor 重建手冊 collection
            from prepare_md_chunks import DocumentProcessor
            processor = DocumentProcessor(
                collection_name=args.name,
                manual_dir=str(Path(args.input).parent),
                config_path=args.config
            )
            # 刪除現有 collection
            processor.collection_manager.delete_collection(args.name)
            # 重新處理所有文檔
            processor.process_all_documents()
        else:
            raise ValueError(f"不支持的文件類型: {args.type}")
        
        logger.info(f"成功重建 collection {args.name}")
        
    except Exception as e:
        logger.error(f"重建 collection 時出錯: {str(e)}")
        sys.exit(1)

def show_collection_info(args: argparse.Namespace) -> None:
    """顯示 collection 詳細信息"""
    try:
        manager = CollectionManager()
        info = manager.get_collection_info(args.name)
        
        if not info:
            logger.error(f"找不到 collection {args.name}")
            sys.exit(1)
            
        print(f"\nCollection: {info.name}")
        print("-" * 80)
        print(f"向量維度: {info.dimension}")
        print(f"向量數量: {info.num_vectors}")
        print(f"創建時間: {info.created_at}")
        print(f"更新時間: {info.updated_at}")
        
        print("\n來源文件:")
        for file_info in info.get_source_files_info():
            print(f"  路徑: {file_info['path']}")
            print(f"  類型: {file_info['type']}")
            print(f"  是否為後處理文件: {file_info['is_post_file']}")
            print(f"  總行數: {file_info['total_rows']}")
            print(f"  有效行數: {file_info['valid_rows']}")
            print()
        
        print("\n配置信息:")
        config = info.config
        print(f"  Embedding 提供商: {config['embedding']['provider']}")
        print(f"  Embedding 模型: {config['embedding']['model']}")
        print(f"  問題生成: {'啟用' if config['question_generation']['enabled'] else '禁用'}")
        if config['question_generation']['enabled']:
            print(f"  最大問題數: {config['question_generation']['max_questions']}")
        
        print("\n統計信息:")
        for k, v in info.chunk_stats.items():
            print(f"  {k}: {v}")
        
        if "index_built_at" in info.chunk_stats:
            print("\n索引信息:")
            print(f"  建立時間: {info.chunk_stats['index_built_at']}")
            if "index_params" in info.chunk_stats:
                params = info.chunk_stats["index_params"]
                print(f"  圖度數 (R): {params['R']}")
                print(f"  搜索列表大小 (L): {params['L']}")
                print(f"  剪枝參數 (alpha): {params['alpha']}")
                print(f"  線程數: {params['threads']}")
                print(f"  PQ 子量化器數: {params['pq_subquantizers']}")
                print(f"  PQ 中心點數: {params['pq_centroids']}")
        
        print("-" * 80)
        
    except Exception as e:
        logger.error(f"獲取 collection 信息時出錯: {str(e)}")
        sys.exit(1)

def create_config(args: argparse.Namespace) -> None:
    """創建新的配置文件"""
    try:
        config = PreprocessingConfig(
            collection=args.collection,
            embedding={
                'provider': args.embedding_provider,
                'model': args.embedding_model,
                'project_id': args.project_id,
                'api_key': args.api_key,
                'max_retries': args.max_retries,
                'retry_delay': args.retry_delay
            },
            question_generation={
                'enabled': args.enable_questions,
                'provider': args.llm_provider,
                'model': args.llm_model,
                'max_questions': args.max_questions,
                'temperature': args.temperature,
                'max_retries': args.max_retries,
                'retry_delay': args.retry_delay,
                'project_id': args.project_id
            }
        )
        
        save_config(config, args.output)
        logger.info(f"成功創建配置文件 {args.output}")
        
    except Exception as e:
        logger.error(f"創建配置文件時出錯: {str(e)}")
        sys.exit(1)

def parse_args() -> argparse.Namespace:
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description="文檔預處理和索引建立工具")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # process 命令
    process_parser = subparsers.add_parser("process", help="處理文檔")
    process_parser.add_argument("--type", required=True, choices=["md", "docx", "csv"], help="文檔類型")
    process_parser.add_argument("--input", required=True, help="輸入文件路徑")
    process_parser.add_argument("--config", default="config.yaml", help="配置文件路徑")
    process_parser.add_argument("--dry-run", action="store_true", help="試運行模式（僅生成問題，不建立向量和索引，僅支持 CSV 文件）")
    
    # list 命令
    subparsers.add_parser("list", help="列出所有 collections")
    
    # show 命令
    show_parser = subparsers.add_parser("show", help="顯示 collection 詳情")
    show_parser.add_argument("--name", required=True, help="collection 名稱")
    
    # delete 命令
    delete_parser = subparsers.add_parser("delete", help="刪除 collection")
    delete_parser.add_argument("--name", required=True, help="collection 名稱")
    
    # rebuild 命令
    rebuild_parser = subparsers.add_parser("rebuild", help="重建 collection")
    rebuild_parser.add_argument("--name", required=True, help="collection 名稱")
    rebuild_parser.add_argument("--type", required=True, choices=["md", "docx", "csv"], help="文檔類型")
    rebuild_parser.add_argument("--input", required=True, help="輸入文件路徑")
    rebuild_parser.add_argument("--config", default="config.yaml", help="配置文件路徑")
    rebuild_parser.add_argument("--dry-run", action="store_true", help="試運行模式（僅生成問題，不建立向量和索引，僅支持 CSV 文件）")
    
    # build-index 命令
    build_index_parser = subparsers.add_parser("build-index", help="建立 collection 索引")
    build_index_parser.add_argument("collection", help="collection 名稱")
    build_index_parser.add_argument("--R", type=int, default=32, help="圖的度數（默認：32）")
    build_index_parser.add_argument("--threads", type=int, default=1, help="線程數（默認：1）")
    build_index_parser.add_argument("--verbose", action="store_true", help="顯示詳細日誌")
    build_index_parser.add_argument("--config", default="config.yaml", help="配置文件路徑")
    
    # create-config 命令
    create_config_parser = subparsers.add_parser("create-config", help="創建配置文件")
    create_config_parser.add_argument("--output", default="config.yaml", help="輸出文件路徑")
    
    return parser.parse_args()

def main() -> None:
    """主函數"""
    args = parse_args()
    
    if not args.command:
        logger.error("請指定命令")
        sys.exit(1)
        
    setup_logging(args.verbose if hasattr(args, "verbose") else False)
    
    try:
        if args.command == "process":
            # 根據文件類型選擇處理方法
            process_file(args)
            
        elif args.command == "list":
            manager = CollectionManager()
            collections = manager.list_collections()
            if not collections:
                logger.info("沒有找到任何 collections")
            else:
                for collection in collections:
                    logger.info(f"Collection: {collection.name}")
                    logger.info(f"  總文本塊數: {collection.chunk_stats.get('total_chunks', 0)}")
                    logger.info(f"  總問題數: {collection.chunk_stats.get('total_questions', 0)}")
                    logger.info(f"  最後處理的文件: {', '.join(collection.chunk_stats.get('last_processed_files', []))}")
                    if "index_built_at" in collection.chunk_stats:
                        logger.info(f"  索引建立時間: {collection.chunk_stats['index_built_at']}")
                    logger.info("---")
            
        elif args.command == "show":
            manager = CollectionManager()
            info = manager.get_collection_info(args.name)
            if not info:
                logger.error(f"找不到 collection: {args.name}")
                sys.exit(1)
            logger.info(f"Collection: {args.name}")
            logger.info(f"  配置: {info.config}")
            logger.info(f"  維度: {info.dimension}")
            logger.info(f"  統計信息: {info.chunk_stats}")
            
        elif args.command == "delete":
            manager = CollectionManager()
            manager.delete_collection(args.name)
            logger.info(f"已刪除 collection: {args.name}")
            
        elif args.command == "rebuild":
            # 根據文件類型選擇重建方法
            process_file(args)
            
        elif args.command == "build-index":
            # 嘗試從配置文件讀取索引參數
            try:
                with open(args.config, "r", encoding="utf-8") as f:
                    config_dict = yaml.safe_load(f)
                index_params = config_dict.get("index", {})
                R = index_params.get("R", args.R)
                threads = index_params.get("threads", args.threads)
            except Exception as e:
                logger.warning(f"從配置文件讀取索引參數時出錯: {str(e)}，使用命令行參數")
                R = args.R
                threads = args.threads
                
            build_index(
                collection_name=args.collection,
                R=R,
                threads=threads,
                verbose=args.verbose
            )
            
        elif args.command == "create-config":
            config = {
                "collection": "your_collection_name",
                "embedding": {
                    "provider": "openai",
                    "model": "text-embedding-3-small",
                    "api_key": None,  # 從環境變數讀取
                    "max_retries": 3,
                    "retry_delay": 2
                },
                "question_generation": {
                    "enabled": False,
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "max_questions": 5,
                    "temperature": 0.7,
                    "max_retries": 3,
                    "retry_delay": 2
                },
                "chunk": {
                    "size": 300,
                    "overlap": 50,
                    "min_size": 50
                },
                "output": {
                    "format": "parquet",
                    "compression": "snappy"
                },
                "index": {
                    "R": 32,
                    "threads": 1
                }
            }
            
            with open(args.output, "w", encoding="utf-8") as f:
                yaml.dump(config, f, allow_unicode=True, sort_keys=False)
            logger.info(f"已創建配置文件: {args.output}")
            
    except Exception as e:
        logger.error(f"執行命令時出錯: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 