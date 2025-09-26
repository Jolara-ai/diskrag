#!/usr/bin/env python3
"""
批次處理工具 - 處理整個目錄的檔案
"""
import argparse
import logging
from pathlib import Path
import sys
import yaml
from typing import List, Dict, Any
import concurrent.futures
from diskrag import DiskRAG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchProcessor:
    """批次處理多個檔案"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.rag = DiskRAG(config_path)
        self.supported_extensions = {'.csv', '.md', '.markdown', '.docx', '.doc'}
        
    def process_directory(self, directory: str, collection_prefix: str = None,
                         recursive: bool = False, pattern: str = "*") -> Dict[str, Any]:
        """處理目錄中的所有支援檔案"""
        dir_path = Path(directory)
        if not dir_path.exists():
            raise ValueError(f"目錄不存在: {directory}")
            
        # 收集檔案
        if recursive:
            files = []
            for ext in self.supported_extensions:
                files.extend(dir_path.rglob(f"{pattern}{ext}"))
        else:
            files = []
            for ext in self.supported_extensions:
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
                self.rag.process(str(file), collection)
                
                # 建立索引
                logger.info(f"建立索引: {collection}")
                self.rag.build_index(collection)
                
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
        all_metadata = []
        
        manager = self.rag.manager
        
        # 收集所有資料
        for collection in collections:
            try:
                vectors_path = manager.get_vectors_path(collection)
                metadata_df = pl.read_parquet(manager.get_metadata_path(collection))
                
                vectors = np.load(str(vectors_path))
                all_vectors.append(vectors)
                
                for row in metadata_df.iter_rows(named=True):
                    all_texts.append(row["text"])
                    all_metadata.append(row["metadata"])
                    
            except Exception as e:
                logger.warning(f"無法讀取 collection {collection}: {str(e)}")
                
        if not all_vectors:
            raise ValueError("沒有可合併的資料")
            
        # 合併向量
        merged_vectors = np.vstack(all_vectors)
        
        # 建立新 collection
        dimension = merged_vectors.shape[1]
        config = self.rag._create_default_config(target_collection)
        
        manager.create_collection(
            collection_name=target_collection,
            config=config.to_dict(),
            dimension=dimension,
            source_files=[f"merged from: {', '.join(collections)}"]
        )
        
        # 更新資料
        manager.update_collection(
            collection_name=target_collection,
            vectors=merged_vectors,
            texts=all_texts,
            metadata_list=all_metadata
        )
        
        logger.info(f"合併完成: {len(all_texts)} 個文檔")
        
def main():
    parser = argparse.ArgumentParser(
        description='DiskRAG 批次處理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
範例:
  # 處理目錄中的所有檔案
  python batch_process.py process /path/to/docs
  
  # 遞迴處理子目錄
  python batch_process.py process /path/to/docs --recursive
  
  # 使用檔案模式
  python batch_process.py process /path/to/docs --pattern "faq_*"
  
  # 合併 collections
  python batch_process.py merge col1 col2 col3 --target merged_collection
        '''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # process 命令
    process_parser = subparsers.add_parser('process', help='批次處理目錄')
    process_parser.add_argument('directory', help='要處理的目錄')
    process_parser.add_argument('--prefix', help='collection 名稱前綴')
    process_parser.add_argument('--recursive', '-r', action='store_true',
                              help='遞迴處理子目錄')
    process_parser.add_argument('--pattern', default='*',
                              help='檔案名稱模式 (預設: *)')
    
    # merge 命令
    merge_parser = subparsers.add_parser('merge', help='合併 collections')
    merge_parser.add_argument('collections', nargs='+', help='要合併的 collections')
    merge_parser.add_argument('--target', required=True, help='目標 collection 名稱')
    
    # config 參數
    parser.add_argument('--config', default='config.yaml',
                       help='設定檔路徑 (預設: config.yaml)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    processor = BatchProcessor(args.config)
    
    try:
        if args.command == 'process':
            results = processor.process_directory(
                args.directory,
                args.prefix,
                args.recursive,
                args.pattern
            )
            
            print(f"\n處理完成:")
            print(f"成功: {results['processed']}")
            print(f"失敗: {results['failed']}")
            
            if results['failed'] > 0:
                print("\n失敗的檔案:")
                for file_info in results['files']:
                    if file_info['status'] == 'failed':
                        print(f"  {file_info['file']}: {file_info['error']}")
                        
        elif args.command == 'merge':
            processor.merge_collections(args.collections, args.target)
            print(f"\n成功合併到 {args.target}")
            
    except Exception as e:
        logger.error(f"執行失敗: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()