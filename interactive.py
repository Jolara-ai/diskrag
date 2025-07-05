#!/usr/bin/env python3
"""
互動式搜尋介面
"""
import cmd
import sys
from pathlib import Path
from typing import List, Dict, Any
from diskrag import DiskRAG
import json

class InteractiveShell(cmd.Cmd):
    """互動式 DiskRAG Shell"""
    
    intro = """
    ╔══════════════════════════════════════╗
    ║       DiskRAG 互動式搜尋介面        ║
    ╚══════════════════════════════════════╝
    
    輸入 'help' 查看可用命令
    輸入 'quit' 或 Ctrl+D 退出
    """
    
    prompt = 'diskrag> '
    
    def __init__(self, config_path: str = "config.yaml"):
        super().__init__()
        self.rag = DiskRAG(config_path)
        self.current_collection = None
        self.last_results = []
        
    def do_use(self, collection: str):
        """使用指定的 collection: use <collection_name>"""
        if not collection:
            print("請指定 collection 名稱")
            return
            
        # 檢查 collection 是否存在
        collections = self.rag.manager.list_collections()
        collection_names = [c.name for c in collections]
        
        if collection not in collection_names:
            print(f"錯誤: collection '{collection}' 不存在")
            print(f"可用的 collections: {', '.join(collection_names)}")
            return
            
        self.current_collection = collection
        self.prompt = f'diskrag[{collection}]> '
        print(f"已切換到 collection: {collection}")
        
    def do_list(self, args):
        """列出所有 collections"""
        collections = self.rag.manager.list_collections()
        if not collections:
            print("沒有任何 collection")
            return
            
        print("\n可用的 Collections:")
        print("-" * 60)
        for col in collections:
            status = "✓" if col.name == self.current_collection else " "
            print(f"{status} {col.name:<20} 向量數: {col.num_vectors:<10} 建立: {col.created_at[:10]}")
        print("-" * 60)
        
    def do_search(self, query: str):
        """搜尋當前 collection: search <query>"""
        if not self.current_collection:
            print("請先使用 'use <collection>' 選擇 collection")
            return
            
        if not query:
            print("請輸入搜尋查詢")
            return
            
        try:
            results = self.rag.search(self.current_collection, query, top_k=5)
            self.last_results = results
            
            print(f"\n搜尋結果 (共 {len(results)} 筆):")
            print("=" * 80)
            
            for i, result in enumerate(results, 1):
                similarity = 1 - result['distance']
                print(f"\n[{i}] 相似度: {similarity:.2%}")
                print("-" * 40)
                
                # 顯示文字（限制長度）
                text = result['text']
                if len(text) > 200:
                    text = text[:200] + "..."
                print(f"內容: {text}")
                
                # 顯示元資料
                if result.get('metadata'):
                    meta = result['metadata']
                    if isinstance(meta, str):
                        try:
                            meta = json.loads(meta)
                        except:
                            pass
                    
                    if isinstance(meta, dict):
                        if meta.get('manual'):
                            print(f"來源: {meta['manual']}")
                        if meta.get('section'):
                            print(f"章節: {meta['section']}")
                        if meta.get('question'):
                            print(f"問題: {meta['question']}")
                            
            print("=" * 80)
            
        except Exception as e:
            print(f"搜尋失敗: {str(e)}")
            
    def do_show(self, index: str):
        """顯示搜尋結果的完整內容: show <index>"""
        if not self.last_results:
            print("沒有搜尋結果")
            return
            
        try:
            idx = int(index) - 1
            if 0 <= idx < len(self.last_results):
                result = self.last_results[idx]
                print("\n" + "=" * 80)
                print(f"完整內容 (相似度: {1 - result['distance']:.2%}):")
                print("=" * 80)
                print(result['text'])
                print("=" * 80)
                
                if result.get('metadata'):
                    print("\n元資料:")
                    print(json.dumps(result['metadata'], indent=2, ensure_ascii=False))
            else:
                print(f"無效的索引: {index}")
        except ValueError:
            print(f"請輸入有效的數字索引")
            
    def do_info(self, collection: str = None):
        """顯示 collection 資訊: info [collection_name]"""
        target = collection or self.current_collection
        if not target:
            print("請指定 collection 名稱或先使用 'use' 選擇")
            return
            
        info = self.rag.manager.get_collection_info(target)
        if not info:
            print(f"找不到 collection: {target}")
            return
            
        print(f"\nCollection 資訊: {target}")
        print("-" * 60)
        print(f"向量數量: {info.num_vectors}")
        print(f"向量維度: {info.dimension}")
        print(f"建立時間: {info.created_at}")
        print(f"更新時間: {info.updated_at}")
        
        if info.chunk_stats:
            print("\n統計資訊:")
            for key, value in info.chunk_stats.items():
                print(f"  {key}: {value}")
                
    def do_process(self, args: str):
        """處理檔案: process <file> [collection]"""
        parts = args.split()
        if not parts:
            print("請指定檔案路徑")
            return
            
        file_path = parts[0]
        collection = parts[1] if len(parts) > 1 else Path(file_path).stem
        
        try:
            print(f"處理檔案: {file_path} -> {collection}")
            self.rag.process(file_path, collection)
            print(f"處理完成，建立索引中...")
            self.rag.build_index(collection)
            print(f"完成！collection '{collection}' 已就緒")
            
            # 自動切換到新 collection
            self.do_use(collection)
            
        except Exception as e:
            print(f"處理失敗: {str(e)}")
            
    def do_delete(self, collection: str):
        """刪除 collection: delete <collection_name>"""
        if not collection:
            print("請指定要刪除的 collection")
            return
            
        # 確認刪除
        confirm = input(f"確定要刪除 collection '{collection}'？(y/N) ")
        if confirm.lower() == 'y':
            try:
                self.rag.delete_collection(collection)
                print(f"已刪除 collection: {collection}")
                
                if self.current_collection == collection:
                    self.current_collection = None
                    self.prompt = 'diskrag> '
                    
            except Exception as e:
                print(f"刪除失敗: {str(e)}")
        else:
            print("取消刪除")
            
    def do_clear(self, args):
        """清除螢幕"""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def do_quit(self, args):
        """退出程式"""
        print("\n再見！")
        return True
        
    def do_exit(self, args):
        """退出程式"""
        return self.do_quit(args)
        
    def do_EOF(self, args):
        """處理 Ctrl+D"""
        print()  # 換行
        return self.do_quit(args)
        
    def emptyline(self):
        """空行不執行任何操作"""
        pass
        
    def default(self, line):
        """處理未知命令"""
        # 如果已選擇 collection，將輸入視為搜尋查詢
        if self.current_collection and not line.startswith('!'):
            self.do_search(line)
        else:
            print(f"未知命令: {line}")
            print("輸入 'help' 查看可用命令")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='DiskRAG 互動式搜尋介面')
    parser.add_argument('--config', default='config.yaml',
                       help='設定檔路徑 (預設: config.yaml)')
    parser.add_argument('--collection', help='啟動時使用的 collection')
    
    args = parser.parse_args()
    
    shell = InteractiveShell(args.config)
    
    # 如果指定了 collection，自動切換
    if args.collection:
        shell.do_use(args.collection)
        
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\n\n再見！")
        sys.exit(0)

if __name__ == "__main__":
    main()