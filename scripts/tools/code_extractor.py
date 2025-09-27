import ast
import os
import sys
import argparse
from pathlib import Path
from typing import List, Set, Dict, Optional

class ImportVisitor(ast.NodeVisitor):
    """
    一個 AST 訪問者，用於從 AST 中提取導入語句。
    """
    def __init__(self, current_file_path: Path):
        self.imports = set()
        self.current_dir = current_file_path.parent

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        # 處理相對導入，例如 from . import utils 或 from ..config import settings
        if node.level > 0:
            # 將點轉換為對應的上層目錄
            base_path = self.current_dir
            for _ in range(node.level - 1):
                base_path = base_path.parent
            
            if node.module:
                module_path = f"{base_path.name}.{node.module}"
            else:
                module_path = base_path.name
            
            # 這裡我們將相對路徑轉換為一個可解析的模組字串，
            # 但真正的路徑解析會在主邏輯中處理。
            # 簡化處理：我們只關心模組名稱本身。
            module_name = node.module or ''
            self.imports.add(module_name)

        elif node.module:
            self.imports.add(node.module)
        self.generic_visit(node)

def resolve_module_path(module_name: str, project_root: Path, search_path: Path) -> Optional[Path]:
    """
    嘗試將模組名稱解析為專案內的實際文件路徑。

    Args:
        module_name: 導入的模組名稱 (例如 'my_app.utils.helpers')。
        project_root: 專案的根目錄。
        search_path: 當前文件所在的目錄，用於解析相對導入。

    Returns:
        如果模組是專案內文件，則返回其 Path 物件，否則返回 None。
    """
    # 將 'my_app.utils' 轉換為 'my_app/utils'
    module_parts = module_name.split('.')
    
    # 嘗試不同的起始搜索路徑（當前目錄和專案根目錄）
    for base_path in [search_path, project_root]:
        # 可能性 1: 是一個 .py 文件 (e.g., .../utils.py)
        potential_path = base_path.joinpath(*module_parts).with_suffix('.py')
        if potential_path.is_file() and str(potential_path).startswith(str(project_root)):
            return potential_path.resolve()

        # 可能性 2: 是一個套件目錄 (e.g., .../utils/__init__.py)
        potential_path = base_path.joinpath(*module_parts, '__init__.py')
        if potential_path.is_file() and str(potential_path).startswith(str(project_root)):
            return potential_path.resolve()
            
    return None


def extract_code_bundle(entry_point: str, project_root: str) -> Dict[Path, str]:
    """
    從入口文件開始，遞歸地提取所有相關的專案內程式碼。

    Args:
        entry_point: 程式的入口文件路徑。
        project_root: 專案的根目錄。

    Returns:
        一個字典，鍵是文件的絕對路徑(Path)，值是文件內容。
    """
    try:
        project_root_path = Path(project_root).resolve()
        entry_point_path = Path(entry_point).resolve()
    except FileNotFoundError:
        print(f"錯誤：入口文件或專案根目錄不存在。")
        sys.exit(1)

    files_to_process: List[Path] = [entry_point_path]
    processed_files: Set[Path] = set()
    code_collection: Dict[Path, str] = {}

    while files_to_process:
        current_file = files_to_process.pop(0)

        if current_file in processed_files:
            continue
        
        # 確保我們不會處理專案目錄外的文件
        if not str(current_file).startswith(str(project_root_path)):
            print(f"警告：跳過外部文件 {current_file}")
            continue

        processed_files.add(current_file)

        try:
            with open(current_file, 'r', encoding='utf-8') as f:
                source_code = f.read()
            code_collection[current_file] = source_code
            tree = ast.parse(source_code)
        except (FileNotFoundError, UnicodeDecodeError, SyntaxError) as e:
            print(f"警告：無法讀取或解析文件 {current_file}：{e}")
            continue

        # 尋找此文件中的 import
        visitor = ImportVisitor(current_file)
        visitor.visit(tree)
        
        # 解析找到的 import，並將專案內模組加入待處理佇列
        for imp_name in visitor.imports:
            # 處理相對導入的路徑基礎
            search_path_for_resolve = current_file.parent

            # `from . import utils` 的 imp_name 會是 'utils'
            # `from .sub import helper` 的 imp_name 會是 'sub.helper'
            # 我們需要正確的搜索路徑
            resolved_path = resolve_module_path(imp_name, project_root_path, search_path_for_resolve)
            
            if resolved_path and resolved_path not in processed_files:
                files_to_process.append(resolved_path)

    return code_collection

def save_to_markdown(code_collection: Dict[Path, str], output_file: str, project_root: str):
    """
    將收集到的程式碼保存為一個 Markdown 文件。
    """
    project_root_path = Path(project_root).resolve()
    
    # 依檔案路徑排序，確保每次輸出順序一致
    sorted_files = sorted(code_collection.keys())

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# 程式碼上下文捆綁包\n\n")
        f.write(f"**專案根目錄:** `{project_root}`\n")
        f.write(f"**生成時間:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        for file_path in sorted_files:
            relative_path = file_path.relative_to(project_root_path)
            file_content = code_collection[file_path]
            
            f.write(f"## `FILE: {str(relative_path).replace(os.sep, '/')}`\n\n")
            f.write("```python\n")
            f.write(file_content.strip())
            f.write("\n```\n\n")
            f.write("---\n\n")
            
    print(f"✅ 成功將 {len(code_collection)} 個文件的程式碼捆綁到 {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="一個 Python 腳本，透過 AST 追蹤專案內的引用，並將相關程式碼捆綁成單一文件，以供 LLM 分析。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "entry_point", 
        help="程式的入口文件，例如：src/main.py"
    )
    parser.add_argument(
        "-p", "--project-root", 
        default=".",
        help="專案的根目錄。用於區分專案內/外模組。\n預設為當前目錄 ('.')。"
    )
    parser.add_argument(
        "-o", "--output", 
        default="code_bundle.md",
        help="輸出的文件名。建議使用 .md 副檔名。\n預設為 'code_bundle.md'。"
    )
    
    args = parser.parse_args()

    code_bundle = extract_code_bundle(args.entry_point, args.project_root)
    
    if code_bundle:
        save_to_markdown(code_bundle, args.output, args.project_root)
    else:
        print("沒有找到任何可捆綁的程式碼。請檢查入口文件和專案路徑。")

if __name__ == "__main__":
    main()