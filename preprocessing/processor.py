import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import polars as pl
import numpy as np
import json

from .config import PreprocessingConfig, get_text_hash
from .question_generator import QuestionGenerator
from .embedding import EmbeddingGenerator
from .collection import CollectionManager

logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        # 將 QuestionGenerationConfig 轉換為字典
        question_gen_config = {
            "enabled": config.question_generation.enabled,
            "provider": config.question_generation.provider,
            "model": config.question_generation.model,
            "max_questions": config.question_generation.max_questions,
            "temperature": config.question_generation.temperature,
            "max_retries": config.question_generation.max_retries,
            "retry_delay": config.question_generation.retry_delay,
            "project_id": config.question_generation.project_id
        }
        self.question_generator = QuestionGenerator(question_gen_config)
        self.embedding_generator = EmbeddingGenerator(config.embedding)
        self.collection_manager = CollectionManager()

    def _save_generated_questions(self, texts: List[str], metadata_list: List[Dict[str, Any]]) -> None:
        """保存生成的問題
        
        Args:
            texts: 原始文本列表
            metadata_list: 元數據列表
        """
        try:
            # 生成問題
            questions = self.question_generator.generate_questions(texts)
            if not questions:
                return
            
            # 準備問題的文本和元數據
            question_texts = []
            question_metadata = []
            
            for i, (text, questions_for_text) in enumerate(zip(texts, questions)):
                for q in questions_for_text:
                    question_texts.append(q)
                    # 使用原始文本的元數據作為基礎
                    base_metadata = metadata_list[i].copy()
                    base_metadata.update({
                        "id": get_text_hash(q),
                        "text": q,
                        "source_id": f"{base_metadata['source_id']}_q_{get_text_hash(q)}",
                        "metadata": json.dumps({
                            "type": "generated_question",
                            "original_id": base_metadata["id"]
                        }, ensure_ascii=False)
                    })
                    question_metadata.append(base_metadata)
            
            if not question_texts:
                return
            
            # 生成問題的向量
            question_results, valid_indices = self.embedding_generator.generate_embeddings(question_texts)
            
            if not question_results:
                logger.warning("沒有生成任何有效的問題向量")
                return
            
            # 只保留有效的問題
            question_vectors = np.array([r.vector for r in question_results])
            valid_question_texts = [r.text for r in question_results]
            valid_question_metadata = [r.metadata for r in question_results]
            
            # 更新 collection
            self.collection_manager.update_collection(
                collection_name=self.config.collection,
                vectors=question_vectors,
                texts=valid_question_texts,
                metadata_list=valid_question_metadata
            )
            
            # 更新統計信息
            info = self.collection_manager.get_collection_info(self.config.collection)
            if info:
                info.chunk_stats["total_questions"] += len(valid_question_texts)
                self.collection_manager.save_collection_info(self.config.collection, info)
            
            logger.info(f"成功生成並保存 {len(valid_question_texts)} 個問題")
            
        except Exception as e:
            logger.error(f"保存生成的問題時出錯: {str(e)}")
            # 不拋出異常，讓主流程繼續執行

    def _generate_and_save_questions(self, input_file: str, df: pl.DataFrame) -> Optional[str]:
        """生成問題並保存到新的 CSV 文件
        
        Args:
            input_file: 原始 CSV 文件路徑
            df: 原始數據的 DataFrame
            
        Returns:
            Optional[str]: 生成問題後的 CSV 文件路徑，如果生成失敗則返回 None
        """
        try:
            logger.info("開始生成問題...")
            # 準備原始文本和元數據
            all_generated_questions = []
            
            for row in df.iter_rows(named=True):
                if not row.get("question") or not row.get("answer"):
                    continue
                
                # 檢查是否為生成的問題
                is_generated = row.get("is_generated", False)
                original_question = row.get("original_question")
                
                # 如果是生成的問題，記錄日誌
                if is_generated:
                    logger.info(f"處理生成的問題:")
                    logger.info(f"  原始問題: {original_question}")
                    logger.info(f"  生成問題: {row['question']}")
                    logger.info(f"  答案: {row['answer']}")
                    logger.info(f"  備註: {row.get('note', '')}")
                
                # 準備元數據
                metadata = {
                    "type": "faq",
                    "question": row["question"],
                    "answer": row["answer"],
                    "is_generated": is_generated,
                    "original_question": original_question,
                    "note": row.get("note", ""),
                    "source_id": str(get_text_hash(f"問題：{row['question']}\n答案：{row['answer']}"))
                }
                
                # 如果不是生成的問題，嘗試生成新問題
                if not is_generated:
                    try:
                        # 使用 generate_similar_questions 生成問題
                        generated = self.question_generator.generate_similar_questions(
                            original_question=row["question"],
                            answer=row["answer"],
                            source_type="faq",
                            source_id=metadata["source_id"],
                            metadata=metadata
                        )
                        
                        # 提取問題文本
                        questions = [q.question for q in generated]
                        all_generated_questions.append((row, questions))
                        
                    except Exception as e:
                        logger.error(f"為問答對生成問題時出錯: {str(e)}")
                        continue
                
                # 如果是生成的問題，直接添加到結果中
                else:
                    all_generated_questions.append((row, [row["question"]]))
            
            if not all_generated_questions:
                logger.warning("沒有生成任何問題")
                return None
            
            # 準備新的數據
            new_rows = []
            for original_row, questions in all_generated_questions:
                # 如果是原始問題（非生成的），添加原始行
                if not original_row.get("is_generated", False):
                    new_row = dict(original_row)
                    new_row["is_generated"] = False
                    new_rows.append(new_row)
                
                # 添加生成的問題
                for q in questions:
                    # 如果是已經存在的生成問題，直接使用原始行
                    if original_row.get("is_generated", False):
                        new_rows.append(dict(original_row))
                    else:
                        # 如果是新生成的問題，創建新行
                        generated_row = {
                            "question": q,
                            "answer": original_row["answer"],  # 使用原始答案
                            "note": f"Generated from: {original_row['question']}",
                            "is_generated": True,
                            "original_question": original_row["question"]
                        }
                        new_rows.append(generated_row)
            
            # 創建新的 DataFrame
            new_df = pl.DataFrame(new_rows)
            
            # 生成輸出文件路徑
            input_path = Path(input_file)
            output_path = input_path.parent / f"{input_path.stem}_post{input_path.suffix}"
            
            # 保存到新的 CSV 文件
            new_df.write_csv(output_path)
            logger.info(f"已生成 {sum(len(q) for _, q in all_generated_questions)} 個問題並保存到: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"生成問題並保存時出錯: {str(e)}")
            return None

    def _normalize_text(self, text: str) -> str:
        """替換文本中的特殊字符
        
        Args:
            text: 原始文本
            
        Returns:
            str: 替換後的文本
        """
        # 定義需要替換的字符映射
        char_map = {
            '：': ':',  # 中文冒號
            '？': '?',  # 中文問號
            '！': '!',  # 中文驚嘆號
            '（': '(',  # 中文括號
            '）': ')',
            '【': '[',  # 中文方括號
            '】': ']',
            '「': '"',  # 中文引號
            '」': '"',
            '『': "'",  # 中文單引號
            '』': "'",
            '、': ',',  # 中文頓號
            '；': ';',  # 中文分號
            '，': ',',  # 中文逗號
            '。': '.',  # 中文句號
        }
        
        # 替換字符
        for cn_char, en_char in char_map.items():
            text = text.replace(cn_char, en_char)
        
        return text

    def _clean_json_value(self, value: Any) -> Any:
        """清理 JSON 值，確保其可以被正確序列化
        
        Args:
            value: 需要清理的值
            
        Returns:
            清理後的值
        """
        if value is None:
            return None  # JSON 的 null
        
        if isinstance(value, (str, int, float, bool)):
            return value
        
        if isinstance(value, dict):
            return {k: self._clean_json_value(v) for k, v in value.items() if v is not None}
        
        if isinstance(value, (list, tuple)):
            # 過濾掉 None 值，並清理每個元素
            return [self._clean_json_value(v) for v in value if v is not None]
        
        # 其他類型轉換為字符串
        return str(value)

    def _validate_json(self, data: Any, context: str = "") -> None:
        """驗證數據是否可以正確序列化為 JSON
        
        Args:
            data: 需要驗證的數據
            context: 上下文信息，用於錯誤提示
        """
        try:
            # 先清理數據
            cleaned_data = self._clean_json_value(data)
            
            # 嘗試序列化
            json_str = json.dumps(cleaned_data, ensure_ascii=False)
            
            # 驗證是否可以正確解析
            parsed = json.loads(json_str)
            
            # 檢查是否有尾隨逗號
            if json_str.rstrip().endswith(','):
                raise ValueError(f"{context} JSON 字符串包含尾隨逗號")
            
            # 檢查是否有未完成的鍵值對
            if '":' in json_str and not any(f'"{k}":' in json_str for k, v in cleaned_data.items() if v is not None):
                raise ValueError(f"{context} JSON 字符串包含未完成的鍵值對")
            
            return cleaned_data
            
        except json.JSONDecodeError as e:
            logger.error(f"{context} JSON 序列化失敗: {str(e)}")
            logger.error(f"原始數據: {data}")
            raise ValueError(f"{context} JSON 序列化失敗: {str(e)}")
        except Exception as e:
            logger.error(f"{context} 數據驗證失敗: {str(e)}")
            logger.error(f"原始數據: {data}")
            raise ValueError(f"{context} 數據驗證失敗: {str(e)}")

    def process_file(self, input_file: str, dry_run: bool = False) -> None:
        """處理 FAQ CSV 文件
        
        Args:
            input_file: 輸入的 CSV 文件路徑
            dry_run: 是否為試運行模式（只生成問題，不建立向量和索引）
        """
        try:
            # 檢查文件類型
            input_path = Path(input_file)
            if input_path.suffix.lower() != '.csv':
                raise ValueError(f"FAQ 處理器只支持 CSV 文件，收到: {input_path.suffix}")
            
            # 讀取 CSV 文件
            try:
                df = pl.read_csv(input_file, truncate_ragged_lines=True)
                logger.info(f"成功讀取 CSV 文件，共 {len(df)} 行")
                logger.info(f"CSV 文件的列: {df.columns}")
                
            except Exception as e:
                logger.error(f"讀取 CSV 文件時出錯: {str(e)}")
                raise ValueError(f"無法讀取 CSV 文件: {str(e)}")
            
            # 檢查必要的列
            required_columns = ["question", "answer"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"CSV 文件缺少必要的列: {', '.join(missing_columns)}")
            
            # 準備文本和元數據
            all_texts = []
            all_metadata = []
            
            # 處理每一行FAQ數據
            for i, row in enumerate(df.iter_rows(named=True)):
                # 檢查必要的字段是否存在且不為空
                if not row.get("question") or not row.get("answer"):
                    logger.warning(f"跳過第 {i+1} 行，缺少必要的字段:")
                    logger.warning(f"  問題: {row.get('question', '')}")
                    logger.warning(f"  答案: {row.get('answer', '')}")
                    continue
                
                # 生成或使用現有的qa_id
                qa_id = row.get('id') or get_text_hash(row['question'] + row['answer'])
                
                # 標準化問題和答案
                normalized_question = self._normalize_text(row['question'])
                normalized_answer = self._normalize_text(row['answer'])
                
                # 建立共享的元數據
                shared_metadata = {
                    "qa_id": qa_id,
                    "answer": normalized_answer,
                    "source_file": row.get('source_file'),
                    "source_page": row.get('source_page'),
                    "source_section": row.get('source_section'),
                    "source_image": row.get('source_image'),
                }
                
                # 1. 處理原始問題
                original_question_meta = shared_metadata.copy()
                original_question_meta.update({
                    "is_generated": False,
                    "original_question": normalized_question,
                    "text": normalized_question,
                    "text_hash": get_text_hash(normalized_question),
                    "vector_index": len(all_texts),
                    "metadata": json.dumps({
                        "type": "faq",
                        "is_generated": False,
                        "original_question": normalized_question,
                        "qa_id": qa_id
                    }, ensure_ascii=False)
                })
                
                all_texts.append(normalized_question)
                all_metadata.append(original_question_meta)
                
                logger.info(f"處理原始問題 (第 {i+1} 行): {normalized_question[:50]}...")
                
                # 2. 生成相似問題（如果啟用）
                if self.config.question_generation.enabled:
                    try:
                        generated_questions = self.question_generator.generate_similar_questions(
                            original_question=normalized_question,
                            answer=normalized_answer,
                            source_type="faq",
                            source_id=qa_id,
                            metadata=shared_metadata
                        )
                        
                        for gen_q in generated_questions:
                            generated_question_meta = shared_metadata.copy()
                            generated_question_meta.update({
                                "is_generated": True,
                                "original_question": normalized_question,
                                "text": gen_q.question,
                                "text_hash": get_text_hash(gen_q.question),
                                "vector_index": len(all_texts),
                                "metadata": json.dumps({
                                    "type": "faq",
                                    "is_generated": True,
                                    "original_question": normalized_question,
                                    "qa_id": qa_id
                                }, ensure_ascii=False)
                            })
                            
                            all_texts.append(gen_q.question)
                            all_metadata.append(generated_question_meta)
                            
                            logger.info(f"  生成問題: {gen_q.question[:50]}...")
                    
                    except Exception as e:
                        logger.warning(f"生成相似問題時出錯 (第 {i+1} 行): {str(e)}")
                        # 繼續處理，不中斷整個流程
            
            if not all_texts:
                logger.warning("CSV 文件中沒有有效的問答對")
                return
            
            logger.info(f"總共準備了 {len(all_texts)} 個問題（原始 + 生成）")
            
            # 如果是 dry-run 模式，到此為止
            if dry_run:
                logger.info("Dry run 完成，跳過向量生成和索引建立")
                return
            
            # 生成向量
            logger.info(f"為 {len(all_texts)} 個問題生成向量...")
            try:
                embedding_results, valid_indices = self.embedding_generator.generate_embeddings(all_texts)
            except Exception as e:
                logger.error(f"生成向量時出錯: {str(e)}")
                raise ValueError(f"向量生成失敗: {str(e)}")
            
            if not embedding_results:
                logger.error("沒有生成任何有效的向量")
                return
            
            # 只保留有效的文本和元數據
            try:
                vectors = np.array([r.vector for r in embedding_results])
                texts_to_store = [all_texts[i] for i in valid_indices]
                metadata_to_store = [all_metadata[i] for i in valid_indices]
                
                # 更新vector_index
                for i, metadata in enumerate(metadata_to_store):
                    metadata["vector_index"] = i
                
                assert len(vectors) == len(texts_to_store) == len(metadata_to_store), (
                    f"資料長度不一致: vectors={len(vectors)}, texts={len(texts_to_store)}, metadata={len(metadata_to_store)}"
                )
            except Exception as e:
                logger.error(f"處理向量結果時出錯: {str(e)}")
                raise ValueError(f"處理向量數據時出錯: {str(e)}")
            
            logger.info(f"成功生成 {len(vectors)} 個向量")
            
            # 檢查 collection 是否存在，如果不存在則創建
            try:
                info = self.collection_manager.get_collection_info(self.config.collection)
                if not info:
                    logger.info(f"創建新的 collection: {self.config.collection}")
                    
                    # 驗證源文件信息
                    source_file_info = {
                        "path": str(input_path),
                        "type": "faq_csv",
                        "total_rows": len(df),
                        "valid_rows": len(vectors),
                        "original_questions": len([m for m in metadata_to_store if not m.get("is_generated", False)]),
                        "generated_questions": len([m for m in metadata_to_store if m.get("is_generated", False)])
                    }
                    
                    # 創建collection
                    self.collection_manager.create_collection(
                        collection_name=self.config.collection,
                        dimension=len(vectors[0]),
                        source_files=[str(input_path)],
                        config=self.config.to_dict()
                    )
                else:
                    logger.info(f"更新現有的 collection: {self.config.collection}")
                
                # 更新 collection
                self.collection_manager.update_collection(
                    collection_name=self.config.collection,
                    vectors=vectors,
                    texts=texts_to_store,
                    metadata_list=metadata_to_store
                )
                
                logger.info(f"成功處理 FAQ 文件，共 {len(vectors)} 個問題向量")
                
            except Exception as e:
                logger.error(f"更新 collection 時出錯: {str(e)}")
                raise ValueError(f"Collection 更新失敗: {str(e)}")
                
        except Exception as e:
            logger.error(f"處理 FAQ 文件時出錯: {str(e)}")
            raise

    def rebuild_collection(self, file_path: str) -> None:
        """Rebuild collection from scratch"""
        try:
            # Delete existing collection
            self.collection_manager.delete_collection(self.config.collection)
        except ValueError as e:
            if "does not exist" not in str(e):
                raise
                
        # Process file to create new collection
        self.process_file(file_path) 