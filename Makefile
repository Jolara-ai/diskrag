.PHONY: help install demo clean test process-faq search-faq run-api verify

# Virtual Environment Path
VENV_DIR := venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip

# Use venv python if it exists, otherwise system python
PYTHON := $(shell if [ -f $(VENV_PYTHON) ]; then echo $(VENV_PYTHON); else echo python3; fi)
PIP := $(shell if [ -f $(VENV_PIP) ]; then echo $(VENV_PIP); else echo pip3; fi)

help:
	@echo "DiskRAG 可用命令："
	@echo ""
	@echo "安裝與設定："
	@echo "  make install           - 安裝依賴套件並設定環境（必須先執行）"
	@echo ""
	@echo "快速體驗："
	@echo "  make demo              - 快速體驗（處理資料、建立索引、搜尋測試）"
	@echo "                         注意：需先執行 make install"
	@echo ""
	@echo "資料處理："
	@echo "  make process-faq       - 處理 FAQ 資料（需要 ARGS='collection_name csv_file'）"
	@echo "  make search-faq        - 搜尋 FAQ（需要 ARGS='collection_name query'）"
	@echo ""
	@echo "服務與測試："
	@echo "  make run-api           - 啟動 API 服務"
	@echo "  make test              - 執行測試"
	@echo "  make verify            - 驗證安裝"
	@echo ""
	@echo "清理："
	@echo "  make clean             - 清理編譯產物和暫存檔"
	@echo ""
	@echo "範例："
	@echo "  make demo"
	@echo "  make process-faq ARGS='my_faq data/faq.csv'"
	@echo "  make search-faq ARGS='my_faq \"你的問題\"'"

demo:
	@echo "開始快速體驗..."
	@./scripts/demo.sh

install:
	@echo "使用 uv 安裝依賴套件..."
	@./scripts/install.sh

clean:
	@echo "清理編譯產物和暫存檔..."
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.so" -delete
	@find . -name "*.c" -delete

test:
	@echo "執行測試..."
	@PYTHONPATH=. $(PYTHON) scripts/test_faq_workflow.py
	@./scripts/test_pydiskann_cython.sh

run-api:
	@echo "啟動 API 服務..."
	@$(PYTHON) app.py

process-faq:
	@if [ -z "$(ARGS)" ]; then \
		echo "錯誤: 需要提供 ARGS 參數"; \
		echo "用法: make process-faq ARGS='collection_name csv_file'"; \
		echo "範例: make process-faq ARGS='my_faq data/faq.csv'"; \
		exit 1; \
	fi
	@./scripts/process_faq.sh $(ARGS)

search-faq:
	@if [ -z "$(ARGS)" ]; then \
		echo "錯誤: 需要提供 ARGS 參數"; \
		echo "用法: make search-faq ARGS='collection_name query'"; \
		echo "範例: make search-faq ARGS='my_faq \"你的問題\"'"; \
		exit 1; \
	fi
	@./scripts/search_faq.sh $(ARGS)

verify:
	@echo "驗證安裝..."
	@PYTHONPATH=. $(PYTHON) scripts/verify_installation.py
