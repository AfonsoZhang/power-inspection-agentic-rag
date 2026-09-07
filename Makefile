.PHONY: install install-dev index test lint eval-retrieval eval-kpi eval-judge app

install:            ## 安装全部运行时依赖
	pip install -r requirements.txt

install-dev:        ## 只装单测需要的轻量依赖
	pip install -r requirements-dev.txt

index:              ## 构建 ChromaDB 向量索引
	python scripts/build_index.py

test:               ## 跑单测（不需要 API Key，不联网）
	pytest -q

lint:
	ruff check .

eval-retrieval:     ## 检索质量评测（只需本地 embedding，无需 API Key）
	python eval/retrieval_eval.py

eval-kpi:           ## 业务 KPI（需要 API Key）
	python eval/business_kpi.py

eval-judge:         ## LLM-as-Judge 三模式对比（需要 API Key）
	python eval/ragas_eval.py

app:
	streamlit run app/streamlit_app.py --server.headless true
