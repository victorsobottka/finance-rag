setup:
	conda env create -f environment.yml
	conda activate finance-rag

ingest:
	python ingest.py

embed:
	python vectorstore.py

test:
	pytest tests/ -v

lint:
	ruff check .

run:
	python app.py

deploy:
	git push origin main

.PHONY: setup ingest embed test lint run deploy
