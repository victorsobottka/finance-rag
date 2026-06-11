---
title: Finance RAG
emoji: 📊
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "6.10.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# Finance RAG — Earnings Report Q&A

![CI](https://github.com/victorsobottka/finance-rag/actions/workflows/ci.yml/badge.svg)

A finance RAG that answers questions over earnings reports and SEC filings, deployed on Hugging Face Spaces.

## How it works
1. Apple 10-K PDF is chunked into 400-token passages
2. Each chunk is embedded and stored in ChromaDB
3. Your question is matched against chunks using MMR retrieval
4. Top 6 chunks are passed to Llama 3.1 with a finance-specific prompt
5. Answer is returned with source citations

## Stack
- **LLM:** Llama 3.1 8B via Groq (free)
- **Embeddings:** HuggingFace all-MiniLM-L6-v2 (local, free)
- **Vector store:** ChromaDB
- **Framework:** LangChain LCEL
- **UI:** Gradio 6


## Evaluation (RAGAS)

| Metric | Score | Target |
|--------|-------|--------|
| Faithfulness | 0.80 | > 0.70 |
| Context Precision | 0.90 | > 0.70 |

Evaluated on 5 AAPL questions. Run: `python tests/eval_rag.py`
