![CI](https://github.com/victorsobottka/finance-rag/actions/workflows/ci.yml/badge.svg)

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

Ask questions about Apple's 10-K filing using RAG (Retrieval-Augmented Generation).
Powered by Llama 3.1 (Groq) + LangChain + ChromaDB.

## Stack
- **LLM:** Llama 3.1 8B via Groq (free)
- **Embeddings:** HuggingFace all-MiniLM-L6-v2 (local, free)
- **Vector store:** ChromaDB
- **Framework:** LangChain LCEL
- **UI:** Gradio 6

## How it works
1. Apple 10-K PDF is chunked into 400-token passages
2. Each chunk is embedded and stored in ChromaDB
3. Your question is matched against chunks using MMR retrieval
4. Top 6 chunks are passed to Llama 3.1 with a finance-specific prompt
5. Answer is returned with source citations



# finance-rag
A finance RAG that answers questions over earnings reports, SEC filings, or your own domain docs and it is deployed on Hugging Face Spaces. 
