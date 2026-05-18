import time
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vectorstore import load_vectorstore
from rag_chain import build_rag_chain
from edgar_fetcher import extract_ticker_from_text

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

app = FastAPI(
    title="Finance RAG API",
    description="Query any US public company's 10-K filing via SEC EDGAR",
    version="1.0"
)

# Load once on startup
vs = load_vectorstore()

class QueryRequest(BaseModel):
    question: str
    company: str = ""   # optional — detected from question if empty

class QueryResponse(BaseModel):
    answer: str
    ticker: str
    question: str
    latency_ms: int

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    start = time.time()

    # Resolve ticker
    ticker = extract_ticker_from_text(request.company or request.question)
    if not ticker:
        raise HTTPException(
            status_code=400,
            detail="Could not identify a company. Include company name or ticker."
        )

    try:
        retriever = vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 30, "filter": {"ticker": ticker}}
        )
        chain = build_rag_chain(vs, retriever=retriever)
        answer = chain.invoke(request.question)
        latency = int((time.time() - start) * 1000)

        return QueryResponse(
            answer=answer,
            ticker=ticker,
            question=request.question,
            latency_ms=latency
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0"}

# Run: uvicorn api:app --reload --port 8000
