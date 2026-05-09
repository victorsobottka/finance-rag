# app.py — Gradio UI for Hugging Face Spaces
# conda activate finance-rag

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import gradio as gr
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from rag_chain import build_rag_chain
from edgar_fetcher import fetch_and_save
from ingest import chunk_documents


# =============================================================================
# GLOBALS — loaded once on startup
# =============================================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="finance_docs"
)


# =============================================================================
# CACHE UTILS — fetch + index on demand
# =============================================================================

def is_ticker_indexed(ticker: str) -> bool:
    """Check if a ticker already has chunks in ChromaDB."""
    try:
        results = vectorstore.get(
            where={"ticker": ticker.upper()},
            limit=1
        )
        return len(results["ids"]) > 0
    except Exception:
        return False


def ensure_ticker_indexed(ticker: str) -> None:
    """
    Fetch + index a ticker from SEC EDGAR if not already in ChromaDB.
    Returns immediately if already indexed.
    """
    ticker = ticker.upper()
    if is_ticker_indexed(ticker):
        print(f"{ticker} already indexed — skipping fetch")
        return

    print(f"Fetching {ticker} 10-K from SEC EDGAR...")
    path = fetch_and_save(ticker, form_type="10-K")

    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = ticker
        doc.metadata["ticker"] = ticker
        doc.metadata["doc_type"] = "10-K"

    chunks = chunk_documents(docs)
    vectorstore.add_documents(chunks)
    print(f"{ticker} indexed — {len(chunks)} chunks added")


# =============================================================================
# PRE-LOAD AAPL on startup
# =============================================================================

print("Loading RAG system...")
ensure_ticker_indexed("AAPL")
print("Ready.")


# =============================================================================
# ANSWER FUNCTION
# =============================================================================

def answer(message: str, history: list, ticker: str) -> str:
    ticker = ticker.strip().upper()

    if not ticker:
        return "Please enter a ticker symbol (e.g. AAPL, NVDA, JPM)."

    if not message.strip():
        return "Please enter a question about the filing."

    try:
        ensure_ticker_indexed(ticker)

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 6,
                "fetch_k": 20,
                "filter": {"ticker": ticker}
            }
        )
        chain = build_rag_chain(vectorstore, retriever=retriever)
        return chain.invoke(message)

    except ValueError:
        return (
            f"'{ticker}' was not found in SEC EDGAR. "
            f"Please check the ticker symbol and try again."
        )
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# GRADIO UI
# =============================================================================

demo = gr.ChatInterface(
    fn=answer,
    additional_inputs=[
        gr.Textbox(
            value="AAPL",
            label="Ticker symbol",
            placeholder="e.g. AAPL, NVDA, JPM, NFLX...",
            max_lines=1,
        )
    ],
    title="Finance RAG — SEC Filing Q&A",
    description=(
        "Ask questions about any US public company's latest 10-K. "
        "Enter any ticker — data pulled live from SEC EDGAR. "
        "Powered by Llama 3.1 (Groq) + LangChain."
    ),
    examples=[
        ["What was the gross margin percentage?", "AAPL"],
        ["What are the key risk factors?", "AAPL"],
        ["How did revenue grow year over year?", "MSFT"],
        ["What is the cash and equivalents position?", "NVDA"],
    ],
    )
if __name__ == "__main__":
    demo.launch()
