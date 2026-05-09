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
from edgar_fetcher import fetch_and_save, search_company
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
# EXTRACT TICKER FUNCTION
# =============================================================================

def extract_ticker(message: str, default_ticker: str) -> str:
    """
    Detect any company name or ticker in the message using SEC database.
    Works for any of the 10,000+ companies registered with the SEC.
    """
    words = message.split()

    # Check 1, 2, and 3-word combinations in the message
    for n in range(3, 0, -1):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i+n])
            result = search_company(phrase)
            if result:
                print(f"Detected ticker {result} from '{phrase}'")
                return result

    return default_ticker

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
       # Tell user what's happening if fetching for first time
        if not is_ticker_indexed(ticker):
           print(f"Triggering fetch for {ticker}...")
           ensure_ticker_indexed(ticker)
           print(f"{ticker} now indexed")

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

    except ValueError as e:
        print(f"ValueError for {ticker}: {e}")
        return (
            f"'{ticker}' was not found in SEC EDGAR. "
            f"Please check the ticker symbol and try again."
        )
    except Exception as e:
        print(f"Unexpected error for {ticker}: {e}")
        return f"Error processing {ticker}: {str(e)}"


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
