# app.py — Gradio UI for Hugging Face Spaces
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
from edgar_fetcher import fetch_and_save, extract_ticker_from_text
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
# CACHE UTILS
# =============================================================================

def is_ticker_indexed(ticker: str) -> bool:
    try:
        results = vectorstore.get(where={"ticker": ticker.upper()}, limit=1)
        return len(results["ids"]) > 0
    except Exception:
        return False


def ensure_ticker_indexed(ticker: str) -> None:
    ticker = ticker.upper()
    if is_ticker_indexed(ticker):
        print(f"{ticker} already indexed")
        return

    print(f"Fetching {ticker} from SEC EDGAR...")
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
# ANSWER FUNCTION — no ticker parameter needed
# =============================================================================

# Track current company per session
current_ticker = {"value": "AAPL"}

def answer(message: str, history: list) -> str:
    if not message.strip():
        return "Please ask a question about a company's financial filing."

    # Detect company from message
    detected = extract_ticker_from_text(message)
    if detected:
        current_ticker["value"] = detected

    ticker = current_ticker["value"]

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
        response = chain.invoke(message)
        return f"[{ticker}] {response}"

    except ValueError:
        return (
            f"Could not find SEC filings for '{message}'. "
            f"Try using the official company name or ticker symbol."
        )
    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {str(e)}"


# =============================================================================
# GRADIO UI — no additional_inputs
# =============================================================================

demo = gr.ChatInterface(
    fn=answer,
    title="Finance RAG — SEC Filing Q&A",
    description=(
        "Ask questions about any US public company's latest 10-K filing. "
        "Just mention the company name — no ticker needed. "
        "Try: 'What was Apple's gross margin?' or 'Tell me about Nvidia's revenue.'"
    ),
    examples=[
        ["What was Apple's gross margin percentage?"],
        ["How did Nvidia's revenue grow last year?"],
        ["What are Microsoft's key risk factors?"],
        ["What is Google's cash position?"],
    ],
    type="messages",
)

if __name__ == "__main__":
    demo.launch()
