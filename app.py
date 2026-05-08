# app.py — Gradio UI for Hugging Face Spaces
# conda activate finance-rag

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from pathlib import Path
import gradio as gr
from dotenv import load_dotenv
load_dotenv("/home/sobottka/Documents/Projects/finance-rag/.env")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rag_chain import build_rag_chain


# =============================================================================
# LOAD CHAIN — runs once on startup
# =============================================================================

def load_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    if not Path("./chroma_db").exists():
        print("No vectorstore found — building from PDFs in data/...")
        from ingest import load_pdf, chunk_documents
        from vectorstore import build_vectorstore

        docs = []
        for pdf in Path("./data").glob("*.pdf"):
            print(f"  Loading {pdf.name}...")
            docs += load_pdf(str(pdf))

        if not docs:
            raise FileNotFoundError("No PDFs found in data/ folder.")

        chunks = chunk_documents(docs)
        vs = build_vectorstore(chunks, embeddings=embeddings)
    else:
        print("Loading existing vectorstore...")
        vs = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings,
            collection_name="finance_docs"
        )
    return build_rag_chain(vs)


print("Loading RAG chain...")
chain = load_chain()
print("Ready.")


# =============================================================================
# ANSWER FUNCTION
# =============================================================================

def answer(message: str, history: list, ticker: str) -> str:
    ticker = ticker.strip().upper()
    if not ticker:
        return "Please enter a ticker symbol (e.g. AAPL, NVDA, JPM)."
    try:
        ensure_ticker_indexed(vectorstore, embeddings, ticker)
        # ... rest of chain
    except ValueError as e:
        return f"'{ticker}' not found in SEC database. Please check the ticker and try again."
    except Exception as e:
        return f"Error: {str(e)}"

# =============================================================================
# GRADIO UI — Gradio 6 compatible
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
        "Enter any ticker — data pulled live from SEC EDGAR."
    ),
    type="messages",
)

if __name__ == "__main__":
    demo.launch()
