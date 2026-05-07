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

def answer(message: str, history: list) -> str:
    if not message.strip():
        return "Please enter a question about the financial documents."
    try:
        return chain.invoke(message)
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# GRADIO UI — Gradio 6 compatible
# =============================================================================

demo = gr.ChatInterface(
    fn=answer,
    title="Finance RAG — Earnings Report Q&A",
    description=(
        "Ask questions about Apple's 10-K filing. "
        "Powered by Llama 3.1 (Groq) + LangChain RAG. "
        "Every answer cites the source passage."
    ),
    examples=[
        "What was the gross margin percentage?",
        "What are the key risk factors?",
        "How did services revenue perform?",
        "What is the current cash position?",
    ],
    type="messages",
)

if __name__ == "__main__":
    demo.launch()
