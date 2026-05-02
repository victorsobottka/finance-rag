# vectorstore.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from ingest import load_pdf, chunk_documents

def build_vectorstore(chunks: list, use_openai=False):
    """
    Build and persist a ChromaDB vector store.
    HuggingFace embeddings = free, good for dev.
    OpenAI embeddings = better quality, costs ~$0.001 per 1K tokens.
    """
    if use_openai:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        # Free, runs locally — good for iteration
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    # Add metadata to each chunk for filtering later
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "chunk_id": i,
            "ticker": chunk.metadata.get("source", "unknown").upper(),
            "doc_type": "earnings_report"  # or "10K", "10Q", etc.
        })

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="finance_docs"
    )
    vectorstore.persist()
    print(f"Stored {len(chunks)} chunks in ChromaDB")
    return vectorstore

def load_vectorstore(use_openai=False):
    """Load existing ChromaDB — no re-embedding needed."""
    embeddings = OpenAIEmbeddings() if use_openai else HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="finance_docs"
    )

if __name__ == "__main__":
    docs = load_pdf("data/aapl_10k_2024.pdf")
    chunks = chunk_documents(docs)
    vs = build_vectorstore(chunks)
    
    # Quick sanity check
    results = vs.similarity_search("What was the revenue?", k=3)
    for r in results:
        print(r.page_content[:100])
