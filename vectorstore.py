# vectorstore.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def build_vectorstore(chunks: list, embeddings=None, use_openai=False):
    """
    Build and persist a ChromaDB vector store.
    HuggingFace embeddings = free, good for dev.
    OpenAI embeddings = better quality, costs ~$0.001 per 1K tokens.
    """
    if embeddings is None:
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


