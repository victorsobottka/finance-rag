# ingest.py — cleaned up version
# load_pdf() removed — edgar_fetcher.py handles all data loading now
# chunk_documents() is still used by app.py

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(docs: list) -> list:
    """
    Split documents into finance-optimised chunks.
    400 tokens with 80 overlap — preserves numerical context.
    Called from app.py after edgar_fetcher saves the filing text.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks from {len(docs)} pages")
    return chunks
