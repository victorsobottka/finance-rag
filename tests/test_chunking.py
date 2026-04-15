from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=80
    )
    return splitter.split_documents(docs)

def test_chunks_are_not_too_large():
    docs = [Document(page_content="word " * 300)]
    chunks = chunk_documents(docs)
    assert len(chunks) > 1, "Should split a long document"
    for chunk in chunks:
        assert len(chunk.page_content) <= 500, "Chunk too large"

def test_no_empty_chunks():
    docs = [Document(page_content="Apple revenue was $89.5 billion.")]
    chunks = chunk_documents(docs)
    assert all(len(c.page_content.strip()) > 0 for c in chunks)
