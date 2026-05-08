# rag_chain.py
# Complete RAG chain — retrieval + generation
# conda activate finance-rag

__import__('pysqlite3')

from dotenv import load_dotenv
load_dotenv("/home/sobottka/Documents/Projects/finance-rag/.env")

import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor


# =============================================================================
# RETRIEVAL
# =============================================================================

def build_retriever(vectorstore, search_type="mmr"):
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )

def build_compressed_retriever(vectorstore):
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)
    base_retriever = build_retriever(vectorstore)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )


# =============================================================================
# GENERATION
# =============================================================================

FINANCE_PROMPT = ChatPromptTemplate.from_template("""
You are a financial analyst assistant serving CFOs and investors.
Answer questions using ONLY the provided context from financial documents.
Be precise with numbers. Always cite the source document.
If the context does not contain enough information, say so clearly.

Context from financial documents:
{context}

Question: {question}

Answer (be specific, cite figures, note any caveats):
""")

def format_docs(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        ticker = doc.metadata.get("ticker", "Unknown")
        doc_type = doc.metadata.get("doc_type", "filing")
        formatted.append(f"[Source {i} — {ticker} {doc_type}]\n{doc.page_content}")
    return "\n\n".join(formatted)


# =============================================================================
# FULL CHAIN
# =============================================================================

def build_rag_chain(vectorstore, retriever=None):
    if retriever is None:
        retriever = build_retriever(vectorstore)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | FINANCE_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain

# =============================================================================
# TEST — run directly to verify end-to-end
# =============================================================================

if __name__ == "__main__":
    from vectorstore import load_vectorstore

    print("Loading vectorstore...")
    vs = load_vectorstore()

    print("Building RAG chain...")
    chain = build_rag_chain(vs)

    test_questions = [
        "What was the gross margin percentage last quarter?",
        "How did revenue compare to analyst expectations?",
        "What are the key risk factors mentioned?",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        print(f"A: {chain.invoke(q)}")
        print("-" * 60)
