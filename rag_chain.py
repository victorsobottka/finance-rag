# rag_chain.py
# Complete RAG chain — retrieval + generation
# conda activate finance-rag

__import__('pysqlite3')

from dotenv import load_dotenv
load_dotenv("/home/sobottka/Documents/Projects/finance-rag/.env")

import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langfuse.langchain import CallbackHandler

# =============================================================================
# LangFuse
# =============================================================================


def get_langfuse_handler():
    """Returns a Langfuse callback handler if keys are configured."""
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        return CallbackHandler()
    return None


# =============================================================================
# RETRIEVAL
# =============================================================================

def build_retriever(vectorstore, search_type="mmr"):
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={
            "k": 8,
            "fetch_k": 50,
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
Answer using ONLY the provided context from financial documents.

STRICT RULES:
- Never calculate or derive figures — only report numbers explicitly stated in the context
- If you cannot find the exact figure asked for, say "The exact figure is not in the retrieved context"
- Always cite the source: [Source N — TICKER 10-K]
- Net income and profit are the same — look for "net income" in the context
- Be precise — never approximate or infer

Context from financial documents:
{context}

Question: {question}

Answer (cite exact figures from context only, no calculations):
""")

def format_docs(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        ticker = doc.metadata.get("ticker", "Unknown")
        doc_type = doc.metadata.get("doc_type", "filing")
        formatted.append(f"[Source {i} — {ticker} {doc_type}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def rewrite_query(question: str) -> str:
    """
    Rewrite vague financial questions to match exact SEC filing terminology.
    Improves retrieval by targeting the right sections.
    """
    rewrites = {
        "profit": "net income",
        "earnings": "net income",
        "how much did they make": "net income",
        "revenue growth": "total revenues year over year increase",
        "gross margin": "gross margin percentage",
        "cash": "cash and cash equivalents",
    }
    q_lower = question.lower()
    for vague, precise in rewrites.items():
        if vague in q_lower:
            q_lower = q_lower.replace(vague, precise)
    return q_lower

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
