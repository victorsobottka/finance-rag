# rag_chain.py  (retrieval half)
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import ContextualCompressionRetriever
# from langchain.retrievers import ContextualCompressionRetriever
from vectorstore import load_vectorstore
from langchain_community.document_compressors import LLMChainExtractor  # ← fixed

def build_retriever(vectorstore, search_type="mmr"):
    """
    MMR (Maximal Marginal Relevance) retrieval avoids returning
    5 chunks that all say the same thing. It balances relevance
    WITH diversity — critical for finance docs where sections repeat.
    """
    base_retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={
            "k": 6,           # fetch 6 candidates
            "fetch_k": 20,    # from a pool of 20
            "lambda_mult": 0.7  # 0=max diversity, 1=max relevance
        }
    )
    return base_retriever

def build_compressed_retriever(vectorstore):
    """
    Optional: LLM-powered compression strips irrelevant sentences
    from each retrieved chunk before passing to the generator.
    Costs more tokens but improves answer precision significantly.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)
    
    base_retriever = build_retriever(vectorstore)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

# Test retrieval
if __name__ == "__main__":
    vs = load_vectorstore()
    retriever = build_retriever(vs)
    
    docs = retriever.invoke("What was Apple's gross margin in Q4?")
    print(f"Retrieved {len(docs)} chunks:")
    for d in docs:
        print(f"  [{d.metadata.get('ticker','?')}] {d.page_content[:80]}...")

# rag_chain.py  (generation half)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from vectorstore import load_vectorstore
from rag_chain import build_retriever

FINANCE_PROMPT = ChatPromptTemplate.from_template("""
You are a financial analyst assistant serving CFOs and investors.
Answer questions using ONLY the provided context from financial documents.
Be precise with numbers. Always cite the source document.
If the context doesn't contain enough information, say so clearly.

Context from financial documents:
{context}

Question: {question}

Answer (be specific, cite figures, note any caveats):
""")

def format_docs(docs):
    """Format retrieved docs with source attribution."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        ticker = doc.metadata.get("ticker", "Unknown")
        doc_type = doc.metadata.get("doc_type", "filing")
        formatted.append(f"[Source {i} — {ticker} {doc_type}]\n{doc.page_content}")
    return "\n\n".join(formatted)

def build_rag_chain(vectorstore):
    """Full RAG chain using LCEL (LangChain Expression Language)."""
    retriever = build_retriever(vectorstore)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
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

# Test end-to-end
if __name__ == "__main__":
    vs = load_vectorstore()
    chain = build_rag_chain(vs)
    
    questions = [
        "What was the gross margin percentage last quarter?",
        "How did revenue compare to analyst expectations?",
        "What are the key risk factors mentioned?",
    ]
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {chain.invoke(q)}")
        print("---")        
