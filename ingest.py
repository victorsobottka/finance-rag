# ingest.py
import requests
from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_sec_filing(ticker: str, form="10-K") -> list:
    """Fetch latest 10-K from SEC EDGAR as text."""
    cik_map = {"AAPL":"0000320193","MSFT":"0000789019","GOOGL":"0001652044"}
    cik = cik_map.get(ticker.upper(), "")
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {"User-Agent": "finance-rag your@email.com"}
    data = requests.get(url, headers=headers).json()
    company = data.get("name","")
    print(f"Loaded SEC data for {company}")
    # In production: follow accession numbers to fetch actual filing text
    return [{"content": f"{company} {form} filing", "source": ticker}]

def load_pdf(path: str) -> list:
    """Load a local earnings report PDF."""
    loader = PyPDFLoader(path)
    return loader.load()

def chunk_documents(docs: list) -> list:
    """
    Finance-specific chunking strategy.
    Smaller chunks (400 tokens) with big overlap (80 tokens)
    because financial facts are dense — one sentence can hold
    a critical number that needs surrounding context to mean anything.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks from {len(docs)} pages")
    return chunks

# Run ingestion
if __name__ == "__main__":
    # Option A: load from local PDF
    docs = load_pdf("data/aapl_10k_2024.pdf")
    # Option B: load all PDFs in folder
    # for f in os.listdir("data"):
    #     docs += load_pdf(f"data/{f}")
    chunks = chunk_documents(docs)
    print(f"Ready to embed: {len(chunks)} chunks")
