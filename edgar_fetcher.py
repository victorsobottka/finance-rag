# edgar_fetcher.py
import requests
from pathlib import Path
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "finance-rag victorsobottka@gmail.com"}

def get_cik(ticker: str) -> str:
    """Resolve any US ticker to its SEC CIK automatically."""
    url = "https://www.sec.gov/files/company_tickers.json"
    data = requests.get(url, headers=HEADERS).json()
    ticker_upper = ticker.upper()
    for entry in data.values():
        if entry["ticker"].upper() == ticker_upper:
            # Zero-pad to 10 digits as SEC requires
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"Ticker '{ticker}' not found in SEC database")

def fetch_and_save(ticker: str, form_type: str = "10-K") -> str:
    cik = get_cik(ticker)
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = requests.get(url, headers=HEADERS).json()
    filings = data["filings"]["recent"]

    for i, form in enumerate(filings["form"]):
        if form == form_type:
            accession = filings["accessionNumber"][i].replace("-", "")
            doc_name = filings["primaryDocument"][i]
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{accession}/{doc_name}"
            )
            resp = requests.get(filing_url, headers=HEADERS)

            # Strip HTML tags — plain text chunks much better for RAG
            soup = BeautifulSoup(resp.text, "html.parser")
            clean_text = soup.get_text(separator="\n", strip=True)

            Path("data").mkdir(exist_ok=True)
            path = f"data/{ticker.upper()}_{form_type}.txt"
            with open(path, "w", encoding="utf-8") as f:
                f.write(clean_text)
            print(f"Fetched {form_type} for {ticker} → {path}")
            return path

    raise ValueError(f"No {form_type} found for {ticker}")
