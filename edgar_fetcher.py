import requests
from pathlib import Path
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "finance-rag victorsobottka@gmail.com"}

# Cache the SEC ticker list in memory so we don't fetch it on every call
_SEC_TICKERS = None

def get_sec_tickers() -> dict:
    """Load the full SEC company/ticker list — cached after first call."""
    global _SEC_TICKERS
    if _SEC_TICKERS is None:
        url = "https://www.sec.gov/files/company_tickers.json"
        data = requests.get(url, headers=HEADERS).json()
        _SEC_TICKERS = {
            entry["ticker"].upper(): entry
            for entry in data.values()
        }
    return _SEC_TICKERS

def get_cik(ticker: str) -> str:
    """Resolve a ticker symbol to a zero-padded CIK."""
    tickers = get_sec_tickers()
    entry = tickers.get(ticker.upper())
    if not entry:
        raise ValueError(f"Ticker '{ticker}' not found in SEC database")
    return str(entry["cik_str"]).zfill(10)

def search_company(query: str) -> str | None:
    """
    Search SEC database for a company by name or ticker.
    Returns the matched ticker symbol, or None if not found.

    Examples:
        search_company("nvidia")   → "NVDA"
        search_company("apple")    → "AAPL"
        search_company("jp morgan")→ "JPM"
        search_company("TSLA")     → "TSLA"
    """
    tickers = get_sec_tickers()
    query_upper = query.upper().strip()
    query_lower = query.lower().strip()

    # 1. Exact ticker match first (fastest)
    if query_upper in tickers:
        return query_upper

    # 2. Search company names — look for query anywhere in the name
    best_match = None
    best_score = 0

    for ticker, entry in tickers.items():
        company_name = entry.get("title", "").lower()

        # Exact company name match
        if query_lower == company_name:
            return ticker

        # Query is contained in company name
        if query_lower in company_name:
            # Score by how much of the name the query covers
            score = len(query_lower) / len(company_name)
            if score > best_score:
                best_score = score
                best_match = ticker

    # Only return if reasonably confident
    if best_match and best_score > 0.3:
        return best_match

    return None

def fetch_and_save(ticker: str, form_type: str = "10-K") -> str:
    """Fetch a filing from SEC EDGAR and save as clean text."""
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
            soup = BeautifulSoup(resp.text, "html.parser")
            clean_text = soup.get_text(separator="\n", strip=True)

            Path("data").mkdir(exist_ok=True)
            path = f"data/{ticker.upper()}_{form_type}.txt"
            with open(path, "w", encoding="utf-8") as f:
                f.write(clean_text)
            print(f"Fetched {form_type} for {ticker} → {path}")
            return path

    raise ValueError(f"No {form_type} found for {ticker}")
