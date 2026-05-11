# edgar_fetcher.py
import requests
from pathlib import Path
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "finance-rag victorsobottka@gmail.com"}

_SEC_TICKERS = None

def get_sec_tickers() -> dict:
    """Load full SEC ticker list — cached after first call."""
    global _SEC_TICKERS
    if _SEC_TICKERS is None:
        url = "https://www.sec.gov/files/company_tickers.json"
        data = requests.get(url, headers=HEADERS).json()
        _SEC_TICKERS = {
            entry["ticker"].upper(): {
                "cik": str(entry["cik_str"]).zfill(10),
                "name": entry["title"].lower()
            }
            for entry in data.values()
        }
    return _SEC_TICKERS

def get_cik(ticker: str) -> str:
    """Resolve ticker to zero-padded CIK."""
    tickers = get_sec_tickers()
    entry = tickers.get(ticker.upper())
    if not entry:
        raise ValueError(f"Ticker '{ticker}' not found in SEC database")
    return entry["cik"]

def search_company(query: str) -> str | None:
    """
    Find ticker from any company name or ticker string.
    Handles: "nvidia", "NVDA", "jp morgan", "berkshire hathaway"
    Returns ticker string or None if not found.
    """
    tickers = get_sec_tickers()
    query_clean = query.lower().strip()

    if not query_clean:
        return None

    # 1. Exact ticker match
    if query_clean.upper() in tickers:
        return query_clean.upper()

    best_ticker = None
    best_score = 0.0

    for ticker, entry in tickers.items():
        name = entry["name"]  # already lowercase

        # 2. Exact company name match
        if query_clean == name:
            return ticker

        # 3. Company name starts with query
        if name.startswith(query_clean):
            score = len(query_clean) / len(name)
            if score > best_score:
                best_score = score
                best_ticker = ticker
            continue

        # 4. Query contained in company name
        if query_clean in name:
            score = len(query_clean) / len(name)
            if score > best_score:
                best_score = score
                best_ticker = ticker

    # Require at least 60% name coverage to avoid false matches
    if best_ticker and best_score >= 0.6:
        return best_ticker

    return None

# Words that should never be treated as company names
STOPWORDS = {
    "what", "was", "the", "apple", "gross", "margin", "percentage",
    "profit", "revenue", "income", "net", "how", "did", "tell", "me",
    "about", "is", "are", "for", "and", "its", "their", "last", "year",
    "cash", "sales", "cost", "total", "share", "stock", "price", "rate",
    "growth", "loss", "debt", "risk", "market", "financial", "report",
    "quarter", "annual", "fiscal", "per", "with", "from", "that", "this",
    "high", "low", "more", "less", "than", "which", "when", "where",
}

def extract_ticker_from_text(text: str) -> str | None:
    """
    Scan a sentence for company names or tickers.
    Filters common English words to avoid false matches.
    """
    words = text.replace("?", "").replace(".", "").replace(",", "").split()

    # Try 3-word, 2-word, then 1-word phrases
    for n in [3, 2, 1]:
        for i in range(len(words) - n + 1):
            phrase_words = words[i:i+n]

            # Skip if any word in phrase is a stopword (for n=1 and n=2)
            if n <= 2 and any(w.lower() in STOPWORDS for w in phrase_words):
                continue

            phrase = " ".join(phrase_words)
            result = search_company(phrase)
            if result:
                print(f"Detected: '{phrase}' → {result}")
                return result

    return None

def fetch_and_save(ticker: str, form_type: str = "10-K") -> str:
    """Fetch filing from SEC EDGAR, strip HTML, save as clean text."""
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
            print(f"Saved {ticker} {form_type} → {path}")
            return path

    raise ValueError(f"No {form_type} found for {ticker}")
