# edgar_fetcher.py
import json
import requests
from pathlib import Path
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "finance-rag victorsobottka@gmail.com"}

# Load company index once at import time
with open("company_index.json", "r") as f:
    _INDEX = json.load(f)

TICKER_TO_CIK = _INDEX["ticker_to_cik"]      # {"AAPL": "0000320193", ...}
NAME_TO_TICKER = _INDEX["name_to_ticker"]    # {"apple inc": "AAPL", ...}


def get_cik(ticker: str) -> str:
    """Resolve ticker to zero-padded CIK."""
    cik = TICKER_TO_CIK.get(ticker.upper())
    if not cik:
        raise ValueError(f"Ticker '{ticker}' not found in company index")
    return cik


def extract_ticker_from_text(text: str) -> str | None:
    """
    Extract company ticker from natural language message.
    Strategy:
      1. Explicit ALL-CAPS ticker (e.g. AAPL, NVDA)
      2. Known alias map (e.g. "apple" → AAPL)
      3. Full SEC company name match from company_index.json
    """
    words = text.replace("?", "").replace(".", "").replace(
        ",", "").replace("'s", "").split()

    # 1. Explicit uppercase ticker
    for word in words:
        clean = word.strip(".,?!'\"")
        if clean.isupper() and 2 <= len(clean) <= 5 and clean.isalpha():
            if clean in TICKER_TO_CIK:
                print(f"Detected explicit ticker: {clean}")
                return clean

    # 2. Alias map — common names
    ALIASES = {
        "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA",
        "google": "GOOGL", "alphabet": "GOOGL", "amazon": "AMZN",
        "meta": "META", "facebook": "META", "tesla": "TSLA",
        "netflix": "NFLX", "jpmorgan": "JPM", "jp morgan": "JPM",
        "goldman sachs": "GS", "goldman": "GS", "berkshire": "BRK-B",
        "spotify": "SPOT", "uber": "UBER", "airbnb": "ABNB",
        "palantir": "PLTR", "salesforce": "CRM", "adobe": "ADBE",
        "intel": "INTC", "amd": "AMD", "disney": "DIS",
        "walmart": "WMT", "visa": "V", "mastercard": "MA",
        "paypal": "PYPL", "coinbase": "COIN", "shopify": "SHOP",
    }

    text_lower = text.lower()
    for n in [2, 1]:
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i+n]).lower().strip(".,?!'\"")
            if phrase in ALIASES:
                print(f"Detected alias: '{phrase}' → {ALIASES[phrase]}")
                return ALIASES[phrase]

    # 3. Full company name lookup from SEC index
    # Try 3-word, 2-word, 1-word phrases
    for n in [3, 2, 1]:
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i+n]).lower().strip(".,?!'\"")
            if phrase in NAME_TO_TICKER:
                ticker = NAME_TO_TICKER[phrase]
                print(f"Detected from SEC index: '{phrase}' → {ticker}")
                return ticker

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
