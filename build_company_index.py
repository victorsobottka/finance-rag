# build_company_index.py
# Run this ONCE locally to generate company_index.json
# Then commit company_index.json to your repo

import requests
import json

HEADERS = {"User-Agent": "finance-rag victorsobottka@gmail.com"}

def build_index():
    print("Downloading SEC company list...")
    url = "https://www.sec.gov/files/company_tickers.json"
    data = requests.get(url, headers=HEADERS).json()

    # Build two lookup structures:
    # 1. ticker → cik (for fetch_and_save)
    # 2. company_name → ticker (for name search)
    ticker_to_cik = {}
    name_to_ticker = {}

    for entry in data.values():
        ticker = entry["ticker"].upper()
        cik = str(entry["cik_str"]).zfill(10)
        name = entry["title"].lower().strip()

        ticker_to_cik[ticker] = cik
        name_to_ticker[name] = ticker

    index = {
        "ticker_to_cik": ticker_to_cik,
        "name_to_ticker": name_to_ticker
    }

    with open("company_index.json", "w") as f:
        json.dump(index, f)

    print(f"Saved {len(ticker_to_cik)} companies to company_index.json")

if __name__ == "__main__":
    build_index()
