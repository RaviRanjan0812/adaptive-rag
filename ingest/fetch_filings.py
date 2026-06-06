"""Download SEC EDGAR filings.

The submissions JSON at data.sec.gov already includes the `primaryDocument`
filename for every filing — no need to fetch the index page separately.

Direct document URL pattern:
  https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{primaryDocument}

SEC requires a descriptive User-Agent: "Name email@example.com"
Set SEC_USER_AGENT env var.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import requests

TICKERS      = ["AAPL", "JPM", "NVDA"]
FORMS        = ["10-K", "10-Q"]
MAX_PER_FORM = 1

RAW_DIR    = Path("data/raw")
USER_AGENT = os.getenv("SEC_USER_AGENT", "AdaptiveRAG research@example.com")
HEADERS    = {
    "User-Agent":      USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}
SLEEP = 0.15


def _get(url: str) -> requests.Response:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(SLEEP)
    return resp


def cik_for_ticker(ticker: str) -> str:
    data = _get("https://www.sec.gov/files/company_tickers.json").json()
    for v in data.values():
        if v["ticker"].upper() == ticker.upper():
            return str(v["cik_str"]).zfill(10)
    raise ValueError(f"CIK not found for {ticker}")


def filings_for_cik(cik: str, form: str, max_n: int) -> list[dict]:
    """Return recent filings. primaryDocument comes directly from the submissions JSON."""
    url  = f"https://data.sec.gov/submissions/CIK{cik}.json"
    subs = _get(url).json()
    recent = subs.get("filings", {}).get("recent", {})

    forms            = recent.get("form", [])
    accessions       = recent.get("accessionNumber", [])
    dates            = recent.get("reportDate", [])
    primary_docs     = recent.get("primaryDocument", [])

    results = []
    for f, acc, d, pdoc in zip(forms, accessions, dates, primary_docs):
        if f == form:
            results.append({
                "accession":       acc.replace("-", ""),  # no-dash 18-char form for URL
                "accession_dash":  acc,
                "period":          d,
                "primary_doc":     pdoc,                  # filename e.g. aapl-20240928.htm
            })
            if len(results) >= max_n:
                break
    return results


def fetch_filing_text(cik: str, accession: str, primary_doc: str) -> Optional[str]:
    """Fetch the primary document directly — no index lookup needed."""
    cik_int = int(cik)
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{primary_doc}"
    print(f"\n    URL: {url}", end=" ", flush=True)
    try:
        text = _get(url).text
        cleaned = _clean(text)
        if len(cleaned) < 500:
            print(f"[too short: {len(cleaned)} chars]", end=" ")
            return None
        return cleaned
    except Exception as exc:
        print(f"[FAILED: {exc}]", end=" ")
        return None


def _clean(raw: str) -> str:
    text = re.sub(r"<ix:[^>]+>",  " ", raw, flags=re.I)
    text = re.sub(r"</ix:[^>]+>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>",     " ", text)
    text = re.sub(r"&amp;",  "&", text)
    text = re.sub(r"&lt;",   "<", text)
    text = re.sub(r"&gt;",   ">", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    print(f"User-Agent: {USER_AGENT}\n")

    for ticker in TICKERS:
        print(f"=== {ticker} ===")
        try:
            cik = cik_for_ticker(ticker)
            print(f"  CIK: {cik}")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        ticker_dir = RAW_DIR / ticker
        ticker_dir.mkdir(exist_ok=True)

        for form in FORMS:
            try:
                filings = filings_for_cik(cik, form, MAX_PER_FORM)
            except Exception as exc:
                print(f"  ERROR listing {form}: {exc}")
                continue

            if not filings:
                print(f"  No {form} found")
                continue

            for filing in filings:
                period      = filing["period"] or "unknown"
                safe_period = period.replace("-", "")
                doc_id      = f"{ticker}-{form.replace('-','')}-{safe_period}"
                out_path    = ticker_dir / f"{form.replace('-','')}.txt"

                if out_path.exists():
                    print(f"  {doc_id} cached")
                    manifest.append({"doc_id": doc_id, "ticker": ticker, "form": form,
                                     "period": period, "path": str(out_path)})
                    continue

                print(f"  {doc_id}  primary_doc={filing['primary_doc']}", end="")
                text = fetch_filing_text(cik, filing["accession"], filing["primary_doc"])
                if text:
                    out_path.write_text(text, encoding="utf-8")
                    manifest.append({"doc_id": doc_id, "ticker": ticker, "form": form,
                                     "period": period, "path": str(out_path)})
                    print(f"\n  saved {out_path} ({len(text):,} chars)")
                else:
                    print("\n  FAILED — skipped")

    manifest_path = RAW_DIR / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for rec in manifest:
            f.write(json.dumps(rec) + "\n")
    print(f"\nManifest: {manifest_path} ({len(manifest)} docs)")


if __name__ == "__main__":
    main()
