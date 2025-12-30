"""
Scrape full-text articles for NewsAPI URLs and export cleaned content.

This replaces truncated NewsAPI "content" with full article text using requests + trafilatura.
Output CSV is a cleaned intermediate (not chunked); downstream should map to canonical schema.
"""

import argparse
import random
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests
import trafilatura

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ArticleScraper/1.0)"}

CANONICAL_FIELDS = [
    "id",
    "report_id",
    "ticker",
    "company",
    "date",
    "source",
    "doc_type",
    "item",
    "section_type",
    "section_heading",
    "chunk_index",
    "page_start",
    "page_end",
    "text",
    "source_file",
]


def fetch_html(url: str, timeout: int = 12) -> Optional[str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.text
    except Exception:
        return None


def extract_article_text(html: str, url: str) -> Optional[str]:
    text = trafilatura.extract(
        html,
        url=url,
        include_tables=False,
        include_links=False,
        include_comments=False,
        favor_recall=False,
    )
    if not text:
        return None
    text = text.strip()
    return text if len(text) >= 300 else None


def is_relevant(text: str, ticker: Optional[str]) -> bool:
    if not ticker:
        return True
    lower = text.lower()
    t = ticker.upper()
    count = 0
    if t == "NKE":
        count += lower.count("nike")
        count += lower.count("nke")
    elif t == "LULU":
        count += lower.count("lulu")
        count += lower.count("lululemon")
    elif t == "ATZ" or t == "ATZ.TO":
        count += lower.count("atz")
        count += lower.count("aritzia")
    else:
        count += lower.count(t.lower())
    return count >= 3


def make_row(
    *,
    report_id: str,
    text: str,
    chunk_index: int,
    source: str,
    source_file: str,
    ticker: Optional[str] = None,
    company: Optional[str] = None,
    date: Optional[str] = None,
    doc_type: Optional[str] = None,
    item: Optional[str] = None,
    section_type: Optional[str] = None,
    section_heading: Optional[str] = None,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
) -> Dict:
    return {
        "id": f"{report_id}-{chunk_index}",
        "report_id": report_id,
        "ticker": ticker or "",
        "company": company or "",
        "date": date or "",
        "source": source,
        "doc_type": doc_type or "",
        "item": item or "",
        "section_type": section_type or "",
        "section_heading": section_heading or "",
        "chunk_index": chunk_index,
        "page_start": page_start or "",
        "page_end": page_end or "",
        "text": (text or "").strip(),
        "source_file": source_file,
    }


def scrape_news_file(input_csv: Path, output_csv: Path, min_chars: int = 500) -> None:
    df = pd.read_csv(input_csv)
    rows = []
    for _, row in df.iterrows():
        url = str(row.get("url") or row.get("source_file") or "").strip()
        if not url:
            continue

        html = fetch_html(url)
        if not html:
            continue

        text = extract_article_text(html, url)
        if not text or len(text) < min_chars:
            continue

        ticker_val = row.get("ticker", "")
        ticker = str(ticker_val).upper() if pd.notna(ticker_val) else None
        if ticker == "":
            ticker = None
        if not is_relevant(text, ticker):
            continue

        title = row.get("title") or row.get("section_heading") or ""
        published = row.get("publishedAt") or row.get("date") or ""
        source_name = row.get("source_name") or row.get("source") or ""
        report_id = row.get("report_id") if pd.notna(row.get("report_id")) else None
        if not report_id:
            report_id = hash(url)

        rows.append(
            make_row(
                report_id=str(report_id),
                text=text,
                chunk_index=0,
                source="news",
                source_file=url,
                ticker=ticker or "",
                company="",
                date=published,
                doc_type="news_article",
                section_type="news",
                section_heading=title,
            )
        )

        time.sleep(random.uniform(0.4, 1.3))

    out = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not out.empty:
        out = out.reindex(columns=CANONICAL_FIELDS)
    out.to_csv(output_csv, index=False)
    print(f"[done] {input_csv} -> {len(out)} rows -> {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape NewsAPI URLs into full article text.")
    parser.add_argument("--input-csv", type=Path, default=Path("nlp/processed_data/final/newsapi_articles.csv"))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("nlp/processed_data/final/newsapi_scraped.csv"),
        help="Output CSV with full article_text.",
    )
    parser.add_argument("--min-chars", type=int, default=500, help="Minimum article length to keep.")
    args = parser.parse_args()

    scrape_news_file(args.input_csv, args.output_csv, min_chars=args.min_chars)


if __name__ == "__main__":
    main()
