import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup


# Default target items: MD&A plus nearby risk/market-risk sections for 10-K/10-Q,
# and common 8-K items.
DEFAULT_ITEMS = {
    "1",
    "1A",
    "2",
    "3",
    "7",
    "7A",
    "2.02",
    "7.01",
    "8.01",
}

ITEM_LABELS = {
    "1": "Business",
    "1A": "Risk Factors",
    "2": "MD&A",
    "3": "Legal Proceedings / Market Risk (10-Q)",
    "7": "MD&A (10-K)",
    "7A": "Market Risk (10-K)",
    "2.02": "Results of Operations and Financial Condition (8-K)",
    "7.01": "Regulation FD Disclosure (8-K)",
    "8.01": "Other Events (8-K)",
}

TEN_Q_ITEMS = {"1", "1A", "2", "3", "7", "7A"}
EIGHT_K_ITEMS = {"2.02", "7.01", "8.01"}

KEYWORDS = {
    # Demand / revenue
    "revenue",
    "sales",
    "demand",
    "traffic",
    "comps",
    "same-store",
    "orders",
    "bookings",
    # Profitability
    "margin",
    "gross",
    "operating",
    "ebit",
    "ebitda",
    "profit",
    "loss",
    "leverage",
    # Costs / inventory
    "sg&a",
    "opex",
    "costs",
    "inventory",
    "markdown",
    "freight",
    "logistics",
    # Guidance / outlook
    "guidance",
    "outlook",
    "forecast",
    "expect",
    "anticipate",
    # Cash / liquidity
    "cash",
    "liquidity",
    "debt",
    "credit",
    "covenant",
    "capex",
    "capital",
    "free cash flow",
    "operating cash",
    # FX / macro
    "fx",
    "currency",
    "inflation",
    "macro",
    # Risks / impairments
    "risk",
    "uncertain",
    "impairment",
    "restructuring",
    "write-down",
}


def read_file_text(path: Path) -> str:
    return path.read_text(errors="ignore")


def extract_plain_text(raw: str) -> str:
    """Strip tags/SGML and normalize whitespace."""
    text_blocks = re.findall(r"<TEXT>(.*?)</TEXT>", raw, flags=re.IGNORECASE | re.DOTALL)
    candidate = "\n\n".join(text_blocks) if text_blocks else raw
    soup = BeautifulSoup(candidate, "html.parser")
    text = soup.get_text("\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def infer_ticker(path: Path) -> Optional[str]:
    for part in reversed(path.parts):
        token = re.sub(r"[^A-Za-z]", "", part)
        if token and token.isalpha() and 1 <= len(token) <= 5:
            return token.upper()
    return None


def extract_header_metadata(raw: str, path: Path) -> Tuple[Optional[str], Optional[str]]:
    filing_type = None
    filing_date = None

    type_match = re.search(r"CONFORMED SUBMISSION TYPE:\s*([0-9A-Z\-]+)", raw, re.IGNORECASE)
    if not type_match:
        type_match = re.search(r"SUBMISSION TYPE:\s*([0-9A-Z\-]+)", raw, re.IGNORECASE)
    if type_match:
        filing_type = type_match.group(1)

    date_match = re.search(r"FILED AS OF DATE:\s*(\d{8})", raw)
    if not date_match:
        date_match = re.search(r"ACCEPTANCE-DATETIME>\s*(\d{14})", raw)
    if date_match:
        digits = date_match.group(1)
        if len(digits) == 8:
            filing_date = f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
        elif len(digits) == 14:
            filing_date = f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return filing_type, filing_date


def find_item_sections(
    text: str,
    desired_items: Iterable[str],
    min_chars: int = 400,
) -> List[Dict[str, str]]:
    """Slice text into item sections and keep target items."""
    desired = set(desired_items)
    pattern = re.compile(r"(?:^|\n)\s*ITEM\s+([0-9]{1,2}[A-Z]?(\.[0-9]{1,2})?)", re.IGNORECASE | re.MULTILINE)
    matches = list(pattern.finditer(text))
    sections: List[Dict[str, str]] = []

    for idx, match in enumerate(matches):
        item_id = match.group(1).upper()
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        snippet = text[start:end].strip()
        if item_id not in desired:
            continue
        if len(snippet) < min_chars:
            continue
        sections.append(
            {
                "item": item_id,
                "item_label": ITEM_LABELS.get(item_id, ""),
                "text": snippet,
            }
        )

    # If multiple captures per item, keep the longest to avoid table-of-contents noise.
    best_by_item: Dict[str, Dict[str, str]] = {}
    for section in sections:
        existing = best_by_item.get(section["item"])
        if not existing or len(section["text"]) > len(existing["text"]):
            best_by_item[section["item"]] = section
    return list(best_by_item.values())


def chunk_text(text: str, max_tokens: int = 250, overlap: int = 20) -> List[str]:
    """
    Roughly cap length for FinBERT/BERT.
    Tokens ~ words; we chunk by words with small overlap to keep continuity.
    Default 250 words ≈ safely under 512 subword tokens.
    """
    words = text.split()
    if len(words) <= max_tokens:
        return [" ".join(words)]

    step = max(max_tokens - overlap, 1)
    chunks = []
    for i in range(0, len(words), step):
        chunk_words = words[i : i + max_tokens]
        chunks.append(" ".join(chunk_words))
        if i + max_tokens >= len(words):
            break
    return chunks


def keyword_score(paragraph: str) -> int:
    text = paragraph.lower()
    return sum(1 for kw in KEYWORDS if kw in text)


def filter_and_rank_paragraphs(paragraphs: List[str], top_n: int = 8, keyword_min: int = 1) -> List[str]:
    ranked = []
    for p in paragraphs:
        words = p.split()
        if len(words) < 15:
            continue
        digit_ratio = sum(ch.isdigit() for ch in p) / max(len(p), 1)
        score = keyword_score(p)
        if digit_ratio > 0.35 and score == 0:
            continue
        ranked.append((score, len(p), p))
    if not ranked:
        return []
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    kept = []
    for score, _, p in ranked:
        if len(kept) >= top_n:
            break
        if score >= keyword_min or len(kept) == 0:
            kept.append(p.strip())
    return kept


def extract_filing_sections_from_file(
    path: Path,
    desired_items: Iterable[str] = DEFAULT_ITEMS,
    min_chars: int = 400,
    max_tokens: int = 250,
    top_n_paragraphs: int = 8,
    keyword_min: int = 1,
    accepted_types: Optional[Iterable[str]] = None,
) -> List[Dict[str, str]]:
    raw = read_file_text(path)
    plain = extract_plain_text(raw)
    filing_type, filing_date = extract_header_metadata(raw, path)

    if accepted_types:
        if not filing_type or all(t.upper() not in filing_type.upper() for t in accepted_types):
            return []

    # Choose item set per filing type
    ft_upper = filing_type.upper() if filing_type else ""
    if "8-K" in ft_upper:
        active_items = set(desired_items).intersection(EIGHT_K_ITEMS)
    else:
        active_items = set(desired_items).intersection(TEN_Q_ITEMS)

    ticker = infer_ticker(path)

    rows = []
    for section in find_item_sections(plain, active_items, min_chars=min_chars):
        paragraphs = [p.strip() for p in section["text"].split("\n\n") if p.strip()]
        filtered = filter_and_rank_paragraphs(paragraphs, top_n=top_n_paragraphs, keyword_min=keyword_min)
        if not filtered:
            continue
        condensed = "\n\n".join(filtered)
        chunks = chunk_text(condensed, max_tokens=max_tokens)
        if not chunks:
            continue
        for idx, chunk in enumerate(chunks):
            row = {
                "ticker": ticker,
                "filing_type": filing_type,
                "filing_date": filing_date,
                "item": section["item"],
                "item_label": section["item_label"],
                "chunk_index": idx,
                "text": chunk,
                "source_file": str(path),
            }
            rows.append(row)
    return rows


def extract_sec_filings(
    root_dir: Path,
    output_csv: Optional[Path] = None,
    desired_items: Iterable[str] = DEFAULT_ITEMS,
    min_chars: int = 400,
    max_tokens: int = 250,
    top_n_paragraphs: int = 8,
    keyword_min: int = 1,
    accepted_types: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    files = []
    for ext in ("*.txt", "*.xml", "*.sgml", "*.html", "*.htm"):
        files.extend(root_dir.rglob(ext))
    files = sorted(set(files))

    all_rows: List[Dict[str, str]] = []
    for path in files:
        rows = extract_filing_sections_from_file(
            path,
            desired_items,
            min_chars=min_chars,
            max_tokens=max_tokens,
            top_n_paragraphs=top_n_paragraphs,
            keyword_min=keyword_min,
            accepted_types=accepted_types,
        )
        if rows:
            print(f"{path}: {len(rows)} sections captured")
        else:
            print(f"{path}: no target sections found")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if output_csv and not df.empty:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    return df


if __name__ == "__main__":
    # 10-Q extraction
    root_10q = Path("nlp/raw_data/sec_filings_10q")
    output_10q = Path("nlp/processed_data/sec_filings_extracted_10q.csv")
    df_10q = extract_sec_filings(
        root_10q,
        output_csv=output_10q,
        max_tokens=250,
        top_n_paragraphs=8,
        keyword_min=1,
        accepted_types=["10-Q"],
        desired_items=TEN_Q_ITEMS,
    )
    print(f"10-Q sections extracted: {len(df_10q)}")

    # 8-K extraction
    root_8k = Path("nlp/raw_data/sec_filings_8k")
    output_8k = Path("nlp/processed_data/sec_filings_extracted_8k.csv")
    df_8k = extract_sec_filings(
        root_8k,
        output_csv=output_8k,
        max_tokens=250,
        top_n_paragraphs=8,
        keyword_min=1,
        accepted_types=["8-K"],
        desired_items=EIGHT_K_ITEMS,
    )
    print(f"8-K sections extracted: {len(df_8k)}")
