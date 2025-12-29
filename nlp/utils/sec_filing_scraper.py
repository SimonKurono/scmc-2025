import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
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

# Noise handling
NOISE_PREFIXES = [
    "united states securities and exchange commission",
    "securities and exchange commission",
    "washington, d.c.",
    "form 10-q",
    "form 8-k",
    "table of contents",
    "signatures",
]

NOISE_PHRASES = [
    "accompanying notes are an integral part",
    "accompanying condensed consolidated",
    "unaudited condensed consolidated",
]

ABBREV_RE = re.compile(r"(Mr|Mrs|Ms|Dr|Inc|Ltd|Corp|Co|No|Fig|Eq|St)\.$", re.IGNORECASE)

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


@dataclass
class SecDocument:
    doc_type: str
    text_html: str


def read_file_text(path: Path) -> str:
    return path.read_text(errors="ignore")


def parse_sec_documents(raw: str) -> List[SecDocument]:
    docs: List[SecDocument] = []
    for doc_match in re.finditer(r"<DOCUMENT>(.*?)</DOCUMENT>", raw, flags=re.IGNORECASE | re.DOTALL):
        block = doc_match.group(1)
        type_match = re.search(r"<TYPE>\s*([^\s<]+)", block, flags=re.IGNORECASE)
        doc_type = type_match.group(1).strip() if type_match else ""
        text_match = re.search(r"<TEXT>(.*?)</TEXT>", block, flags=re.IGNORECASE | re.DOTALL)
        text_html = text_match.group(1) if text_match else block
        docs.append(SecDocument(doc_type=doc_type, text_html=text_html))
    if not docs:
        docs.append(SecDocument(doc_type="", text_html=raw))
    return docs


def extract_plain_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style"]):
        tag.decompose()

    for br in soup.find_all("br"):
        br.replace_with(" ")

    for tag in soup.find_all(["p", "div", "tr", "li"]):
        if tag.get_text(strip=True):
            tag.append("\n\n")

    text = soup.get_text()
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
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


def trim_item_tail(snippet: str) -> str:
    tail_markers = ("signatures", "exhibit", "index to exhibits")
    lines = snippet.splitlines()
    trimmed: List[str] = []
    for line in lines:
        lower = line.strip().lower()
        if any(lower.startswith(m) for m in tail_markers):
            break
        trimmed.append(line)
    return "\n".join(trimmed).strip()


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
        snippet = trim_item_tail(text[start:end].strip())
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


def keyword_score(paragraph: str) -> int:
    text = paragraph.lower()
    return sum(1 for kw in KEYWORDS if kw in text)


def map_section_type(item: str) -> str:
    mapping = {
        "1": "business",
        "1A": "risk",
        "2": "md&a",
        "3": "results",
        "7": "md&a",
        "7A": "market_risk",
        "2.02": "results",
        "7.01": "results",
        "8.01": "results",
    }
    return mapping.get(item.upper(), "")


def is_noise_paragraph(p: str) -> bool:
    text = p.strip()
    if not text:
        return True
    lower = text.lower()
    if any(lower.startswith(prefix) for prefix in NOISE_PREFIXES):
        return True
    if any(phrase in lower for phrase in NOISE_PHRASES):
        return True
    if len(text) < 20 and text.isupper():
        return True
    digit_ratio = sum(ch.isdigit() for ch in text) / max(len(text), 1)
    if digit_ratio > 0.6 and keyword_score(text) == 0:
        return True
    return False


def filter_and_rank_paragraphs(paragraphs: List[str], top_n: int = 8, keyword_min: int = 1) -> List[str]:
    ranked = []
    for p in paragraphs:
        if is_noise_paragraph(p):
            continue
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


def split_sentences(paragraph: str) -> List[str]:
    paragraph = paragraph.strip()
    if not paragraph:
        return []
    sentences: List[str] = []
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+", paragraph):
        end_idx = match.start()
        candidate = paragraph[start:end_idx]
        # Avoid splitting after common abbreviations.
        if ABBREV_RE.search(candidate.strip().split()[-1] if candidate.strip().split() else ""):
            continue
        seg = candidate.strip()
        if seg:
            sentences.append(seg)
        start = match.end()
    tail = paragraph[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def chunk_sentences(sentences: List[str], max_tokens: int = 250, overlap_sentences: int = 1) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    token_count = 0

    for s in sentences:
        if not s:
            continue
        s_tokens = len(s.split())
        if current and token_count + s_tokens > max_tokens:
            chunks.append(" ".join(current))
            if overlap_sentences > 0:
                current = current[-overlap_sentences:]
                token_count = sum(len(c.split()) for c in current)
            else:
                current = []
                token_count = 0
        current.append(s)
        token_count += s_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks


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
        "text": text.strip(),
        "source_file": source_file,
    }


def extract_filing_sections_from_file(
    path: Path,
    desired_items: Iterable[str] = DEFAULT_ITEMS,
    min_chars: int = 400,
    max_tokens: int = 250,
    top_n_paragraphs: int = 8,
    keyword_min: int = 1,
    accepted_types: Optional[Iterable[str]] = None,
    debug: bool = False,
) -> List[Dict[str, str]]:
    raw = read_file_text(path)
    filing_type, filing_date = extract_header_metadata(raw, path)

    docs = parse_sec_documents(raw)
    main_doc: Optional[SecDocument] = None
    if accepted_types:
        priorities = [t.upper() for t in accepted_types]
        for typ in priorities:
            for doc in docs:
                dt = doc.doc_type.upper()
                if dt.startswith(typ) or dt == f"{typ}/A":
                    main_doc = doc
                    break
            if main_doc:
                break
    if not main_doc and docs:
        main_doc = docs[0]

    if accepted_types:
        if not main_doc or not main_doc.doc_type or all(t.upper() not in main_doc.doc_type.upper() for t in accepted_types):
            return []

    plain = extract_plain_text_from_html(main_doc.text_html)

    effective_type = main_doc.doc_type or filing_type
    ft_upper = effective_type.upper() if effective_type else ""
    if "8-K" in ft_upper:
        active_items = set(desired_items).intersection(EIGHT_K_ITEMS)
    else:
        active_items = set(desired_items).intersection(TEN_Q_ITEMS)

    ticker = infer_ticker(path)

    rows = []
    report_id = path.stem
    for section in find_item_sections(plain, active_items, min_chars=min_chars):
        paragraphs = [p.strip() for p in section["text"].split("\n\n") if p.strip()]
        filtered = filter_and_rank_paragraphs(paragraphs, top_n=top_n_paragraphs, keyword_min=keyword_min)
        if not filtered:
            continue
        sentences: List[str] = []
        for p in filtered:
            sentences.extend(split_sentences(p))
        chunks = chunk_sentences(sentences, max_tokens=max_tokens)
        if not chunks:
            continue
        for idx, chunk in enumerate(chunks):
            row = make_row(
                report_id=report_id,
                text=chunk,
                chunk_index=idx,
                source="sec",
                source_file=str(path),
                ticker=ticker,
                date=filing_date,
                doc_type=effective_type,
                item=section["item"],
                section_type=map_section_type(section["item"]),
                section_heading=section["item_label"],
            )
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
    debug: bool = False,
) -> pd.DataFrame:
    files = []
    for ext in ("*.txt", "*.xml", "*.sgml", "*.html", "*.htm"):
        files.extend(root_dir.rglob(ext))
    files = sorted(set(files))

    all_rows: List[Dict[str, str]] = []
    debug_samples: List[Dict[str, str]] = []

    for path in files:
        rows = extract_filing_sections_from_file(
            path,
            desired_items,
            min_chars=min_chars,
            max_tokens=max_tokens,
            top_n_paragraphs=top_n_paragraphs,
            keyword_min=keyword_min,
            accepted_types=accepted_types,
            debug=debug,
        )
        if rows:
            print(f"{path}: {len(rows)} sections captured")
        else:
            print(f"{path}: no target sections found")
        if debug and len(debug_samples) < 15:
            debug_samples.extend(rows[:3])
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        for col in CANONICAL_FIELDS:
            if col not in df.columns:
                df[col] = ""
        df = df.reindex(columns=CANONICAL_FIELDS)
    if output_csv and not df.empty:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)

    if not df.empty:
        chunks = df["text"].tolist()
        total_chunks = len(chunks)
        punctuation_end_re = re.compile(r"[.?!][\"')]*\s*$")
        punct_ended = sum(1 for c in chunks if punctuation_end_re.search(c))
        boilerplate_re = re.compile("|".join(re.escape(p) for p in NOISE_PHRASES), re.IGNORECASE)
        boilerplate_hits = sum(1 for c in chunks if boilerplate_re.search(c))
        word_counts = [len(c.split()) for c in chunks]
        print(f"[metrics] chunks={total_chunks}")
        print(f"[metrics] end-with-punct={punct_ended/total_chunks:.2%}")
        print(f"[metrics] boilerplate_hits={boilerplate_hits}")
        print(f"[metrics] words min/median/max={min(word_counts)}/{int(median(word_counts))}/{max(word_counts)}")

        if debug and debug_samples:
            print("[debug] sample chunks:")
            for row in debug_samples[:15]:
                txt = row["text"]
                print(f"  item {row.get('item')} chunk: {txt[:80]!r} ... {txt[-80:]!r}")

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
