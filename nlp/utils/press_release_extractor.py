"""
Extractor for Aritzia-style investor relations press releases (earnings, NCIB, voting results).

Outputs canonical FinBERT ingestion schema:
id, report_id, ticker, company, date, source, doc_type, item, section_type,
section_heading, chunk_index, page_start, page_end, text, source_file
"""

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pdfplumber

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
class PRLine:
    page: int
    text: str


@dataclass
class PRSection:
    section_type: str
    section_heading: str
    page_start: int
    page_end: int
    text: str
    report_id: str
    source_file: str
    ticker: Optional[str]
    date: Optional[str]


ABBREV_RE = re.compile(r"(Mr|Mrs|Ms|Dr|Inc|Ltd|Corp|Co|No|Fig|Eq|St)\.$", re.IGNORECASE)


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


LINE_NOISE_PATTERNS = [
    re.compile(r"^\d+$"),  # page numbers
    re.compile(r"^aritzia inc", re.IGNORECASE),  # footer lead
]


def read_lines(pdf_path: Path) -> List[PRLine]:
    lines: List[PRLine] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            for raw in text.splitlines():
                line = raw.strip()
                if not line:
                    continue
                if any(pat.search(line) for pat in LINE_NOISE_PATTERNS):
                    continue
                lines.append(PRLine(page=page_no, text=line))
    return lines


def is_heading_line(line: str) -> bool:
    if len(line) > 120:
        return False
    if re.match(r"^\d+[\.\)]\s+\S", line):
        return True
    words = [w for w in line.split() if any(c.isalpha() for c in w)]
    if len(words) < 2:
        return False
    caps = sum(1 for w in words if w[0].isupper())
    return caps / len(words) >= 0.7


def classify_section_heading(heading: str) -> str:
    h = heading.lower()
    if "highlight" in h:
        return "highlights"
    if "fiscal" in h or "results" in h or "quarter" in h or "compared to" in h:
        return "results"
    if "outlook" in h:
        return "outlook"
    if "ncib" in h or "normal course issuer bid" in h or "share repurchase" in h:
        return "capital_returns"
    if "voting" in h or "annual general" in h or "agm" in h:
        return "governance"
    if "forward-looking" in h or "non-ifrs" in h or "about aritzia" in h:
        return "boilerplate"
    return "other"


def extract_sections(lines: List[PRLine], report_id: str, source_file: str, ticker: Optional[str], date: Optional[str]) -> List[PRSection]:
    sections: List[PRSection] = []
    current: Optional[Dict] = None
    for item in lines:
        if is_heading_line(item.text):
            if current:
                sections.append(
                    PRSection(
                        section_type=current["section_type"],
                        section_heading=current["heading"],
                        page_start=current["page_start"],
                        page_end=current["page_end"],
                        text=" ".join(current["lines"]).strip(),
                        report_id=report_id,
                        source_file=source_file,
                        ticker=ticker,
                        date=date,
                    )
                )
            current = {
                "section_type": classify_section_heading(item.text),
                "heading": item.text,
                "page_start": item.page,
                "page_end": item.page,
                "lines": [],
            }
            continue
        if current:
            current["lines"].append(item.text)
            current["page_end"] = item.page
    if current:
        sections.append(
            PRSection(
                section_type=current["section_type"],
                section_heading=current["heading"],
                page_start=current["page_start"],
                page_end=current["page_end"],
                text=" ".join(current["lines"]).strip(),
                report_id=report_id,
                source_file=source_file,
                ticker=ticker,
                date=date,
            )
        )
    return sections


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    sentences: List[str] = []
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+", text):
        end_idx = match.start()
        candidate = text[start:end_idx]
        words = candidate.strip().split()
        last = words[-1] if words else ""
        if ABBREV_RE.search(last):
            continue
        seg = candidate.strip()
        if seg:
            sentences.append(seg)
        start = match.end()
    tail = text[start:].strip()
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


def infer_doc_type(path: Path) -> str:
    name = path.stem.lower()
    if "ncib" in name:
        return "ncib_announcement"
    if "voting" in name or "agm" in name:
        return "agm_results"
    if "er" in name or "q" in name:
        return "earnings_release"
    return "press_release"


def write_csv(records: Iterable[Dict], output_path: Path) -> None:
    records = list(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CANONICAL_FIELDS)
        writer.writeheader()
        for rec in records:
            row = {k: rec.get(k, "") for k in CANONICAL_FIELDS}
            writer.writerow(row)


def process_pdf(pdf_path: Path, ticker: str = "ATZ", company: str = "Aritzia Inc.", date: Optional[str] = "") -> List[Dict]:
    report_id = pdf_path.stem
    lines = read_lines(pdf_path)
    sections = extract_sections(lines, report_id=report_id, source_file=str(pdf_path), ticker=ticker, date=date)
    rows: List[Dict] = []
    doc_type = infer_doc_type(pdf_path)
    for section in sections:
        sentences = split_sentences(section.text)
        chunks = chunk_sentences(sentences, max_tokens=250)
        for idx, chunk in enumerate(chunks):
            rows.append(
                make_row(
                    report_id=section.report_id,
                    text=chunk,
                    chunk_index=idx,
                    source="press_release",
                    source_file=section.source_file,
                    ticker=section.ticker,
                    company=company,
                    date=section.date,
                    doc_type=doc_type,
                    section_type=section.section_type,
                    section_heading=section.section_heading,
                    page_start=section.page_start,
                    page_end=section.page_end,
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract press release PDF chunks into canonical schema.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("nlp/raw_data/sec_can/news_releases"),
        help="Directory containing press release PDFs.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("nlp/processed_data/press_releases_chunks.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    all_rows: List[Dict] = []
    for pdf_path in sorted(args.input_dir.rglob("*.pdf")):
        recs = process_pdf(pdf_path)
        all_rows.extend(recs)
        print(f"[done] {pdf_path.name}: {len(recs)} chunks")

    write_csv(all_rows, args.output_csv)
    print(f"[done] wrote {len(all_rows)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
