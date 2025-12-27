"""
Local PDF section extractor tailored to Bloomberg-style research notes.

It pulls thesis/growth/risk/valuation/earnings blocks using simple
heading heuristics, then optionally chunks text for BERT-friendly input.
"""

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pdfplumber


# Keywords to identify sections
THESIS_KEYS = ["thesis:", "equity outlook", "focus idea", "bi focus"]
RISK_KEYS = [
    "risk",
    "headwind",
    "downside",
    "tariff",
    "drag",
    "inventory",
    "esg",
]
VALUATION_KEYS = [
    "valuation",
    "multiple",
    "upside",
    "downside",
    "target price",
    "discounted",
]
EARNINGS_KEYS = ["earnings outlook", "2q", "3q", "4q", "preview", "beat", "miss"]
GROWTH_HINTS = [
    "growth",
    "expansion",
    "digital",
    "sales",
    "white space",
    "store",
    "footprint",
    "awareness",
]

MAX_SECTION_CHARS = 1500


@dataclass
class Line:
    page: int
    text: str


@dataclass
class Section:
    section: str
    page_start: int
    page_end: int
    text: str
    report_id: str
    source: str
    ticker: Optional[str]
    file_path: str


def read_pdf_lines(pdf_path: Path) -> List[Line]:
    lines: List[Line] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                # Skip obvious footers / page numbers
                if re.fullmatch(r"page \d+", line.lower()):
                    continue
                if re.fullmatch(r"\d+", line):
                    continue
                lines.append(Line(page=page_no, text=line))
    return lines


def is_heading_line(line: str) -> bool:
    """Heuristic to decide if a line looks like a section heading."""
    if len(line) > 140:
        return False
    if re.match(r"^\d+[\.\)]\s+\S", line):
        return True
    if line.isupper():
        return True
    words = line.split()
    if len(words) <= 8 and all(w[0].isupper() for w in words if w[0].isalpha()):
        return True
    return False


def classify_heading(line: str) -> Optional[str]:
    lower = line.lower()
    if any(key in lower for key in THESIS_KEYS):
        return "thesis"
    if any(key in lower for key in RISK_KEYS):
        return "risk"
    if any(key in lower for key in VALUATION_KEYS):
        return "valuation"
    if any(key in lower for key in EARNINGS_KEYS):
        return "earnings"
    if any(key in lower for key in GROWTH_HINTS):
        return "growth"
    # Numbered headings are typically main bullets -> treat as growth/opportunity blocks
    if re.match(r"^\d+[\.\)]\s+\S", line):
        return "growth"
    return None


def chunk_text(text: str, max_words: int = 450, overlap: int = 60) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_words)
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks


def finalize_section(
    current: Optional[Dict],
    sections: List[Section],
    report_id: str,
    source: str,
    ticker: Optional[str],
    file_path: str,
) -> None:
    if not current:
        return
    text = " ".join(current["lines"]).strip()
    if not text:
        return
    if len(text) > MAX_SECTION_CHARS:
        text = text[:MAX_SECTION_CHARS].rsplit(" ", 1)[0]
    sections.append(
        Section(
            section=current["section"],
            page_start=current["page_start"],
            page_end=current["page_end"],
            text=text,
            report_id=report_id,
            source=source,
            ticker=ticker,
            file_path=file_path,
        )
    )


def extract_sections(lines: List[Line], report_id: str, source: str, ticker: Optional[str], file_path: str) -> List[Section]:
    sections: List[Section] = []
    current: Optional[Dict] = None

    for item in lines:
        heading = classify_heading(item.text) if is_heading_line(item.text) or item.text.lower().startswith("thesis:") else None
        if heading:
            finalize_section(current, sections, report_id, source, ticker, file_path)
            current = {
                "section": heading,
                "page_start": item.page,
                "page_end": item.page,
                "lines": [],
            }
            continue

        if current:
            current["lines"].append(item.text)
            current["page_end"] = item.page

    finalize_section(current, sections, report_id, source, ticker, file_path)
    return sections


def guess_ticker_from_path(path: Path) -> Optional[str]:
    lower = str(path).lower()
    if "atz" in lower:
        return "ATZ"
    if "lulu" in lower:
        return "LULU"
    if "nke" in lower:
        return "NKE"
    return None


def iter_pdfs(input_dir: Path) -> Iterable[Path]:
    return (p for p in input_dir.rglob("*.pdf") if p.is_file())


def write_jsonl(records: Iterable[Dict], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_csv(records: Iterable[Dict], output_path: Path) -> int:
    records = list(records)
    if not records:
        return 0
    # Preserve key order from the first record, then append any new keys encountered.
    fieldnames: List[str] = list(records[0].keys())
    seen = set(fieldnames)
    for rec in records:
        for k in rec.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract structured sections from research PDFs for BERT ingestion.")
    parser.add_argument("--input-dir", type=Path, default=Path("nlp/raw_data/bloomberg"), help="Directory containing PDFs (searched recursively).")
    parser.add_argument("--output-sections", type=Path, default=Path("nlp/processed_data/pdf_sections.jsonl"), help="JSONL of extracted sections.")
    parser.add_argument("--output-sections-csv", type=Path, default=Path("nlp/processed_data/pdf_sections.csv"), help="CSV of extracted sections.")
    parser.add_argument(
        "--output-chunks",
        type=Path,
        default=Path("nlp/processed_data/pdf_sections_chunks.jsonl"),
        help="Optional JSONL of BERT-sized text chunks with section metadata.",
    )
    parser.add_argument(
        "--output-chunks-csv",
        type=Path,
        default=Path("nlp/processed_data/pdf_sections_chunks.csv"),
        help="CSV of BERT-sized text chunks with section metadata.",
    )
    parser.add_argument("--max-words", type=int, default=450, help="Max words per chunk.")
    parser.add_argument("--overlap", type=int, default=60, help="Overlap words between chunks.")
    args = parser.parse_args()

    section_records: List[Dict] = []
    chunk_records: List[Dict] = []

    for pdf_path in iter_pdfs(args.input_dir):
        report_id = pdf_path.stem
        ticker = guess_ticker_from_path(pdf_path)
        try:
            lines = read_pdf_lines(pdf_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Failed to read {pdf_path}: {exc}")
            continue

        sections = extract_sections(lines, report_id, source="bloomberg", ticker=ticker, file_path=str(pdf_path))
        if not sections:
            print(f"[info] No sections detected in {pdf_path}")
            continue

        for s in sections:
            section_records.append(
                {
                    "report_id": s.report_id,
                    "ticker": s.ticker,
                    "source": s.source,
                    "section": s.section,
                    "page_start": s.page_start,
                    "page_end": s.page_end,
                    "text": s.text,
                    "file_path": s.file_path,
                }
            )

            for idx, chunk in enumerate(chunk_text(s.text, max_words=args.max_words, overlap=args.overlap)):
                chunk_records.append(
                    {
                        "report_id": s.report_id,
                        "ticker": s.ticker,
                        "source": s.source,
                        "section": s.section,
                        "chunk_id": idx,
                        "page_start": s.page_start,
                        "page_end": s.page_end,
                        "text": chunk,
                        "file_path": s.file_path,
                    }
                )

    if section_records:
        n_sections_jsonl = write_jsonl(section_records, args.output_sections)
        n_sections_csv = write_csv(section_records, args.output_sections_csv)
        print(f"[done] Wrote {n_sections_jsonl} sections -> {args.output_sections}")
        print(f"[done] Wrote {n_sections_csv} sections -> {args.output_sections_csv}")
    else:
        print("[done] No sections extracted; nothing written.")

    if chunk_records:
        n_chunks_jsonl = write_jsonl(chunk_records, args.output_chunks)
        n_chunks_csv = write_csv(chunk_records, args.output_chunks_csv)
        print(f"[done] Wrote {n_chunks_jsonl} chunks -> {args.output_chunks}")
        print(f"[done] Wrote {n_chunks_csv} chunks -> {args.output_chunks_csv}")


if __name__ == "__main__":
    main()
