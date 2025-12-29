"""
Local PDF section extractor tailored to Bloomberg-style research notes.

Extracts thesis/growth/risk/valuation/earnings sections, chunks text with
FinBERT tokenizer, and writes canonical-schema outputs (sections + chunks).
"""

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pdfplumber
from transformers import AutoTokenizer

# Keywords to identify sections
THESIS_KEYS = ["thesis:", "equity outlook", "focus idea", "bi focus"]
RISK_KEYS = ["risk", "headwind", "downside", "tariff", "drag", "inventory", "esg"]
VALUATION_KEYS = ["valuation", "multiple", "upside", "downside", "target price", "discounted"]
EARNINGS_KEYS = ["earnings outlook", "2q", "3q", "4q", "preview", "beat", "miss"]
GROWTH_HINTS = ["growth", "expansion", "digital", "sales", "white space", "store", "footprint", "awareness"]

BLOOMBERG_NOISE_PATTERNS = [
    re.compile(r"^this document is being provided for the exclusive use", re.IGNORECASE),
    re.compile(r"^bloomberg[®\s]", re.IGNORECASE),
    re.compile(r"^source:\s*bloomberg intelligence", re.IGNORECASE),
    re.compile(r"^bloomberg interactive calculator", re.IGNORECASE),
]

MAX_TOKENS = 450
OVERLAP_TOKENS = 60

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

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
class Line:
    page: int
    text: str


@dataclass
class Section:
    section_type: str  # thesis/growth/risk/valuation/earnings
    heading: str
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
                lower = line.lower()
                if re.fullmatch(r"page \d+", lower):
                    continue
                if re.fullmatch(r"\d+", line):
                    continue
                if any(pat.search(lower) for pat in BLOOMBERG_NOISE_PATTERNS):
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
    words = [w for w in line.split() if any(c.isalpha() for c in w)]
    if len(words) < 2:
        return False
    caps = sum(1 for w in words if w[0].isupper())
    return caps / len(words) >= 0.7


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
    if re.match(r"^\d+[\.\)]\s+\S", line):
        return "growth"
    return None


def chunk_text(text: str, max_tokens: int = MAX_TOKENS, overlap_tokens: int = OVERLAP_TOKENS) -> List[Tuple[str, int]]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be < max_tokens.")
    chunks: List[Tuple[str, int]] = []
    start = 0
    while start < len(token_ids):
        end = min(len(token_ids), start + max_tokens)
        chunk_ids = token_ids[start:end]
        token_len = len(chunk_ids)
        if token_len > 512:
            raise ValueError(f"Chunk exceeded 512 tokens ({token_len})")
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append((chunk_text, token_len))
        if end == len(token_ids):
            break
        start = end - overlap_tokens
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
    sections.append(
        Section(
            section_type=current["section_type"],
            heading=current["heading"],
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
    first_heading_seen = False

    for item in lines:
        raw = item.text
        heading = classify_heading(raw) if is_heading_line(raw) or raw.lower().startswith("thesis:") else None
        if heading:
            if not first_heading_seen:
                heading = "thesis"
                first_heading_seen = True
            finalize_section(current, sections, report_id, source, ticker, file_path)
            current = {
                "section_type": heading,
                "heading": raw,
                "page_start": item.page,
                "page_end": item.page,
                "lines": [],
            }
            continue

        if current:
            current["lines"].append(raw)
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
    stem = Path(path).stem
    first_token = re.split(r"[^A-Za-z0-9]+", stem)[0]
    if first_token and first_token.isalpha() and len(first_token) <= 5:
        return first_token.upper()
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CANONICAL_FIELDS)
        writer.writeheader()
        for rec in records:
            row = {k: rec.get(k, "") for k in CANONICAL_FIELDS}
            writer.writerow(row)
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
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="Max FinBERT tokens per chunk (safety under 512).")
    parser.add_argument("--overlap-tokens", type=int, default=OVERLAP_TOKENS, help="Overlap in tokens between adjacent chunks.")
    parser.add_argument("--ticker", type=str, default=None, help="Optional explicit ticker override.")
    args = parser.parse_args()

    section_records: List[Dict] = []
    chunk_records: List[Dict] = []

    for pdf_path in iter_pdfs(args.input_dir):
        report_id = pdf_path.stem
        ticker = args.ticker or guess_ticker_from_path(pdf_path)
        try:
            lines = read_pdf_lines(pdf_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Failed to read {pdf_path}: {exc}")
            continue

        sections = extract_sections(lines, report_id, source="bloomberg", ticker=ticker, file_path=str(pdf_path))
        if not sections:
            print(f"[info] No sections detected in {pdf_path}")
            continue

        chunk_counter = 0
        for s_idx, s in enumerate(sections):
            section_row = make_row(
                report_id=s.report_id,
                text=s.text,
                chunk_index=s_idx,
                source="bloomberg",
                source_file=s.file_path,
                ticker=s.ticker,
                doc_type="research_note",
                section_type=s.section_type,
                section_heading=s.heading,
                page_start=s.page_start,
                page_end=s.page_end,
            )
            section_records.append(section_row)

            for chunk_text_val, tok_len in chunk_text(s.text, max_tokens=args.max_tokens, overlap_tokens=args.overlap_tokens):
                chunk_row = make_row(
                    report_id=s.report_id,
                    text=chunk_text_val,
                    chunk_index=chunk_counter,
                    source="bloomberg",
                    source_file=s.file_path,
                    ticker=s.ticker,
                    doc_type="research_note",
                    section_type=s.section_type,
                    section_heading=s.heading,
                    page_start=s.page_start,
                    page_end=s.page_end,
                )
                chunk_row["token_estimate"] = tok_len  # retain for jsonl readability
                chunk_records.append(chunk_row)
                chunk_counter += 1

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
