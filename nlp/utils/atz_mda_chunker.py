"""
Chunk Aritzia MD&A PDFs (Canadian filings) into FinBERT-ready text.

Goals:
1) All chunks < 512 tokens (we keep a safety margin using FinBERT tokens).
2) No cut sentences; chunk boundaries fall on sentence edges.
3) Filter out obvious noise (headers, page numbers, web links, boilerplate/copyright).

Output: CSV with chunk metadata and text.
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pdfplumber
from transformers import AutoTokenizer

# Safety margin so tokenized chunks stay under BERT's 512 token limit.
MAX_TOKENS = 450
# If a single sentence is very long, break it into smaller pieces.
MAX_SENTENCE_TOKENS = 220

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

# Line-level noise patterns (headers/footers).
LINE_NOISE_PATTERNS = [
    re.compile(r"^\d+$"),  # bare page numbers
    re.compile(r"^page \d+", re.IGNORECASE),
    re.compile(r"^aritzi?a? inc\.?$", re.IGNORECASE),
    re.compile(r"^management.?s discussion and analysis$", re.IGNORECASE),
    re.compile(r"^table of contents$", re.IGNORECASE),
    re.compile(r"^unaudited$", re.IGNORECASE),
]

# Sentence-level noise patterns (boilerplate / non-substantive).
SENTENCE_NOISE_PATTERNS = [
    re.compile(r"forward[- ]looking", re.IGNORECASE),
    re.compile(r"sedar", re.IGNORECASE),
    re.compile(r"www\.", re.IGNORECASE),
    re.compile(r"copyright", re.IGNORECASE),
    re.compile(r"trademark", re.IGNORECASE),
    re.compile(r"caution", re.IGNORECASE),
]

ABBREV_RE = re.compile(r"(U\.S\.A|U\.S|U\.K\.?|Inc\.?|Ltd\.?|Corp\.?|Co\.?|No\.)$", re.IGNORECASE)


def token_count(text: str) -> int:
    # Use FinBERT tokenizer to mirror model tokenization; exclude CLS/SEP here.
    return len(tokenizer.encode(text, add_special_tokens=False))


def clean_line(line: str) -> Optional[str]:
    line = line.strip()
    if not line:
        return None
    for pat in LINE_NOISE_PATTERNS:
        if pat.search(line):
            return None
    return line


def split_sentences(text: str) -> List[str]:
    sentences: List[str] = []
    start = 0
    # Split on sentence-ending punctuation + whitespace, but skip common abbreviations.
    for match in re.finditer(r"(?<=[.!?])\s+", text):
        end_idx = match.start()
        candidate = text[start:end_idx]
        if ABBREV_RE.search(candidate):
            continue
        seg = candidate.strip()
        if seg:
            sentences.append(seg)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def is_noise_sentence(sentence: str) -> bool:
    stripped = sentence.strip()
    if not stripped:
        return True
    # Very short and not punctuated -> likely junk.
    if len(stripped) < 10 and not re.search(r"[.!?]$", stripped):
        return True
    # Short all-caps anchors: keep.
    if stripped.isupper() and len(stripped) < 10:
        return False
    # Long all-caps only drop if boilerplate.
    if stripped.isupper() and len(stripped) < 120:
        for pat in SENTENCE_NOISE_PATTERNS:
            if pat.search(stripped):
                return True
        return False
    # Drop if matching explicit boilerplate patterns.
    for pat in SENTENCE_NOISE_PATTERNS:
        if pat.search(stripped):
            return True
    return False


def break_long_sentence(sentence: str, max_tokens: int) -> List[str]:
    token_ids = tokenizer.encode(sentence, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return [sentence]
    chunks: List[str] = []
    start = 0
    while start < len(token_ids):
        end = min(len(token_ids), start + max_tokens)
        chunk_ids = token_ids[start:end]
        chunks.append(tokenizer.decode(chunk_ids))
        start = end
    return chunks


def read_sentences_with_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    """Read a PDF into (page, sentence) tuples after line-level cleanup."""
    sentences: List[Tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            cleaned_lines = []
            for raw in text.splitlines():
                line = clean_line(raw)
                if line:
                    cleaned_lines.append(line)
            if not cleaned_lines:
                continue
            page_text = " ".join(cleaned_lines)
            for sent in split_sentences(page_text):
                if is_noise_sentence(sent):
                    continue
                # If a sentence is extremely long, break it safely.
                for sub in break_long_sentence(sent, MAX_SENTENCE_TOKENS):
                    sentences.append((page_no, sub))
    return sentences


def chunk_sentences(
    sentences: List[Tuple[int, str]],
    max_tokens: int = MAX_TOKENS,
) -> List[Dict]:
    chunks: List[Dict] = []
    current: List[str] = []
    token_sum = 0
    page_start: Optional[int] = None
    page_end: Optional[int] = None

    def flush():
        nonlocal current, token_sum, page_start, page_end
        if not current:
            return
        text = " ".join(current).strip()
        actual_tokens = token_count(text)
        if actual_tokens > 512:
            raise ValueError(f"Chunk exceeded 512 tokens ({actual_tokens})")
        chunks.append(
            {
                "text": text,
                "token_estimate": actual_tokens,
                "page_start": page_start,
                "page_end": page_end,
            }
        )
        current.clear()
        token_sum = 0
        page_start = None
        page_end = None

    for page, sent in sentences:
        sent_tokens = token_count(sent)
        if sent_tokens > max_tokens:
            # Defensive split: should not happen due to break_long_sentence.
            for piece in break_long_sentence(sent, max_tokens):
                piece_tokens = token_count(piece)
                if token_sum + piece_tokens > max_tokens and current:
                    flush()
                current.append(piece)
                token_sum += piece_tokens
                page_start = page if page_start is None else page_start
                page_end = page
                flush()
            continue

        if token_sum + sent_tokens > max_tokens and current:
            flush()

        current.append(sent)
        token_sum += sent_tokens
        page_start = page if page_start is None else page_start
        page_end = page

    flush()
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


def process_pdf(pdf_path: Path, ticker: str = "ATZ", max_tokens: int = MAX_TOKENS) -> List[Dict]:
    sentences = read_sentences_with_pages(pdf_path)
    chunk_dicts = chunk_sentences(sentences, max_tokens=max_tokens)
    report_id = pdf_path.stem
    rows: List[Dict] = []
    for idx, chunk in enumerate(chunk_dicts):
        row = make_row(
            report_id=report_id,
            text=chunk["text"],
            chunk_index=idx,
            source="mda_canada",
            source_file=str(pdf_path),
            ticker=ticker,
            doc_type="md&a",
            section_type="md&a",
            page_start=chunk.get("page_start"),
            page_end=chunk.get("page_end"),
        )
        row["token_estimate"] = chunk.get("token_estimate", "")
        rows.append(row)
    return rows


def write_csv(records: Iterable[Dict], output_path: Path) -> None:
    records = list(records)
    if not records:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CANONICAL_FIELDS)
        writer.writeheader()
        for rec in records:
            row = {k: rec.get(k, "") for k in CANONICAL_FIELDS}
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk ATZ MD&A PDFs into FinBERT-friendly CSV.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("nlp/raw_data/sec_can/atz_q_mda"),
        help="Directory containing ATZ MDA PDFs.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("nlp/processed_data/atz_mda_chunks.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help="Max tokens per chunk (FinBERT tokens, safety under 512).",
    )
    args = parser.parse_args()

    all_records: List[Dict] = []
    for pdf_path in sorted(args.input_dir.glob("*.pdf")):
        records = process_pdf(pdf_path, ticker="ATZ", max_tokens=args.max_tokens)
        all_records.extend(records)
        print(f"[done] {pdf_path.name}: {len(records)} chunks")

    write_csv(all_records, args.output_csv)
    total = len(all_records)
    if total:
        max_tokens_seen = max(r["token_estimate"] for r in all_records)
        print(f"[done] Wrote {total} chunks -> {args.output_csv} (max tokens observed: {max_tokens_seen})")
    else:
        print("[warn] No chunks written.")


if __name__ == "__main__":
    main()
