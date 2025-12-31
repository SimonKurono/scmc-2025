## NLP Sentiment Pipeline (FinBERT)

This folder contains the NLP workflow to collect news/filing text, clean it, score sentiment with FinBERT, and aggregate results.

### Data Sources
- **Yahoo Finance (`yfinance`)**: Headlines per ticker (NKE, LULU, ATZ.TO). Limited (~10) per ticker.
- **NewsAPI**: Additional headlines; merged with Yahoo output.
- **Bloomberg PDFs/Reports**: Parsed into sections/chunks via custom extractors.
- **SEC Filings**: 10-Q and 8-K (Items 2.02, 7.01, 8.01) scraped into text, filtered by keywords, and chunked.
- **MD&A**: Manager's discussion and analysis as replacement for SEC's 10-Q for Aritizia as it's Canadian
- **SEDAR+ Press Releases**: Company press releases as replacement for SEC's 8-K for Aritizia as it's Canadian

### Cleaning & Preparation
- Normalize text, strip HTML/SGML, remove tables/boilerplate where possible.
- Deduplicate rows (optionally on `text` or `text + source` depending on provenance needs).
- SEC filings: slice by item headers, keep top keyword-rich paragraphs, chunk to ~250 words to stay under 512 tokens.
- News: standardize columns (`ticker`, `date`, `source`, `title/text`, `summary`, `link`), drop obvious duplicates.

### FinBERT Scoring
- Model: `ProsusAI/finbert` (sequence classification).
- For each chunk/text:
  - Run tokenizer/model → `finbert_neg`, `finbert_neu`, `finbert_pos`.
  - Compute composite score `cwds_score = (finbert_pos - finbert_neg) * (1 - finbert_neu)`.
  - Optional label: argmax of probabilities or sign-based from `cwds_score`.

### Aggregation & Outputs
- Master corpus CSVs under `nlp/processed_data/...` (e.g., `master_corpus_final_finbert.csv`, `..._metrics.csv`) with scores and metadata.
- Example analyses in `finbert.ipynb`:
  - Sentiment over time (by date/ticker, 2024+ filters).
  - Mean sentiment by source with counts.
  - Scatter plots of `cwds_score` vs date.
  - Label/score summaries by source/ticker.

### Re-running the Pipeline (high level)
1. **Collect**: run news scrapers and SEC scraper (`nlp/utils/sec_filing_scraper.py`) to refresh CSVs.
2. **Clean/Dedupe**: drop duplicates on chosen keys; ensure dates are datetime.
3. **Score**: run FinBERT scoring in `finbert.ipynb` to produce `finbert_*` probs and `cwds_score`.
4. **Post-filter**: apply any business rules (e.g., remove worst-scoring subsets).
5. **Analyze**: use notebook cells for plots/metrics; export final CSVs under `nlp/processed_data/master/`.

### Notes
- Token-length guard: SEC chunks are capped to ~250 words to stay under 512 BERT tokens.
- Adjust thresholds/windows in the notebook if you want different neutral bands, rolling windows, or stricter caps.
- When merging sources, keep a `source` column for provenance; dedupe on `text` if you prefer unique language over unique sources.
