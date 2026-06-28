# Sauder Capital Markets Competition (SCMC) 2025 - Investment Research

**DECK**
[![SCMC Deck](https://github.com/user-attachments/assets/182200b6-8349-46ea-953e-f62f6e175376)](https://drive.google.com/file/d/1At5mxMRDwwqfZ3qowksNBsdDj7Cebgct/view?usp=sharing)

This repository contains the analytical framework and tools developed for the Sauder Capital Markets Competition 2025. The project integrates advanced Natural Language Processing (NLP) for sentiment analysis with rigorous quantitative financial modeling to drive informed investment decisions.

## 🚀 Core Components

### 🧠 NLP Sentiment Engine
A sophisticated pipeline designed to transform unstructured financial text into actionable sentiment headers.
- **Model**: Powered by **Hugging Face Transformers**, specifically utilizing `ProsusAI/finbert`—a BERT model pre-trained on financial corpora for superior sentiment classification.
- **Data Universe**:
    - **SEC Filings**: US 10-Q and 8-K filings filtered for critical items (2.02, 7.01, 8.01).
    - **Canadian Filings**: SEDAR+ Press Releases and MD&A sections (serving as Canadian equivalents to SEC filings).
    - **News Feed**: Real-time headlines aggregated from Yahoo Finance and NewsAPI.
- **Technical Excellence**:
    - **Data Preparation**: Automated cleaning of HTML/SGML, removal of boilerplate/tables, and normalization of financial text.
    - **Intelligent Chunking**: Implements a sliding window strategy, capping text at ~250 words per chunk. This ensures high-fidelity sentiment signals while strictly adhering to BERT’s 512-token limit.
    - **Composite Scoring**: Calculates a bespoke `cwds_score` that weights positive and negative probabilities against neutral confidence.

### 📊 Quantitative Modeling
A robust suite of financial models implemented to evaluate asset risk and return profiles.
- **Valuation Frameworks**: 
    - **IRR & Implied IRR**: Calculating internal rates of return to assess project and equity attractiveness.
    - **CAPM**: Capital Asset Pricing Model implementation for estimating expected returns based on systematic risk (Beta).
- **Risk Assessment**: RAR (Risk-Adjusted Return) analysis to normalize performance across varying volatility levels.

## 📂 Directory Structure

```text
.
├── context/               # Competition case materials & team deliverables
├── nlp/                   # Sentiment analysis pipeline
│   ├── processed_data/    # Final scored datasets (CSVs)
│   ├── raw_data/          # Scraped filings and news
│   ├── utils/             # Scrapers and extraction logic
│   └── finbert.ipynb      # Main NLP analysis & scoring notebook
├── quant/                 # Financial modeling suite
│   ├── data/              # Market & financial data
│   └── *.ipynb            # CAPM, IRR, and RAR analysis notebooks
└── requirements.txt       # Project dependencies
```

## 🛠️ Getting Started

### Prerequisites
- Python 3.9+
- Virtual Environment (recommended)

### Installation
1. Clone the repository.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration
Ensure you have a `.env` file in the root directory for API access (e.g., NewsAPI keys).

## 🧰 Tech Stack
- **Deep Learning**: PyTorch, Hugging Face Transformers
- **Data Science**: Pandas, NumPy, Statsmodels
- **Financial Data**: yfinance, SecEdgar
- **Visualization**: Matplotlib, Seaborn
- **NLP Utilities**: Beautifulsoup4, PDFPlumber, Trafilatura

---
*Developed for the Sauder Capital Markets Competition 2025.*
