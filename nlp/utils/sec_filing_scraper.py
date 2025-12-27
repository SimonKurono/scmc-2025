from secedgar import filings, FilingType

# 10Q filings for Apple and Facebook (tickers "aapl" and "fb")
my_filings = filings(cik_lookup=["nke","lulu"],
                     filing_type=FilingType.FILING_10Q,
                     user_agent="Simon Kurono (simonkurono@gmail.com)")
my_filings.save('./nlp/raw_data/sec_filings')