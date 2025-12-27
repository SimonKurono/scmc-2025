import yfinance as yf

import pandas as pd


ticker = yf.Ticker("AAPL")

#get news

news_list = ticker.news

print(news_list[0])