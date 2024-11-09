# data_extraction.py

import yfinance as yf
import pandas as pd

def fetch_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        data[ticker] = yf.download(ticker, start=start_date, end=end_date)
    return data

if __name__ == "__main__":
    tickers = ["TSLA", "BND", "SPY"]
    start_date = "2015-01-01"
    end_date = "2024-10-31"
    data = fetch_data(tickers, start_date, end_date)
    
    # Save to CSV for each ticker
    for ticker in data:
        data[ticker].to_csv(f"{ticker}_data.csv")
