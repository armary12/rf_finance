import yfinance as yf
import pandas as pd

# List of tickers
tickers = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "BRK-B", "LLY", "TSM", "AVGO", "TSLA",
    "JPM", "NVO", "WMT", "UNH", "XOM", "V", "MA", "PG", "JNJ", "COST", "ORCL", "HD", "ABBV",
    "ASML", "BAC", "KO", "MRK", "NFLX", "CVX", "AZN", "CRM", "SAP", "TM", "PEP", "ADBE", 
    "TMO", "TMUS", "NVS", "AMD", "SHEL", "LIN", "ACN", "APD", "MCD", "DHR", "ABT", "BABA", 
    "QCOM", "CSCO", "PM", "WFC", "GE", "TXN", "INTU", "IBM", "AMGN", "AXP", "VZ", "NOW", 
    "ISRG", "AMAT", "GS", "PFE", "CAT", "MS", "NEE", "RTX", "DIS", "TTE", "RY", "SPGI", "UL",
    "CMCSA", "HSBC", "FCX", "UNP", "HDB", "T", "PGR", "LOW", "LMT", "SNY", "BHP", "NXPI", 
    "HON", "COP", "BLK", "SYK", "ELV", "REGN", "TJX", "VRTX", "BKNG", "SCHW", "BUD", "ETN", 
    "NKE", "MUFG", "PLD", "C", "BSX", "MU", "PANW", "LRCX", "ANET", "CB", "SONY", "MMC", 
    "UPS", "RIO", "KLAC", "ADI", "ADP", "KKR", "AMT", "SBUX", "MDT", "BA", "BX", "TD", "BMY", 
    "IBN", "DE", "HCA", "MELI", "SO", "MDLZ", "CI", "FI", "UBS", "GILD", "SHOP", "PBR", 
    "PBR-A", "BP", "ICE", "INFY", "SHW", "DUK", "MO", "INTC", "ENB", "MCO", "RELX", "SMFG", 
    "CL", "GSK", "ZTS", "WM", "SNPS", "GD", "RACE", "EQIX", "BTI", "SCCO", "APH", "TT", 
    "CTAS", "EQNR", "CNQ", "CME", "CDNS", "PH", "NGG", "NOC", "TRI", "CP", "DELL", "MCK", 
    "EOG", "WELL", "AON", "CMG", "ITW", "CVS", "DEO", "MAR", "SAN", "CNI", "TDG", "MSI", 
    "FDX", "MMM", "BN", "ECL", "BDX", "PYPL", "BMO", "PNC", "USB", "ORLY", "CSX", "RSG", 
    "TGT", "SLB", "EPD", "AJG", "APO", "MPC"
]

# Define the date range
start_date = "2020-01-01"
end_date = "2022-12-31"

# Download data for each ticker
for ticker in tickers:
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    data.to_csv(f"data/us/2020-2022/1d/{ticker}.csv")
    print(f"Data for {ticker} downloaded and saved as {ticker}_2014_2020.csv")
