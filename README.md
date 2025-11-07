

Step 1: Import Libraries
import os, pandas as pd, numpy as np, yfinance as yf, requests
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import datetime


os → file handling

pandas/numpy → data handling & math

yfinance → fetch stock data from Yahoo Finance

requests → download web pages (Wikipedia)

tqdm → progress bar for loops

plotly → data visualization (interactive graphs)

Step 2: Define Cache Files
CACHE_FILE = "sp500_tickers.csv"
STOCK_DATA_CACHE = "stock_data.csv"
RETURNS_CACHE = "returns_data.csv"


Used to save fetched data locally so the next time you run it, it won’t re-download everything.
(Caching saves time & API calls.)

Step 3: Get the List of S&P 500 Companies
def get_sp500_tickers():
    if os.path.exists(CACHE_FILE):
        return pd.read_csv(CACHE_FILE)


Checks if a local CSV file already has ticker data.

If yes → loads it from cache.

If not, it fetches from Wikipedia:

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
response = requests.get(url)
sp500_table = pd.read_html(StringIO(response.text))[0]


Reads the HTML table of S&P 500 companies.

Extracts Ticker, Company Name, and Sector.

Saves it as sp500_tickers.csv for future runs.

Step 4: Fetch Stock Valuation Data
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info


For each ticker:

Fetches data from Yahoo Finance API.

Extracts valuation ratios:

P/E = Price-to-Earnings

P/B = Price-to-Book

P/S = Price-to-Sales

EV/EBITDA = Enterprise Value to EBITDA

These are stored in a DataFrame and cached as stock_data.csv.

Step 5: Compute RV Score (Relative Value)
df = pd.merge(df, sp500_tickers, on='Ticker')
for metric in metrics:
    df[metric + '_Percentile'] = df[metric].rank(pct=True, ascending=True)


Merges financial ratios with company info (sector, name).

Calculates percentile rank for each metric (e.g. a lower P/E → higher value score).

Averages all four percentiles to get:

df['RV_Score'] = df[[...]].mean(axis=1)


RV_Score = overall measure of how undervalued the stock is.

Sorts by RV_Score and selects the Top 50 value stocks.

Step 6: Get S&P 500 Index 1-Year Return
sp500_data = yf.download("^GSPC", period="1y", auto_adjust=True)


Downloads last 1 year of S&P 500 index data.

If it fails → tries SPY ETF as a backup.

Calculates percentage return over 1 year.

Step 7: Fetch Each Stock’s 1-Year Return

Uses caching via returns_data.csv

Only downloads missing tickers (not cached yet)

For each ticker:

data = yf.download(ticker, period="1y")
ret = (price.iloc[-1] - price.iloc[0]) / price.iloc[0] * 100


→ computes 1-year return percentage.

Saves this incrementally (so if it crashes mid-way, progress isn’t lost).

Step 8: Compare Portfolio vs S&P 500
portfolio_return = comparison['Return(1Y)%'].mean(skipna=True)


Average 1-year return of the top 50 value stocks.

Compares with S&P 500 return to see if the “value portfolio” beats the market.

Also prints data completeness stats (how many stocks had valid return data).

Step 9: Build Interactive Dashboards (HTML)
🧮 Dashboard 1: stock_dashboard.html

This dashboard visualizes valuation metrics and RV score analysis.

Charts included:

Box Plot of Value Metrics → how P/E, P/B, etc. are distributed.

Bar Chart → average metrics by sector.

Box Plot → RV Score by sector.

Correlation Matrix → correlation between valuation metrics.

Scatter Plot → RV Score vs 1-year return for top 50.

Scatter Matrix → relationships among valuation metrics.

If no real return data exists, it generates dummy random returns (for display).

All charts are written inside an HTML file using:

f.write(f'<div class="chart">{fig_metrics.to_html(full_html=False)}</div>')

Step 10: Quant Dashboard (quant_dashboard.html)

This second dashboard focuses purely on returns.

Histogram of all 1-year returns → distribution.

Top 10 Performing Stocks → bar chart.

Bottom 10 Performing Stocks → bar chart.

returns_df['Return(1Y)%'] = returns_df['Return(1Y)%'].apply(extract_return)


Ensures returns are numeric (even if cached as strings).

Saves everything into a styled HTML dashboard.

📈 Final Output

After running the script, you get:

sp500_tickers.csv → list of S&P 500 companies

stock_data.csv → valuation metrics

returns_data.csv → 1-year returns

stock_dashboard.html → detailed analysis of valuation metrics

quant_dashboard.html → performance-based dashboard

🧩 Summary
Step	Task	Output
1	Get S&P 500 tickers	sp500_tickers.csv
2	Fetch valuation ratios	stock_data.csv
3	Rank stocks by RV score	Top 50 Value Stocks
4	Compare with S&P 500 returns	Printed stats
5	Build dashboards	stock_dashboard.html, quant_dashboard.html
