import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import datetime

CACHE_FILE = "sp500_tickers.csv"
STOCK_DATA_CACHE = "stock_data.csv"
RETURNS_CACHE = "returns_data.csv"

def get_sp500_tickers():
    # If cache file exists, load from it
    if os.path.exists(CACHE_FILE):
        print("📁 Loading S&P 500 tickers from cache...")
        return pd.read_csv(CACHE_FILE)
    
    print("🌐 Fetching S&P 500 tickers from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(url, headers=headers)
    
    # Parse HTML table
    sp500_table = pd.read_html(StringIO(response.text))[0]
    sp500_tickers = sp500_table[['Symbol', 'Security', 'GICS Sector']]
    sp500_tickers.columns = ['Ticker', 'Name', 'Sector']
    
    # Save to cache for next time
    sp500_tickers.to_csv(CACHE_FILE, index=False)
    print(f"✅ S&P 500 tickers saved to cache ({CACHE_FILE})")
    
    return sp500_tickers

sp500_tickers = get_sp500_tickers()

# Step 2: Function to fetch stock data safely
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'Ticker': ticker,
            'P/E': info.get('trailingPE', np.nan),
            'P/B': info.get('priceToBook', np.nan),
            'P/S': info.get('priceToSalesTrailing12Months', np.nan),
            'EV/EBITDA': info.get('enterpriseToEbitda', np.nan)
        }
    except:
        return None

# Step 3: Fetch valuation data with caching
if os.path.exists(STOCK_DATA_CACHE):
    print("📁 Loading stock data from cache...")
    df = pd.read_csv(STOCK_DATA_CACHE)
else:
    print("🌐 Fetching stock data...")
    data = []
    for ticker in tqdm(sp500_tickers['Ticker']):
        stock_data = fetch_stock_data(ticker)
        if stock_data:
            data.append(stock_data)
    df = pd.DataFrame(data)
    df.to_csv(STOCK_DATA_CACHE, index=False)
    print(f"✅ Stock data saved to cache ({STOCK_DATA_CACHE})")

# Step 4: Merge with sector info
df = pd.merge(df, sp500_tickers, on='Ticker')

# Step 5: Compute percentile ranks and RV Score
metrics = ['P/E', 'P/B', 'P/S', 'EV/EBITDA']
for metric in metrics:
    df[metric + '_Percentile'] = df[metric].rank(pct=True, ascending=True)

df['RV_Score'] = df[[m + '_Percentile' for m in metrics]].mean(axis=1)
df = df.sort_values('RV_Score')

top_50 = df.head(50)
print("\nTop 50 Value Stocks:")
print(top_50[['Ticker', 'Name', 'Sector', 'RV_Score']])

# Step 6: Compare 1-year returns vs S&P 500
print("\nFetching 1-year price data for comparison...")
sp500_data = yf.download("^GSPC", period="1y", auto_adjust=True)

if sp500_data.empty:
    print("⚠️ Warning: Failed to fetch S&P 500 index data. Trying SPY ETF as fallback...")
    sp500_data = yf.download("SPY", period="1y", auto_adjust=True)
    if sp500_data.empty:
        print("⚠️ Warning: Failed to fetch SPY ETF data as well. Setting S&P 500 return to 0.")
        sp500_return = 0.0
    else:
        sp500 = sp500_data.get('Adj Close', sp500_data.get('Close'))
        if sp500 is None:
            print("⚠️ Warning: Neither 'Adj Close' nor 'Close' columns found in SPY ETF data.")
            sp500_return = 0.0
        else:
            sp500_return = ((sp500.iloc[-1] - sp500.iloc[0]) / sp500.iloc[0] * 100).item()
else:
    sp500 = sp500_data.get('Adj Close', sp500_data.get('Close'))
    if sp500 is None:
        print("⚠️ Warning: Neither 'Adj Close' nor 'Close' columns found in S&P 500 data.")
        sp500_return = 0.0
    else:
        sp500_return = ((sp500.iloc[-1] - sp500.iloc[0]) / sp500.iloc[0] * 100).item()

# Step 7: Fetch 1-year returns for top 50 stocks with caching
if os.path.exists(RETURNS_CACHE):
    print("📁 Loading 1-year returns from cache...")
    returns_df = pd.read_csv(RETURNS_CACHE)
    # --- FIX: Extract numeric value from string if needed ---
    import re
    def extract_return(val):
        if isinstance(val, str):
            # Find the last float in the string (handles negative and decimal numbers)
            matches = re.findall(r'-?\d+\.\d+', val)
            if matches:
                return float(matches[-1])
            else:
                return np.nan
        return val
    returns_df['Return(1Y)%'] = returns_df['Return(1Y)%'].apply(extract_return)
else:
    # start with empty cache DataFrame
    returns_df = pd.DataFrame(columns=['Ticker', 'Return(1Y)%'])

# ensure ticker strings
returns_df['Ticker'] = returns_df['Ticker'].astype(str) if not returns_df.empty else pd.Series(dtype=str)
fetched = set(returns_df['Ticker'].tolist()) if not returns_df.empty else set()
tickers_to_fetch = [t for t in top_50['Ticker'].astype(str).tolist() if t not in fetched]

if tickers_to_fetch:
    print(f"\n🌐 Fetching 1-year returns for {len(tickers_to_fetch)} tickers (skipping cached)...")
    for ticker in tqdm(tickers_to_fetch):
        try:
            data = yf.download(ticker, period="1y", auto_adjust=True, threads=False)
            # fallback if empty
            if data.empty:
                data = yf.download(ticker, period="1y", auto_adjust=False, threads=False)
            if data.empty:
                print(f"⚠️ No data for ticker: {ticker}")
                continue

            # robustly pick a price series
            price = None
            if 'Adj Close' in data.columns:
                price = data['Adj Close']
            elif 'Close' in data.columns:
                price = data['Close']
            else:
                num_cols = data.select_dtypes(include=[np.number]).columns
                if len(num_cols) > 0:
                    price = data[num_cols[0]]

            if price is None or price.shape[0] < 2:
                print(f"⚠️ Not enough price points for ticker: {ticker}")
                continue

            ret = (price.iloc[-1] - price.iloc[0]) / price.iloc[0] * 100
            # append and persist incrementally to avoid re-fetching next run
            returns_df = pd.concat([returns_df, pd.DataFrame([{'Ticker': ticker, 'Return(1Y)%': ret}])],
                                   ignore_index=True)
            returns_df.to_csv(RETURNS_CACHE, index=False)
        except Exception as e:
            print(f"⚠️ Error fetching data for {ticker}: {e}")
else:
    print("✅ All top-50 returns loaded from cache.")

# dedupe keeping latest fetched value
if not returns_df.empty:
    returns_df = returns_df.drop_duplicates(subset='Ticker', keep='last')
else:
    # no data fetched — proceed with empty dataframe (merge will produce NaNs rather than raising)
    print("⚠️ No return data available in cache or from fetches. Proceeding with NaNs for returns.")

comparison = pd.merge(top_50, returns_df, on='Ticker', how='left')

# Step 8: Simple Backtest
# Convert Return(1Y)% to numeric, coercing errors to NaN
comparison['Return(1Y)%'] = pd.to_numeric(comparison['Return(1Y)%'], errors='coerce')

# Calculate mean excluding NaN values
portfolio_return = comparison['Return(1Y)%'].mean(skipna=True)

if pd.isna(portfolio_return):
    print("\n⚠️ Warning: Could not calculate portfolio return due to missing data")
    portfolio_return = 0
    sp500_return = 0  # Set S&P 500 return to 0 if portfolio return is missing
else:
    print(f"\nAverage Portfolio Return (Top 50): {portfolio_return:.2f}%")
    
print(f"S&P 500 Return (1Y): {sp500_return:.2f}%")

# Print data quality statistics
total_stocks = len(comparison)
stocks_with_returns = comparison['Return(1Y)%'].notna().sum()
print(f"\nData Quality: {stocks_with_returns}/{total_stocks} stocks have return data")

# Step 9: Create Dashboard
print("\nCreating dashboard visualizations...")

# 1. Value Metrics Distribution
fig_metrics = go.Figure()
metrics = ['P/E', 'P/B', 'P/S', 'EV/EBITDA']

for metric in metrics:
    # Remove infinite and null values
    clean_data = df[df[metric].notna() & (df[metric] != np.inf) & (df[metric] > 0)]
    # Remove outliers (values beyond 95th percentile)
    threshold = clean_data[metric].quantile(0.95)
    clean_data = clean_data[clean_data[metric] <= threshold]
    
    fig_metrics.add_trace(go.Box(
        y=clean_data[metric],
        name=metric,
        boxpoints='outliers'
    ))

fig_metrics.update_layout(
    title='Distribution of Value Metrics (Excluding Outliers)',
    template='plotly_dark',
    showlegend=False,
    height=600
)

# 2. Sector-wise Average Metrics
sector_metrics = df.groupby('Sector')[metrics].mean().reset_index()
sector_metrics = sector_metrics.melt(id_vars=['Sector'], value_vars=metrics)

fig_sector = px.bar(
    sector_metrics,
    x='Sector',
    y='value',
    color='variable',
    title='Average Value Metrics by Sector',
    barmode='group',
    template='plotly_dark',
    height=600
)
fig_sector.update_layout(xaxis_tickangle=-45)

# 3. RV Score Distribution by Sector
fig_rv = px.box(
    df,
    x='Sector',
    y='RV_Score',
    title='RV Score Distribution by Sector',
    template='plotly_dark',
    height=600
)
fig_rv.update_layout(xaxis_tickangle=-45)

# 4. Correlation Matrix
correlation = df[metrics].corr()
fig_corr = px.imshow(
    correlation,
    title='Correlation Matrix of Value Metrics',
    template='plotly_dark',
    aspect='auto',
    color_continuous_scale='RdBu'
)

# 5. Top 50 Stocks Overview
# Add dummy return data if none exists
if comparison['Return(1Y)%'].isna().all():
    print("\n⚠️ Adding dummy return data for visualization purposes...")
    # Generate random returns between -30% and +30%
    np.random.seed(42)  # For reproducibility
    comparison['Return(1Y)%'] = np.random.uniform(-30, 30, len(comparison))
    comparison['Return(1Y)%_isDummy'] = True
else:
    comparison['Return(1Y)%_isDummy'] = False

fig_top50 = px.scatter(
    comparison,  # Use comparison instead of top_50
    x='RV_Score',
    y='Return(1Y)%',
    color='Sector',
    hover_data=['Ticker', 'Name'],
    title='Top 50 Value Stocks: RV Score vs 1Y Return' + 
          (' (Using Dummy Returns)' if comparison['Return(1Y)%_isDummy'].all() else ''),
    template='plotly_dark',
    height=600
)

# Add a note about dummy data if used
if comparison['Return(1Y)%_isDummy'].all():
    fig_top50.add_annotation(
        text="Note: Using simulated return data for visualization",
        xref="paper", yref="paper",
        x=0, y=1.05,
        showarrow=False,
        font=dict(size=12, color="yellow"),
        align="left"
    )

# 6. Metrics Comparison Scatter Matrix
fig_scatter = px.scatter_matrix(
    df[metrics + ['RV_Score']],
    title='Value Metrics Relationships',
    template='plotly_dark',
    height=800
)

# Save visualizations to HTML
print("\nSaving dashboard to 'stock_dashboard.html'...")
with open('stock_dashboard.html', 'w', encoding='utf-8') as f:  # <-- add encoding='utf-8'
    f.write('''
    <html>
        <head>
            <title>Stock Market Value Analysis Dashboard</title>
            <style>
                body { background-color: #111; color: white; font-family: Arial, sans-serif; }
                .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
                .chart { margin-bottom: 30px; background-color: #222; padding: 20px; border-radius: 10px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Stock Market Value Analysis Dashboard</h1>
    ''')
    f.write(f'<div class="chart">{fig_metrics.to_html(full_html=False)}</div>')
    f.write(f'<div class="chart">{fig_sector.to_html(full_html=False)}</div>')
    f.write(f'<div class="chart">{fig_rv.to_html(full_html=False)}</div>')
    f.write(f'<div class="chart">{fig_corr.to_html(full_html=False)}</div>')
    f.write(f'<div class="chart">{fig_top50.to_html(full_html=False)}</div>')
    f.write(f'<div class="chart">{fig_scatter.to_html(full_html=False)}</div>')
    f.write('''
            </div>
        </body>
    </html>
    ''')

print("✅ Dashboard created successfully! Open 'stock_dashboard.html' in your browser to view it.")

# Step 10: Create Quant Dashboard
print("\nSaving quant dashboard to 'quant_dashboard.html'...")

# Ensure the file is deleted before writing
if os.path.exists('quant_dashboard.html'):
    os.remove('quant_dashboard.html')

# Extract numeric values from the Return(1Y)% column
import re
def extract_return(val):
    if isinstance(val, str):
        # Find the last float in the string
        matches = re.findall(r'-?\d+\.\d+', val)
        if matches:
            return float(matches[-1])
        else:
            return np.nan
    return val
returns_df['Return(1Y)%'] = returns_df['Return(1Y)%'].apply(extract_return)

# Create visualizations for Quant Dashboard
# 1. Distribution of Returns
fig_returns_dist = px.histogram(
    returns_df,
    x='Return(1Y)%',
    nbins=30,
    title='Distribution of 1-Year Returns',
    template='plotly_dark',
    color_discrete_sequence=['#1f77b4']
)

# 2. Top 10 Performing Stocks
top_10_stocks = returns_df.nlargest(10, 'Return(1Y)%')
fig_top_10 = px.bar(
    top_10_stocks,
    x='Ticker',
    y='Return(1Y)%',
    title='Top 10 Performing Stocks (1-Year Returns)',
    template='plotly_dark',
    color='Return(1Y)%',
    color_continuous_scale='Viridis'
)

# 3. Bottom 10 Performing Stocks
bottom_10_stocks = returns_df.nsmallest(10, 'Return(1Y)%')
fig_bottom_10 = px.bar(
    bottom_10_stocks,
    x='Ticker',
    y='Return(1Y)%',
    title='Bottom 10 Performing Stocks (1-Year Returns)',
    template='plotly_dark',
    color='Return(1Y)%',
    color_continuous_scale='Viridis'
)

# Save the Quant Dashboard
with open('quant_dashboard.html', 'w', encoding='utf-8') as f:
    f.write('''
    <html>
        <head>
            <title>Quant Dashboard</title>
            <style>
                body {{ background-color: #111; color: white; font-family: Arial, sans-serif; }}
                .container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
                .chart {{ margin-bottom: 30px; background-color: #222; padding: 20px; border-radius: 10px; }}
                h1, h2 {{ text-align: center; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Quant Dashboard</h1>
                <h2>Key Insights from Returns Data</h2>
                <div class="chart">{returns_dist}</div>
                <div class="chart">{top_10}</div>
                <div class="chart">{bottom_10}</div>
            </div>
        </body>
    </html>
    '''.format(
        returns_dist=fig_returns_dist.to_html(full_html=False),
        top_10=fig_top_10.to_html(full_html=False),
        bottom_10=fig_bottom_10.to_html(full_html=False)
    ))

print("✅ Quant dashboard created successfully! Open 'quant_dashboard.html' in your browser to view it.")
