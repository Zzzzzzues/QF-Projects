# CAPM Project — Beta & Alpha Estimation with Custom Tickers
# ----------------------------------------------------------
# 1) If packages are missing, uncomment the next line:
# !pip install yfinance pandas numpy matplotlib statsmodels

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime

plt.rcParams["figure.figsize"] = (9,6)

# ---------- PARAMETERS (edit these) ----------
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA"]   # You can change later via input()
MARKET_TICKER   = "^GSPC"                    # S&P 500
START_DATE      = "2022-01-01"
END_DATE        = "2025-01-01"

# Risk-free settings (optional). If RF_ANNUAL=None, we run CAPM without excess returns.
# If you want to include a risk-free rate (annual), e.g. 0.05 for 5%, set RF_ANNUAL=0.05
RF_ANNUAL = None
TRADING_DAYS = 252
# --------------------------------------------

def parse_user_tickers(defaults):
    """
    Allow user to type tickers in the notebook/terminal.
    Examples: 'AAPL TSLA AMZN' or 'AAPL,TSLA,AMZN'
    Press Enter to accept defaults.
    """
    try:
        raw = input(f"Enter tickers separated by space/comma (Enter for defaults {defaults}): ").strip()
    except Exception:
        # In some non-interactive environments input() may fail; just use defaults.
        raw = ""
    if not raw:
        return defaults
    parts = [p.strip().upper() for p in raw.replace(",", " ").split() if p.strip()]
    return parts if parts else defaults

def fetch_prices(tickers, market, start, end):
    all_syms = list(set(tickers + [market]))
    raw = yf.download(all_syms, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"].copy()
    return prices

def compute_returns(prices, rf_annual=None):
    rets = prices.pct_change().dropna()
    if rf_annual is None:
        return rets, None
    # Convert annual RF to approximate daily RF
    rf_daily = (1 + rf_annual)**(1/TRADING_DAYS) - 1
    rf_series = pd.Series(rf_daily, index=rets.index)
    return rets, rf_series

def run_capm(returns_df, market_col, rf_daily_series=None):
    market = returns_df[market_col].copy()
    results = []
    for col in returns_df.columns:
        if col == market_col:
            continue

        y = returns_df[col]
        x = market

        if rf_daily_series is not None:
            y = y - rf_daily_series
            x = x - rf_daily_series

        X = sm.add_constant(x)  # [1, market_return]
        model = sm.OLS(y, X, missing='drop').fit()

        alpha = model.params.get('const', np.nan)
        beta  = model.params.get(market_col, np.nan)
        r2    = model.rsquared

        results.append({
            "Ticker": col,
            "Alpha (daily)": alpha,
            "Beta": beta,
            "R²": r2
        })
    return pd.DataFrame(results).set_index("Ticker")

def plot_capm_scatter(returns_df, market_col, summary_df, rf_daily_series=None, out_dir="."):
    market = returns_df[market_col].copy()
    for ticker in summary_df.index:
        y = returns_df[ticker]
        x = market
        if rf_daily_series is not None:
            y = y - rf_daily_series
            x = x - rf_daily_series

        # Regression line using estimated params
        alpha = summary_df.loc[ticker, "Alpha (daily)"]
        beta  = summary_df.loc[ticker, "Beta"]
        line  = alpha + beta * x

        ax = plt.gca()
        ax.clear()
        ax.scatter(x, y, s=8, alpha=0.5)
        ax.plot(x, line, linewidth=2)
        ax.set_title(f"CAPM Regression: {ticker} vs {market_col}")
        ax.set_xlabel(f"{market_col} Daily Return")
        ax.set_ylabel(f"{ticker} Daily Return")
        plt.tight_layout()
        fname = f"{out_dir}/capm_{ticker}.png"
        plt.savefig(fname, dpi=150)
        print(f"Saved plot: {fname}")

def annualize_alpha(alpha_daily):
    # Simple trading-days scaling (approx). Alpha is tricky to annualize; this is for illustration.
    return alpha_daily * TRADING_DAYS

def pretty_summary(summary_df):
    out = summary_df.copy()
    out["Alpha (annual approx)"] = out["Alpha (daily)"].apply(annualize_alpha)
    return out[["Beta", "R²", "Alpha (daily)", "Alpha (annual approx)"]]

# ===== RUN =====
user_tickers = parse_user_tickers(DEFAULT_TICKERS)
print("Using tickers:", user_tickers)

prices = fetch_prices(user_tickers, MARKET_TICKER, START_DATE, END_DATE)
returns, rf_daily = compute_returns(prices, RF_ANNUAL)

summary = run_capm(returns, MARKET_TICKER, rf_daily)
print(pretty_summary(summary).round(4))

# Save summary
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
summary_path = f"beta_summary_{timestamp}.csv"
pretty_summary(summary).round(6).to_csv(summary_path)
print(f"Saved summary: {summary_path}")

# Plots
plot_capm_scatter(returns, MARKET_TICKER, summary, rf_daily, out_dir=".")