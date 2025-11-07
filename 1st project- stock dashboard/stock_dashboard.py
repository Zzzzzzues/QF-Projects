#1. Import Libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Choose your tickers
tickers = ["AAPL", "MSFT", "NVDA"]
data = yf.download(
    tickers, start="2022-01-01", end="2025-01-01",
    auto_adjust=False, progress=False
)["Adj Close"]

# Compute daily returns
returns = data.pct_change().dropna()

# Summary statistics
summary = pd.DataFrame({
    "Mean Return": returns.mean() * 252,
    "Volatility": returns.std() * np.sqrt(252),
})
summary["Sharpe Ratio"] = summary["Mean Return"] / summary["Volatility"]
print(summary)

# Plot cumulative returns
 
fig, ax = plt.subplots(figsize=(10,6))
(1 + returns).cumprod().plot(ax=ax)
ax.set_title("Cumulative Returns (Adjusted Close, 2022–2025)")
ax.set_xlabel("Date")
ax.set_ylabel("Growth of $1")
fig.tight_layout()
fig.savefig("cumulative_returns.png", dpi=150)  # <-- file will appear in your repo
print("Saved chart to cumulative_returns.png")

summary.to_csv("summary_stats.csv", index=True)