import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from typing import Tuple

TRADING_DAYS = 252


# 1) Data loading
def get_price_history(ticker: str,
                      start: str = "2020-01-01",
                      end: str = "2025-01-01") -> pd.Series:
    """
    Download adjusted close prices for a single ticker.
    Returns a Series indexed by date.
    """
    data = yf.download(ticker,
                       start=start,
                       end=end,
                       auto_adjust=True,
                       progress=False)

    if "Close" not in data.columns:
        raise ValueError(f"No 'Close' column found for {ticker}.")

    prices = data["Close"].dropna()

    if prices.empty:
        raise ValueError(f"No price data returned for {ticker} "
                         f"between {start} and {end}.")

    return prices


# 2) Estimate GBM parameters from historical log-returns
# 2) Estimate GBM parameters from historical log-returns
def estimate_gbm_params(prices: pd.Series) -> Tuple[float, float, pd.Series]:
    """
    Estimate annualised drift (mu) and volatility (sigma)
    from historical log-returns.
    """
    # log returns
    log_rets = np.log(prices / prices.shift(1)).dropna()

    # daily stats
    mu_daily = log_rets.mean()
    sigma_daily = log_rets.std()

    # annualise (convert to floats)
    mu_annual = float(mu_daily * TRADING_DAYS)
    sigma_annual = float(sigma_daily * np.sqrt(TRADING_DAYS))

    return mu_annual, sigma_annual, log_rets


# 3) Monte Carlo simulation of future price paths (GBM)
def monte_carlo_paths(S0: float,
                      mu: float,
                      sigma: float,
                      days: int = 252,
                      sims: int = 10_000,
                      seed: int = 42) -> np.ndarray:
    """
    Simulate GBM price paths.

    Returns an array of shape (days+1, sims)
    where row 0 is the starting price S0.
    """
    np.random.seed(seed)
    dt = 1 / TRADING_DAYS

    # random shocks
    Z = np.random.normal(0.0, 1.0, size=(days, sims))

    # GBM log-return increments
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_increments = drift + diffusion

    # cumulative sum in log space
    log_S0 = np.log(S0)
    log_paths = np.vstack([
        np.full(shape=(1, sims), fill_value=log_S0),
        log_S0 + np.cumsum(log_increments, axis=0)
    ])

    # back to price space
    price_paths = np.exp(log_paths)
    return price_paths


# 4) Plotting utilities
def plot_price_paths(price_paths: np.ndarray,
                     ticker: str,
                     out_file: str = "price_paths.png",
                     n_sample_paths: int = 100) -> None:
    """
    Plot a subset of simulated price paths.
    """
    days_plus_one, sims = price_paths.shape
    n = min(n_sample_paths, sims)
    time_axis = np.arange(days_plus_one)

    plt.figure(figsize=(8, 5))
    plt.plot(time_axis, price_paths[:, :n], linewidth=0.8, alpha=0.7)
    plt.xlabel("Days into future")
    plt.ylabel("Price")
    plt.title(f"{ticker} – Monte Carlo Simulated Price Paths")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


def plot_final_distribution(price_paths: np.ndarray,
                            ticker: str,
                            out_file: str = "final_price_distribution.png") -> None:
    """
    Plot histogram of final simulated prices.
    """
    final_prices = price_paths[-1, :]

    plt.figure(figsize=(8, 5))
    plt.hist(final_prices, bins=60)
    plt.xlabel("Final simulated price")
    plt.ylabel("Frequency")
    plt.title(f"{ticker} – Distribution of Price after Simulation Horizon")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


# 5) Simple summary statistics
def summarise_simulation(price_paths: np.ndarray) -> dict:
    """
    Compute summary stats from simulated final prices.
    """
    final_prices = price_paths[-1, :]

    stats = {
        "mean_final": float(final_prices.mean()),
        "median_final": float(np.median(final_prices)),
        "p5": float(np.percentile(final_prices, 5)),
        "p95": float(np.percentile(final_prices, 95)),
        "min": float(final_prices.min()),
        "max": float(final_prices.max())
    }
    return stats


# 6) Main driver
def main():
    # --- user input (interactive, but has defaults) ---
    ticker = input("Enter ticker [default AAPL]: ").strip().upper() or "AAPL"
    start = input("Start date [default 2020-01-01]: ").strip() or "2020-01-01"
    end = input("End date   [default 2025-01-01]: ").strip() or "2025-01-01"

    try:
        days = int(input("Forecast horizon in trading days [default 252]: ").strip() or "252")
    except ValueError:
        days = 252

    try:
        sims = int(input("Number of simulations [default 10000]: ").strip() or "10000")
    except ValueError:
        sims = 10_000

    print(f"\nDownloading data for {ticker} from {start} to {end}...")
    prices = get_price_history(ticker, start, end)
    S0 = float(prices.iloc[-1].item())
    print(f"Last observed price (S0): {S0:.2f}")

    mu, sigma, log_rets = estimate_gbm_params(prices)
    print(f"Estimated annual drift (mu):   {mu:.4f}")
    print(f"Estimated annual volatility σ: {sigma:.4f}")

    print(f"\nSimulating {sims} paths over {days} trading days...")
    paths = monte_carlo_paths(S0, mu, sigma, days=days, sims=sims)

    stats = summarise_simulation(paths)
    print("\n== Simulation summary ==")
    for k, v in stats.items():
        print(f"{k:12s}: {v:.2f}")

    # simple risk-style stats
    final_prices = paths[-1, :]
    prob_up = (final_prices > S0).mean()
    prob_down_20 = (final_prices < 0.8 * S0).mean()

    print(f"\nProbability final price > current price: {prob_up:.1%}")
    print(f"Probability final price < 80% of current: {prob_down_20:.1%}")

    # plots
    plot_price_paths(paths, ticker)
    plot_final_distribution(paths, ticker)
    print("\nSaved plots: 'price_paths.png', 'final_price_distribution.png'")


if __name__ == "__main__":
    main()