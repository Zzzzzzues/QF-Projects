"""
Project 5 – Portfolio Value-at-Risk (VaR) & Expected Shortfall (ES)
-------------------------------------------------------------------
- Supports a preset universe (Mag7) or up to 12 custom tickers
- Downloads historical prices using yfinance
- Computes daily log returns & covariance
- Simulates portfolio return distribution via multivariate normal (GBM assumption)
- Calculates VaR and ES at a chosen confidence level
- Plots loss distribution with VaR & ES markers
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from typing import List, Tuple

TRADING_DAYS = 252
MAX_TICKERS = 12

PRESETS = {
    "Mag7": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA"]
}


# ---------- 1) Helper: parse tickers ----------

def parse_custom_tickers(raw: str) -> List[str]:
    """
    Parse comma/space separated ticker string -> unique uppercase list.
    Example: "aapl, msft nvda" -> ["AAPL", "MSFT", "NVDA"]
    """
    if not raw:
        return []
    parts = [p.strip().upper() for p in raw.replace(",", " ").split()]
    seen, out = set(), []
    for t in parts:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def choose_tickers(preset_name: str = "Mag7", custom_raw: str = "") -> List[str]:
    if preset_name != "Custom":
        tickers = PRESETS.get(preset_name, PRESETS["Mag7"]).copy()
    else:
        tickers = parse_custom_tickers(custom_raw)
        if not tickers:
            raise ValueError("Custom preset selected but no valid tickers were provided.")

    if len(tickers) > MAX_TICKERS:
        print(f"[WARN] Provided {len(tickers)} tickers; capping to {MAX_TICKERS}.")
        tickers = tickers[:MAX_TICKERS]

    return tickers


# ---------- 2) Download prices ----------

def fetch_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Returns DataFrame of adjusted Close prices (index=date, columns=tickers).
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # Handle single vs multi-column result
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw["Close"].to_frame()
        prices.columns = [tickers[0]]

    prices = prices.dropna(how="any")
    if prices.shape[0] == 0:
        raise ValueError("No overlapping history after dropping NA rows.")
    return prices


# ---------- 3) Compute daily log returns & covariance ----------

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns for each asset.
    """
    log_rets = np.log(prices / prices.shift(1)).dropna()
    return log_rets


def estimate_params(log_rets: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Estimate daily mean vector and covariance matrix of log returns.
    (We scale to horizon later.)
    """
    mu_daily = log_rets.mean()      # Series, per asset
    cov_daily = log_rets.cov()      # DataFrame
    return mu_daily, cov_daily


# ---------- 4) Monte Carlo simulation of portfolio returns ----------

def simulate_portfolio_returns(
    mu_daily: pd.Series,
    cov_daily: pd.DataFrame,
    weights: np.ndarray,
    horizon_days: int = 1,
    sims: int = 10_000,
    seed: int = 42
) -> np.ndarray:
    """
    Simulate portfolio log-returns over a horizon, under multivariate normal assumption.
    Returns array of simulated arithmetic returns (not log).
    """
    np.random.seed(seed)
    tickers = mu_daily.index.tolist()
    n = len(tickers)

    mu_h = mu_daily.values * horizon_days
    cov_h = cov_daily.values * horizon_days

    # sample multivariate normal for asset log returns
    draws = np.random.multivariate_normal(mean=mu_h, cov=cov_h, size=sims)  # (sims, n)

    # portfolio log-returns
    port_log_rets = draws @ weights  # (sims, )

    # convert to arithmetic returns: r = e^{log_r} - 1
    port_rets = np.exp(port_log_rets) - 1.0
    return port_rets


# ---------- 5) VaR & ES calculation ----------

def var_es_from_returns(
    port_rets: np.ndarray,
    alpha: float = 0.95
) -> Tuple[float, float]:
    """
    Given simulated portfolio returns, compute VaR and ES at confidence alpha.
    We define losses = -returns (positive = loss).
    VaR_alpha = loss threshold not exceeded with probability alpha.
    ES_alpha  = average loss conditional on loss >= VaR_alpha.
    """
    losses = -port_rets  # loss > 0 means losing money

    var_level = 100 * alpha
    var = np.percentile(losses, var_level)
    tail_losses = losses[losses >= var]

    es = tail_losses.mean() if tail_losses.size > 0 else var
    return float(var), float(es)


# ---------- 6) Plot distribution ----------

def plot_loss_distribution(
    port_rets: np.ndarray,
    var: float,
    es: float,
    alpha: float,
    title: str = "Portfolio Loss Distribution",
    out_file: str = "loss_distribution.png"
) -> None:
    losses = -port_rets

    plt.figure(figsize=(8, 5))
    plt.hist(losses, bins=60)
    plt.axvline(var, linestyle="--", label=f"VaR {int(alpha*100)}% = {var:.4f}")
    plt.axvline(es, linestyle=":", label=f"ES {int(alpha*100)}% = {es:.4f}")
    plt.xlabel("Loss (positive = loss)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


# ---------- 7) Main ----------

def main():
    print("=== Project 5: Portfolio VaR & ES (Monte Carlo) ===\n")

    # --- inputs ---
    preset_name = input("Preset [Mag7/Custom] (default Mag7): ").strip() or "Mag7"
    if preset_name not in PRESETS and preset_name != "Custom":
        print(f"[WARN] Unknown preset '{preset_name}', falling back to Mag7.")
        preset_name = "Mag7"

    custom_raw = ""
    if preset_name == "Custom":
        custom_raw = input("Enter up to 12 tickers (comma/space separated): ")

    start = input("Start date [default 2020-01-01]: ").strip() or "2020-01-01"
    end = input("End date   [default 2025-01-01]: ").strip() or "2025-01-01"

    try:
        horizon_days = int(input("Horizon in trading days [default 1]: ").strip() or "1")
    except ValueError:
        horizon_days = 1

    try:
        sims = int(input("Number of simulations [default 10000]: ").strip() or "10000")
    except ValueError:
        sims = 10_000

    try:
        alpha_str = input("Confidence level for VaR/ES [default 0.95]: ").strip() or "0.95"
        alpha = float(alpha_str)
    except ValueError:
        alpha = 0.95

    # --- workflow ---
    tickers = choose_tickers(preset_name, custom_raw)
    print(f"\nUsing tickers ({len(tickers)}): {tickers}")
    print(f"Date range: {start} → {end}")

    prices = fetch_prices(tickers, start, end)
    log_rets = compute_log_returns(prices)
    mu_daily, cov_daily = estimate_params(log_rets)

    n = len(tickers)
    weights = np.full(shape=n, fill_value=1.0 / n)  # equal-weight

    print(f"\nSimulating {sims} scenarios over horizon = {horizon_days} day(s)...")
    port_rets = simulate_portfolio_returns(
        mu_daily, cov_daily, weights, horizon_days=horizon_days, sims=sims
    )

    var, es = var_es_from_returns(port_rets, alpha=alpha)

    print("\n== Risk Summary ==")
    print(f"Horizon: {horizon_days} day(s)")
    print(f"Confidence: {int(alpha*100)}%")
    print(f"VaR  (loss): {var:.4f}")
    print(f"ES   (loss): {es:.4f}")
    print(f"Mean return: {port_rets.mean():.4f}")
    print(f"Min  return: {port_rets.min():.4f}")
    print(f"Max  return: {port_rets.max():.4f}")

    title = f"Portfolio Loss Distribution (alpha={alpha:.2f}, horizon={horizon_days}d)"
    plot_loss_distribution(port_rets, var, es, alpha, title=title)

    print("\nSaved plot: loss_distribution.png")


if __name__ == "__main__":
    main()