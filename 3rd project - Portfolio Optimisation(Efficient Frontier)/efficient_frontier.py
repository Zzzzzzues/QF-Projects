# Efficient Frontier / Markowitz Portfolio Optimization
# ----------------------------------------------------
# Default: Magnificent 7 tickers, with option for custom user input.
# Outputs: efficient_frontier.png, weights_max_sharpe.csv, weights_min_vol.csv, params.json

# 1) Imports
import json
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# 2) Global parameters
SEED = 42
np.random.seed(SEED)

START = "2022-01-01"
END   = "2025-01-01"
RF    = 0.00          # risk-free rate (annual). Start with 0 for simplicity.
TRADING_DAYS = 252

N_PORTFOLIOS = 10_000
MAX_TICKERS  = 12     # cap to keep the frontier readable & simulation stable

PRESETS = {
    "Mag7":  ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA"],
    "Tech5": ["AAPL","MSFT","NVDA","AVGO","ORCL"],
    "ETF6":  ["SPY","QQQ","IWM","EFA","EEM","AGG"]
}


# 3) Ticker selection helpers
def parse_custom_tickers(raw: str) -> List[str]:
    parts = [p.strip().upper() for p in raw.replace(",", " ").split() if p.strip()]
    # de-dupe while preserving order
    seen, out = set(), []
    for t in parts:
        if t not in seen:
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

# 4) Data download & preparation
def fetch_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        # single ticker case -> ensure 2D frame
        prices = raw.to_frame()
        prices.columns = [tickers[0]]
    # Drop rows with any NA (align history)
    prices = prices.dropna(how="any")
    if prices.shape[0] == 0:
        raise ValueError("No overlapping price history after dropna(). Try different dates/tickers.")
    return prices

def compute_stats(prices: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    rets = prices.pct_change().dropna()
    mean_annual = rets.mean() * TRADING_DAYS
    cov_annual  = rets.cov()  * TRADING_DAYS
    return mean_annual, cov_annual

# 5) Portfolio simulation (long-only, weights >= 0, sum to 1)
def simulate_portfolios(mean_ret: pd.Series,
                        cov_mat: pd.DataFrame,
                        n_sims: int = N_PORTFOLIOS,
                        rf: float = RF) -> pd.DataFrame:
    tickers = list(mean_ret.index)
    n = len(tickers)

    all_rets = np.zeros(n_sims)
    all_vols = np.zeros(n_sims)
    all_shrp = np.zeros(n_sims)
    all_wts  = np.zeros((n_sims, n))

    chol = None  # (optional) could pre-factorize for speed with big n

    for i in range(n_sims):
        # random long-only weights that sum to 1
        w = np.random.random(n)
        w = w / w.sum()

        mu_p  = float(np.dot(w, mean_ret.values))               # expected return
        var_p = float(np.dot(w.T, np.dot(cov_mat.values, w)))   # variance
        vol_p = var_p ** 0.5

        # avoid division by zero if vol is extremely tiny
        shrp = (mu_p - rf) / vol_p if vol_p > 0 else -np.inf

        all_rets[i] = mu_p
        all_vols[i] = vol_p
        all_shrp[i] = shrp
        all_wts[i]  = w

    df = pd.DataFrame({
        "ret": all_rets,
        "vol": all_vols,
        "sharpe": all_shrp
    })
    # append weight columns
    for j, t in enumerate(tickers):
        df[f"w_{t}"] = all_wts[:, j]
    return df

# 6) Identify optimal portfolios
def get_optimal_indices(df: pd.DataFrame) -> Tuple[int, int]:
    idx_max = int(df["sharpe"].idxmax())
    idx_min = int(df["vol"].idxmin())
    return idx_max, idx_min

def extract_weights_row(df: pd.DataFrame, idx: int) -> pd.Series:
    return df.loc[idx, [c for c in df.columns if c.startswith("w_")]]

def weights_series_to_table(w_series: pd.Series) -> pd.DataFrame:
    rows = []
    for col, val in w_series.items():
        rows.append({"Ticker": col.replace("w_", ""), "Weight": float(val)})
    tab = pd.DataFrame(rows)
    tab = tab.sort_values("Weight", ascending=False).reset_index(drop=True)
    return tab

# 7) Plot efficient frontier and highlight optimal points
def plot_frontier(df: pd.DataFrame, idx_max: int, idx_min: int, save_path: str = "efficient_frontier.png"):
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(df["vol"], df["ret"], c=df["sharpe"], alpha=0.6)
    plt.colorbar(sc, label="Sharpe Ratio")

    # Max Sharpe
    plt.scatter(df.loc[idx_max, "vol"], df.loc[idx_max, "ret"], marker="*", s=250)
    plt.annotate("Max Sharpe",
                 (df.loc[idx_max, "vol"], df.loc[idx_max, "ret"]),
                 xytext=(10, 10), textcoords="offset points")

    # Min Vol
    plt.scatter(df.loc[idx_min, "vol"], df.loc[idx_min, "ret"], marker="X", s=150)
    plt.annotate("Min Vol",
                 (df.loc[idx_min, "vol"], df.loc[idx_min, "ret"]),
                 xytext=(10, -12), textcoords="offset points")

    plt.xlabel("Volatility (σ)")
    plt.ylabel("Expected Return (μ)")
    plt.title("Efficient Frontier (Simulated)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# 8) Save artifacts for the repo
def save_artifacts(tickers: List[str],
                   df: pd.DataFrame,
                   idx_max: int,
                   idx_min: int,
                   params_path: str = "params.json"):
    w_max = extract_weights_row(df, idx_max)
    w_min = extract_weights_row(df, idx_min)

    tab_max = weights_series_to_table(w_max)
    tab_min = weights_series_to_table(w_min)

    tab_max.to_csv("weights_max_sharpe.csv", index=False)
    tab_min.to_csv("weights_min_vol.csv", index=False)

    params = {
        "tickers": tickers,
        "start": START,
        "end": END,
        "risk_free": RF,
        "n_portfolios": N_PORTFOLIOS,
        "seed": SEED,
        "timestamp": int(time.time())
    }
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    # small console summary
    print("\n== Summary ==")
    print("Max Sharpe:", {k: round(float(v), 4) for k, v in {
        "ret": df.loc[idx_max, "ret"],
        "vol": df.loc[idx_max, "vol"],
        "sharpe": df.loc[idx_max, "sharpe"]
    }.items()})
    print("Top weights (Max Sharpe):")
    print(tab_max.head(10).to_string(index=False))

    print("\nMin Vol:", {k: round(float(v), 4) for k, v in {
        "ret": df.loc[idx_min, "ret"],
        "vol": df.loc[idx_min, "vol"],
        "sharpe": df.loc[idx_min, "sharpe"]
    }.items()})
    print("Top weights (Min Vol):")
    print(tab_min.head(10).to_string(index=False))

def prompt_tickers() -> list[str]:
    """
    Ask the user whether to use defaults or custom tickers.
    - Enter/blank  -> use Mag7
    - 'Mag7'/'Tech5'/'ETF6' -> use that preset
    - 'Custom' -> prompt for tickers like: AAPL, AMD, IBM
    """
    ans = input("Preset [Enter=Mag7, or type Mag7/Tech5/ETF6/Custom]: ").strip()

    # ENTER or explicit Mag7 -> default
    if ans == "" or ans.lower() == "mag7":
        print("Using preset: Mag7")
        return PRESETS["Mag7"].copy()

    # allow other presets by name
    if ans in PRESETS:
        print(f"Using preset: {ans}")
        return PRESETS[ans].copy()

    # custom flow
    if ans.lower() in ("custom", "c"):
        raw = input("Enter tickers (comma/space separated), e.g. 'AAPL, AMD, IBM': ").strip()
        tickers = parse_custom_tickers(raw)
        if not tickers:
            print("No valid tickers detected — falling back to Mag7.")
            return PRESETS["Mag7"].copy()
        if len(tickers) > MAX_TICKERS:
            print(f"[WARN] Provided {len(tickers)} tickers; capping to {MAX_TICKERS}.")
            tickers = tickers[:MAX_TICKERS]
        print(f"Using custom tickers ({len(tickers)}): {tickers}")
        return tickers

    # fallback
    print("Unrecognized input — using Mag7.")
    return PRESETS["Mag7"].copy()

# 9) Main entry point
def main():
    tickers = prompt_tickers()
    print(f"Date range: {START} → {END}")

    prices = fetch_prices(tickers, START, END)
    mean_ann, cov_ann = compute_stats(prices)

    df = simulate_portfolios(mean_ann, cov_ann, n_sims=N_PORTFOLIOS, rf=RF)
    idx_max, idx_min = get_optimal_indices(df)

    plot_frontier(df, idx_max, idx_min, save_path="efficient_frontier.png")
    save_artifacts(tickers, df, idx_max, idx_min)   

    # If you prefer terminal prompt, uncomment:
    # preset_name = input("Preset [Mag7/Tech5/ETF6/Custom]: ").strip() or "Mag7"
    # custom_raw  = input("Custom tickers (comma/space separated, only if preset=Custom): ").strip() if preset_name == "Custom" else ""


if __name__ == "__main__":
    main()