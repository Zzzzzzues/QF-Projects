# 📉 Project 5 – Portfolio Value-at-Risk (VaR) & Expected Shortfall (ES)

A beginner–intermediate quantitative finance project applying Monte Carlo simulation to model 1-day market risk for a basket of stocks.

## 🔍 What it does
- Downloads historical **adjusted Close prices** using `yfinance`
- Estimates return drift (μ) & volatility (σ) from **daily log returns**
- Simulates **10,000 long-only portfolio return scenarios**
- Computes:
  - **VaR at 95%** (downside cutoff)
  - **Expected Shortfall at 95%** (average tail loss)
  - Full return range & mean outcomes
- Saves an interpretable **loss distribution chart** with VaR/ES markers

## 📊 Example Universe
Default preset: **Magnificent 7 (AAPL · MSFT · NVDA · AMZN · META · GOOGL · TSLA)**  
Users may optionally modify the notebook locally to test their own baskets (typically up to 12 tickers).

## 🛠 Tech Stack
Python · Pandas · NumPy · Matplotlib · Monte Carlo Simulation · Probability · Quant Finance

## ✅ Key Output Interpretation
- **VaR 95%: 3.21%** → Daily losses are within 3.21% in 95% of scenarios
- **ES 95%: 4.02%** → When losses exceed VaR, the average loss is 4.02%
- **Mean daily drift: 0.15%**
- **Max 1-day swing: ~–7.87% to +8.87%**

## 💡 What I Learned
- Why we annualise parameters for comparability
- Portfolio returns follow **lognormal behavior**, not symmetric normals
- VaR and ES communicate **tail risk severity**
- Reproducibility: seeded randomness & structured workflow

---

Next steps for improvement: MLE calibration · GARCH or OU volatility models · OOS stability testing  