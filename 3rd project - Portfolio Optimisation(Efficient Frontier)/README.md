Objective:
Apply Modern Portfolio Theory (MPT) to simulate and visualize the Efficient Frontier for the Magnificent 7 stocks — AAPL, MSFT, NVDA, AMZN, META, GOOGL, and TSLA — identifying optimal portfolios that balance risk and return.
Users can also input their own stock lists for custom analysis.

📘 Project Overview:
This project builds a Python-based portfolio optimizer using historical market data.
It simulates 10,000+ random portfolios, calculates their expected returns, volatility, and Sharpe ratios, and visualizes the Efficient Frontier — the curve representing optimal portfolios for a given risk level.

⚙️ Methodology
	1.	Data Collection
	•	Historical daily price data fetched via yfinance
	•	Data period: 2022-01-01 to 2025-01-01
	2.	Calculations
	•	Daily returns → annualized mean returns and covariance matrix
	•	Portfolio simulation with long-only weights (w ≥ 0, Σw = 1)
	•	Compute for each portfolio:
	•	Expected Return:  μ_p = w^T μ
	•	Volatility:  σ_p = \sqrt{w^TΣw}
	•	Sharpe Ratio:  \frac{μ_p - R_f}{σ_p}
	3.	Optimization Goals
	•	Max Sharpe Ratio Portfolio → highest risk-adjusted return
	•	Min Volatility Portfolio → lowest overall risk
	4.	Visualization
	•	Scatterplot of 10,000 simulated portfolios (volatility vs return)
	•	Color scale by Sharpe ratio
	•	Highlighted optimal points (Max Sharpe ⭐ and Min Volatility ✖️)

🧠 Interpretation
	•	NVIDIA (NVDA) dominates high-return regions due to strong performance and high volatility.
	•	Apple (AAPL) and Microsoft (MSFT) anchor the defensive (low-volatility) portfolios.
	•	The Efficient Frontier visually demonstrates how diversification reduces risk while maintaining attractive returns.