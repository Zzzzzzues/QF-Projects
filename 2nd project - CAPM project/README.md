🧮 CAPM Beta & Alpha Estimation (Python)

📘 Project Overview

This project applies the Capital Asset Pricing Model (CAPM) to estimate how sensitive different stocks are to market movements — measured by Beta (systematic risk) — and how much they outperform or underperform the market — measured by Alpha (excess return).

Using Python, it downloads historical stock and market data via the Yahoo Finance API, computes daily returns, runs linear regressions using statsmodels, and outputs key metrics and plots.

⸻

🎯 Key Objectives
	•	Quantify how individual stocks move relative to the market (Beta).
	•	Measure each stock’s risk-adjusted excess performance (Alpha).
	•	Allow users to input their own stock tickers for analysis.
	•	Visualize the CAPM regression line and save results automatically.

⸻

⚙️ How It Works
	1.	Input your tickers (or use the defaults like AAPL, MSFT, NVDA).
	2.	The script downloads daily prices for your tickers and the S&P 500 (^GSPC).
	3.	It calculates daily returns and runs:
R_i = \alpha + \beta R_m + \varepsilon
	4.	Outputs:
	•	Beta: Market sensitivity
	•	Alpha: Risk-adjusted excess return
	•	R²: How much of the stock’s movement is explained by the market
	•	Regression scatter plots for each stock

    Sample Output
    Ticker,Beta,R²,Alpha (daily),Alpha (annual approx)
     AAPL,1.210689,0.611443,0.000189,0.047683
     MSFT,1.248328,0.627477,7.7e-05,0.01936
     NVDA,2.268515,0.515835,0.001835,0.462472
📊 Interpretation
	•	AAPL: Beta ≈ 1.21 → moves roughly 20 % more than the market. Positive Alpha (~4.8 %/yr) shows slight risk-adjusted outperformance.
	•	MSFT: Beta ≈ 1.25 → moderate volatility, stable correlation (R² ≈ 0.63).
	•	NVDA: Beta ≈ 2.27 → highly volatile; strong positive Alpha (~46 %/yr) suggests major outperformance during this sample period.
Overall, NVDA carried the highest risk and the highest reward.

📘 Technologies Used
	•	Python
	•	pandas, numpy – data manipulation
	•	matplotlib – visualization
	•	statsmodels – linear regression (OLS)
	•	yfinance – Yahoo Finance API for historical data

💡 Key Learnings
	•	Applied CAPM in practice using real data.
	•	Learned how to interpret Alpha, Beta, and R² in financial context.
	•	Strengthened understanding of regression modeling in Python.
	•	Gained experience building user-interactive scripts and automated reporting.