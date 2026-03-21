# ES Futures Quantitative Research Platform

Interactive dashboard for ES Futures (E-mini S&P 500) analysis built with **Dash** and **Plotly**.

## Features

- **Complete CME Sessions** — full 18:00 → 17:00 next-day sessions
- **Statistical Levels** — mean, σ bands, percentiles overlaid on price charts
- **Distribution Fitting** — Normal, Student-t, KDE with AIC/BIC comparison
- **Fat Tail Analysis** — Hill estimator, VaR, Expected Shortfall
- **Volatility Regimes** — rolling volatility regime detection
- **HMM Regimes** — Hidden Markov Model state detection (optional)
- **Extreme Event Detection** — z-score based alert system

## Quick Start (Local)

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:8050` in your browser.

## Deploy to Vercel

1. Push this repo to GitHub
2. Connect the repo in [Vercel](https://vercel.com)
3. Deploy — Vercel auto-detects the `vercel.json` config
4. Your dashboard will be live at `your-project.vercel.app`

## Tech Stack

- Python 3.12
- Dash + Plotly (interactive charts)
- yfinance (market data)
- SciPy (statistics & distribution fitting)
- hmmlearn (optional — HMM regime detection)
