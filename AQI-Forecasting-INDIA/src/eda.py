import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


def adf_test(s: pd.Series) -> dict:
    stat, p, *_ = adfuller(s.values, autolag="AIC")
    return {"adf_statistic": round(float(stat), 3), "p_value": round(float(p), 4),
            "stationary": bool(p < 0.05)}


def decompose(s: pd.Series, period: int = 365):
    if len(s) < 2 * period:
        period = 7
    return seasonal_decompose(s, model="additive", period=period)


def save_eda_plots(s: pd.Series, city: str, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(s.index, s.values, lw=0.8)
    ax.plot(s.rolling(30).mean(), color="red", lw=1.5, label="30-day average")
    ax.set_title(f"{city} - Daily AQI"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "eda_timeseries.png", dpi=120); plt.close(fig)

    monthly = s.groupby(s.index.month).mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(monthly.index, monthly.values)
    ax.set_xlabel("Month"); ax.set_ylabel("Mean AQI")
    ax.set_title(f"{city} - Seasonality by month")
    fig.tight_layout(); fig.savefig(out_dir / "eda_monthly.png", dpi=120); plt.close(fig)

    res = decompose(s)
    fig = res.plot(); fig.set_size_inches(12, 8)
    fig.tight_layout(); fig.savefig(out_dir / "eda_decomposition.png", dpi=120); plt.close(fig)
