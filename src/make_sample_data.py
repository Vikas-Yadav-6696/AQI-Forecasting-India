"""Creates a synthetic city_day.csv (same schema as the Kaggle file) so the
project can be tested without downloading anything.
Run:  python -m src.make_sample_data
"""
import numpy as np
import pandas as pd
from .config import DATA_PATH, aqi_bucket

CITIES = {  # base AQI, winter amplitude, noise
    "Delhi": (200, 130, 35),
    "Lucknow": (170, 110, 30),
    "Kolkata": (130, 70, 22),
    "Mumbai": (95, 40, 15),
    "Chennai": (80, 25, 12),
    "Bengaluru": (75, 20, 12),
}


def main():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2015-01-01", "2020-07-01", freq="D")
    doy = dates.dayofyear.values
    rows = []
    for city, (base, amp, noise) in CITIES.items():
        # peak around mid-November/December (winter smog), low in monsoon
        seasonal = amp * np.cos(2 * np.pi * (doy - 330) / 365.25)
        diwali = 40 * np.exp(-(((doy - 305) % 365)) / 6.0) * (base > 100)
        trend = np.linspace(0, -base * 0.08, len(dates))
        ar = np.zeros(len(dates))
        eps = rng.normal(0, noise, len(dates))
        for i in range(1, len(dates)):
            ar[i] = 0.7 * ar[i - 1] + eps[i]
        aqi = np.clip(base + seasonal + diwali + trend + ar, 15, 500)
        df = pd.DataFrame({"City": city, "Date": dates, "AQI": aqi})
        df["PM2.5"] = aqi * 0.5 + rng.normal(0, 5, len(df))
        df["PM10"] = aqi * 0.9 + rng.normal(0, 8, len(df))
        df["NO2"] = 25 + aqi * 0.1 + rng.normal(0, 4, len(df))
        mask = rng.random(len(df)) < 0.06   # random missing values, like real data
        df.loc[mask, ["AQI", "PM2.5", "PM10"]] = np.nan
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["AQI_Bucket"] = out["AQI"].apply(lambda v: aqi_bucket(v) if pd.notna(v) else np.nan)
    DATA_PATH.parent.mkdir(exist_ok=True)
    out.to_csv(DATA_PATH, index=False)
    print(f"Sample data written to {DATA_PATH}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
