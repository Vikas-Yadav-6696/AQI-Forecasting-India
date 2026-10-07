import pandas as pd
from .config import DATA_PATH, MIN_DAYS


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download city_day.csv from Kaggle (see README) "
            "or run:  python -m src.make_sample_data"
        )
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values(["City", "Date"]).reset_index(drop=True)
    return df


def get_city_series(df: pd.DataFrame, city: str) -> pd.Series:
    """Clean, continuous daily AQI series for one city."""
    s = (
        df.loc[df["City"] == city, ["Date", "AQI"]]
        .drop_duplicates("Date")
        .set_index("Date")["AQI"]
        .astype(float)
    )
    s = s[s > 0]                       # remove invalid values
    s = s.asfreq("D")                  # make the index continuous
    s = s.interpolate(method="time", limit=14, limit_direction="both")
    s = s.dropna()
    s.name = "AQI"
    return s


def list_cities(df: pd.DataFrame, min_days: int = MIN_DAYS) -> list:
    counts = df.dropna(subset=["AQI"]).groupby("City")["AQI"].count()
    return sorted(counts[counts >= min_days].index.tolist())


def city_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.dropna(subset=["AQI"]).groupby("City")["AQI"]
    out = g.agg(["count", "mean", "max"]).round(1)
    out.columns = ["days", "mean_AQI", "max_AQI"]
    return out.sort_values("mean_AQI", ascending=False)
