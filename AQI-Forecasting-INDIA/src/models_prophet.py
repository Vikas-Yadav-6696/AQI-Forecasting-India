import pandas as pd
from prophet import Prophet


def fit_forecast(train: pd.Series, periods: int):
    """Fit Prophet on `train`, forecast `periods` days after its last date.
    Returns (model, Series of predictions indexed by date)."""
    d = train.reset_index()
    d.columns = ["ds", "y"]
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    try:  # Diwali etc. strongly affect Indian AQI
        m.add_country_holidays(country_name="IN")
    except Exception:
        pass
    m.fit(d)
    future = m.make_future_dataframe(periods=periods, freq="D")
    fc = m.predict(future).tail(periods)
    pred = pd.Series(fc["yhat"].clip(lower=0).values,
                     index=pd.DatetimeIndex(fc["ds"]), name="prophet")
    return m, pred
