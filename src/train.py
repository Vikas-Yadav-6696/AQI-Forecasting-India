"""Train + evaluate Naive / Prophet / LSTM for one city.

Usage:
    python -m src.train --city Delhi
    python -m src.train --city Delhi --skip-lstm
"""
import argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import OUTPUT_DIR, TEST_DAYS, HORIZON, EPOCHS, aqi_bucket
from .data_prep import load_raw, get_city_series
from . import eda, models_prophet


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "MAPE_%": round(float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100), 2),
        "R2": round(float(r2_score(y_true, y_pred)), 3),
    }


def run(city, test_days=TEST_DAYS, horizon=HORIZON, epochs=EPOCHS,
        use_lstm=True, log=print):
    out = OUTPUT_DIR / city
    out.mkdir(parents=True, exist_ok=True)

    log(f"[1/5] Loading data for {city}")
    s = get_city_series(load_raw(), city)
    train, test = s.iloc[:-test_days], s.iloc[-test_days:]
    log(f"      {len(s)} days ({s.index[0].date()} -> {s.index[-1].date()})")

    log("[2/5] EDA plots + stationarity test")
    eda.save_eda_plots(s, city, out)
    adf = eda.adf_test(s)

    res = pd.DataFrame({"actual": test})
    res["naive"] = train.iloc[-1]
    log("[3/5] Prophet (evaluation on hold-out)")
    _, p_pred = models_prophet.fit_forecast(train, test_days)
    res["prophet"] = p_pred.reindex(res.index).values

    if use_lstm:
        from .models_lstm import LSTMForecaster
        log("[4/5] LSTM (evaluation on hold-out)")
        lstm = LSTMForecaster(epochs=epochs).fit(train)
        res["lstm"] = lstm.forecast(train, test_days).reindex(res.index).values
        res["lstm_1step"] = lstm.one_step(s, test_days).reindex(res.index).values
    else:
        log("[4/5] LSTM skipped")

    scores = {c: metrics(res["actual"], res[c]) for c in res.columns if c != "actual"}
    res.to_csv(out / "test_forecast.csv", index_label="date")

    log("[5/5] Retraining on full data -> future forecast")
    _, p_future = models_prophet.fit_forecast(s, horizon)
    fut = pd.DataFrame({"prophet": p_future})
    if use_lstm:
        from .models_lstm import LSTMForecaster
        final = LSTMForecaster(epochs=epochs).fit(s)
        fut["lstm"] = final.forecast(s, horizon).values
    fut["category_prophet"] = fut["prophet"].apply(aqi_bucket)
    fut.to_csv(out / "future_forecast.csv", index_label="date")

    fig, ax = plt.subplots(figsize=(12, 5))
    hist = s.iloc[-(test_days + 120):]
    ax.plot(hist.index, hist.values, color="gray", lw=1, label="history")
    ax.plot(res.index, res["actual"], color="black", lw=1.5, label="actual (test)")
    for col, style in [("naive", ":"), ("prophet", "-"), ("lstm", "-")]:
        if col in res:
            ax.plot(res.index, res[col], style, label=col)
    ax.plot(fut.index, fut["prophet"], "--", color="tab:blue", label="prophet future")
    if "lstm" in fut:
        ax.plot(fut.index, fut["lstm"], "--", color="tab:orange", label="lstm future")
    ax.set_title(f"{city} - AQI forecast"); ax.legend(ncol=3)
    fig.tight_layout(); fig.savefig(out / "forecast_comparison.png", dpi=120); plt.close(fig)

    summary = {"city": city, "days": len(s), "test_days": test_days,
               "horizon": horizon, "adf": adf, "metrics": scores}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    log("Done. Results saved in " + str(out))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="Delhi")
    ap.add_argument("--test-days", type=int, default=TEST_DAYS)
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--skip-lstm", action="store_true")
    a = ap.parse_args()
    r = run(a.city, a.test_days, a.horizon, a.epochs, not a.skip_lstm)
    print("\nMetrics on hold-out set:")
    print(pd.DataFrame(r["metrics"]).T.to_string())
