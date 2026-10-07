import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import OUTPUT_DIR, TEST_DAYS, HORIZON, EPOCHS, aqi_bucket
from src.data_prep import load_raw, get_city_series, list_cities, city_summary

st.set_page_config(page_title="India AQI Forecasting", page_icon="🌫️", layout="wide")
st.title("🌫️ Air Quality Forecasting for Indian Cities")
st.caption("Time-series analysis with Prophet and LSTM on CPCB city-level daily AQI data")


@st.cache_data
def get_data():
    return load_raw()


try:
    df = get_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

cities = list_cities(df)
city = st.sidebar.selectbox("City", cities, index=cities.index("Delhi") if "Delhi" in cities else 0)
s = get_city_series(df, city)
st.sidebar.write(f"**{len(s)}** days of data\n\n{s.index[0].date()} → {s.index[-1].date()}")

tab1, tab2, tab3 = st.tabs(["📊 Explore", "🔮 Forecast", "🏙️ Compare cities"])

# ---------------- Explore ----------------
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest AQI", f"{s.iloc[-1]:.0f}", aqi_bucket(s.iloc[-1]))
    c2.metric("Average AQI", f"{s.mean():.0f}")
    c3.metric("Worst day", f"{s.max():.0f}", str(s.idxmax().date()))

    fig = go.Figure()
    fig.add_scatter(x=s.index, y=s.values, name="Daily AQI", line=dict(width=1))
    fig.add_scatter(x=s.index, y=s.rolling(30).mean(), name="30-day avg", line=dict(color="red"))
    fig.update_layout(title=f"{city} - Daily AQI", height=380)
    st.plotly_chart(fig, use_container_width=True)

    a, b = st.columns(2)
    monthly = s.groupby(s.index.month).mean().rename_axis("Month").reset_index(name="Mean AQI")
    a.plotly_chart(px.bar(monthly, x="Month", y="Mean AQI", title="Seasonality: average AQI by month"),
                   use_container_width=True)
    yearly = s.groupby(s.index.year).mean().rename_axis("Year").reset_index(name="Mean AQI")
    b.plotly_chart(px.line(yearly, x="Year", y="Mean AQI", markers=True, title="Yearly trend"),
                   use_container_width=True)

# ---------------- Forecast ----------------
with tab2:
    out = OUTPUT_DIR / city
    st.subheader("Train / re-train models")
    col1, col2, col3, col4 = st.columns(4)
    horizon = col1.slider("Forecast days", 7, 90, HORIZON)
    test_days = col2.slider("Test days", 30, 120, TEST_DAYS)
    epochs = col3.slider("LSTM epochs", 5, 100, EPOCHS)
    use_lstm = col4.checkbox("Include LSTM", True)

    if st.button("🚀 Run training", type="primary"):
        from src import train
        box = st.empty()
        logs = []

        def log(msg):
            logs.append(msg)
            box.code("\n".join(logs))

        with st.spinner("Training... LSTM may take a few minutes"):
            train.run(city, test_days, horizon, epochs, use_lstm, log=log)
        st.success("Done!")

    if (out / "metrics.json").exists():
        m = json.loads((out / "metrics.json").read_text())
        st.subheader("Hold-out evaluation")
        st.dataframe(pd.DataFrame(m["metrics"]).T, use_container_width=True)
        st.caption(
            "naive = repeat last value | prophet & lstm = multi-step forecast over the whole test window | "
            "lstm_1step = next-day prediction using real past values (easier task)"
        )
        adf = m["adf"]
        st.write(f"ADF stationarity test: p = **{adf['p_value']}** → "
                 f"{'stationary' if adf['stationary'] else 'non-stationary'}")

        test = pd.read_csv(out / "test_forecast.csv", index_col="date", parse_dates=True)
        fut = pd.read_csv(out / "future_forecast.csv", index_col="date", parse_dates=True)

        fig = go.Figure()
        hist = s.iloc[-(len(test) + 120):]
        fig.add_scatter(x=hist.index, y=hist.values, name="history", line=dict(color="gray"))
        fig.add_scatter(x=test.index, y=test["actual"], name="actual", line=dict(color="black"))
        for col in [c for c in ["prophet", "lstm", "naive"] if c in test]:
            fig.add_scatter(x=test.index, y=test[col], name=col)
        for col in [c for c in ["prophet", "lstm"] if c in fut]:
            fig.add_scatter(x=fut.index, y=fut[col], name=f"{col} (future)", line=dict(dash="dash"))
        fig.update_layout(title=f"{city} - forecast comparison", height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Future forecast")
        show = fut.copy().round(1)
        st.dataframe(show, use_container_width=True)
        st.download_button("⬇️ Download forecast CSV", fut.to_csv().encode(), f"{city}_forecast.csv")
    else:
        st.info("No results yet for this city. Click **Run training** above.")

# ---------------- Compare ----------------
with tab3:
    summ = city_summary(df).reset_index()
    st.plotly_chart(px.bar(summ, x="City", y="mean_AQI", color="mean_AQI",
                           color_continuous_scale="RdYlGn_r", title="Average AQI by city"),
                    use_container_width=True)
    st.dataframe(summ, use_container_width=True)
