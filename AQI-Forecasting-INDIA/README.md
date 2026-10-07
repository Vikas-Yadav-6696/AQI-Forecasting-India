# Air Quality Forecasting for Indian Cities (Prophet vs LSTM)

Forecast daily AQI for Indian cities using CPCB data. Compares three approaches:
**Naive baseline**, **Facebook Prophet** and a **stacked LSTM** (TensorFlow/Keras),
with a Streamlit dashboard on top.

## Structure
```
aqi_forecast/
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── data/city_day.csv       # <- put the Kaggle dataset here
├── outputs/<City>/         # generated plots, forecasts, metrics.json
└── src/
    ├── config.py           # paths + hyper-parameters
    ├── data_prep.py        # loading, cleaning, interpolation
    ├── eda.py              # plots, decomposition, ADF test
    ├── models_prophet.py
    ├── models_lstm.py
    ├── train.py            # train + evaluate + forecast (CLI)
    └── make_sample_data.py # synthetic data for quick testing
```

## Setup (Anaconda Prompt)
```bash
conda create -n aqi python=3.10 -y
conda activate aqi
conda install -c conda-forge prophet -y
pip install pandas numpy matplotlib plotly scikit-learn statsmodels tensorflow streamlit
```

## Get the data
1. Open https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
2. Download and unzip. Copy **city_day.csv** into the `data/` folder.

No Kaggle account handy? Generate test data: `python -m src.make_sample_data`
(synthetic, only to check that everything works).

## Run
```bash
cd aqi_forecast
python -m src.train --city Delhi          # CLI: full pipeline
python -m src.train --city Mumbai --skip-lstm   # faster, Prophet only
streamlit run app.py                      # dashboard (opens in browser)
```

## Method
1. **Cleaning**: drop invalid values, build a continuous daily index, time-interpolate gaps (max 14 days).
2. **EDA**: trend, monthly seasonality, seasonal decomposition, ADF stationarity test.
3. **Split**: last 60 days = test set (no shuffling, time order preserved).
4. **Models**
   - Naive: repeat last observed value.
   - Prophet: yearly + weekly seasonality + Indian holidays (Diwali).
   - LSTM: 30-day window of AQI + sin/cos day-of-year, 2 LSTM layers, early stopping, recursive multi-step forecast.
5. **Metrics**: MAE, RMSE, MAPE, R². 
6. **Final forecast**: models retrained on all data, next 30 days forecast.

## Troubleshooting
| Problem | Fix |
|---|---|
| `No module named prophet` | `conda install -c conda-forge prophet` |
| `FileNotFoundError: city_day.csv` | Put the file in `data/` |
| LSTM is slow | lower `--epochs 20` or use `--skip-lstm` |
| TensorFlow install fails | use Python 3.10/3.11 |
| `streamlit` not found | `pip install streamlit` inside the `aqi` env |

## Ideas to extend
- Multivariate LSTM (PM2.5, PM10, NO2...) and compare with univariate
- Add SARIMA / XGBoost, try weather data
- Forecast several cities in a loop and rank them
- Deploy on Streamlit Community Cloud
