# 🌍 AQI Forecasting India

### Air Quality Index Forecasting for Indian Cities using Prophet & LSTM

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)](https://pandas.pydata.org/)
[![Prophet](https://img.shields.io/badge/Prophet-Time%20Series-6A5ACD)](https://facebook.github.io/prophet/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **An end-to-end time-series forecasting system for analyzing historical Air Quality Index (AQI) patterns across Indian cities and forecasting future AQI using Prophet and LSTM deep-learning models.**

---

## 📌 Project Overview

**AQI Forecasting India** is a Data Science and Artificial Intelligence project focused on the analysis and forecasting of air quality across Indian cities.

The project uses historical air-quality observations to identify temporal patterns, trends, and variations in AQI and applies two different forecasting approaches:

- **Prophet** — a statistical time-series forecasting framework designed to model trend and seasonality.
- **LSTM (Long Short-Term Memory)** — a recurrent neural-network architecture designed to learn sequential and temporal dependencies.

The project combines:

**Data Processing → Exploratory Data Analysis → Time-Series Preparation → Forecasting → Model Evaluation → Model Comparison → Interactive Visualization**

An interactive **Streamlit dashboard** provides a user-friendly interface for exploring historical AQI data and forecast results.

---

## 🎯 Objectives

The major objectives of this project are:

1. Analyze historical AQI data from Indian cities.
2. Perform data cleaning and preprocessing.
3. Identify temporal trends and seasonal AQI patterns.
4. Perform exploratory data analysis and visualization.
5. Prepare data for time-series forecasting.
6. Develop an AQI forecasting model using **Prophet**.
7. Develop an AQI forecasting model using **LSTM**.
8. Evaluate forecasting performance using appropriate metrics.
9. Compare Prophet and LSTM forecasting performance.
10. Present analytical insights through an interactive Streamlit dashboard.
11. Demonstrate an end-to-end Data Science and AI workflow for an environmental application.

---

## 🧠 Problem Statement

Air pollution is a major environmental and public-health concern in India. Air Quality Index values vary significantly across cities and over time due to factors such as seasonal conditions, weather patterns, urbanization, industrial activity, transportation, and other environmental factors.

Historical AQI data contains temporal patterns that can be analyzed to estimate future air-quality conditions.

The problem addressed by this project is:

> **How can historical AQI observations be used to develop reliable time-series forecasting models for estimating future air-quality conditions across Indian cities?**

To investigate this problem, the project compares a statistical forecasting approach (**Prophet**) with a deep-learning sequence model (**LSTM**).

---

# 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │   CPCB AQI Dataset   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Ingestion       │
                    │ & Validation         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Cleaning        │
                    │ & Preprocessing      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Exploratory Data     │
                    │ Analysis (EDA)       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Time-Series          │
                    │ Preparation          │
                    └──────────┬───────────┘
                               │
                     ┌─────────┴─────────┐
                     │                   │
                     ▼                   ▼
             ┌──────────────┐     ┌──────────────┐
             │   Prophet    │     │     LSTM     │
             │    Model     │     │     Model    │
             └──────┬───────┘     └──────┬───────┘
                    │                    │
                    ▼                    ▼
             ┌──────────────┐     ┌──────────────┐
             │   Forecast   │     │   Forecast   │
             │   Results    │     │   Results    │
             └──────┬───────┘     └──────┬───────┘
                    │                    │
                    └─────────┬──────────┘
                              ▼
                    ┌──────────────────────┐
                    │ Model Evaluation &   │
                    │ Performance          │
                    │ Comparison           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Streamlit Dashboard  │
                    └──────────────────────┘
```

---

# 📊 Dataset

The project uses historical air-quality data associated with Indian cities and sourced from **Central Pollution Control Board (CPCB)** data.

Depending on the dataset version used, the data may contain information related to:

- Date / timestamp
- City
- AQI
- PM2.5
- PM10
- NO₂
- SO₂
- CO
- O₃
- Other air-quality attributes

### Dataset Processing

The preprocessing pipeline may include:

- Missing-value handling
- Duplicate removal
- Date conversion
- Data type correction
- Sorting chronological observations
- City-wise filtering
- Aggregation where required
- Outlier investigation
- Time-series preparation
- Feature scaling for LSTM

> **Note:** Dataset statistics should be reported from the exact dataset included with the project rather than being hard-coded in this README.

---

# 🔎 Exploratory Data Analysis

Exploratory Data Analysis is performed to understand the underlying characteristics of the AQI dataset.

The analysis focuses on:

### Temporal Analysis

- Daily AQI trends
- Monthly AQI variation
- Yearly AQI trends
- Seasonal patterns
- Long-term changes

### City-Level Analysis

- AQI comparison between cities
- City-wise distributions
- Highest and lowest AQI observations
- Historical trends by location

### Statistical Analysis

- AQI distribution
- Pollutant distributions
- Correlation analysis
- Missing-value analysis
- Outlier investigation

Example visualizations include:

```text
AQI Trend
Monthly AQI Distribution
Yearly AQI Comparison
City-wise AQI Comparison
Correlation Matrix
Pollutant Distribution
```

---

# ⏳ Time-Series Forecasting

The project treats AQI as a temporal forecasting problem.

Historical observations are ordered chronologically to preserve the temporal structure of the data.

A typical forecasting workflow is:

```text
Historical AQI
      │
      ▼
Chronological Ordering
      │
      ▼
Train / Validation / Test
      │
      ├───────────────┐
      ▼               ▼
   Prophet           LSTM
      │               │
      ▼               ▼
  Forecast          Forecast
      │               │
      └───────┬───────┘
              ▼
       Model Evaluation
```

Unlike ordinary random classification or regression, time-series forecasting requires careful handling of temporal ordering to avoid using future information during model training.

---

# 🤖 Models

## 1. Prophet

**Prophet** is a time-series forecasting framework that models temporal patterns using components such as:

- Trend
- Seasonality
- Historical patterns
- Additional time-dependent effects where applicable

Prophet is particularly useful when the time series contains strong trend and seasonal behavior.

### General Workflow

```text
Historical AQI
      ↓
Date + AQI preparation
      ↓
Prophet model training
      ↓
Future dataframe
      ↓
Forecast
      ↓
Evaluation
```

---

## 2. LSTM

**Long Short-Term Memory (LSTM)** is a recurrent neural-network architecture designed to learn dependencies in sequential data.

LSTM is useful for time-series problems because it can learn relationships between previous observations and future values.

### General Workflow

```text
Historical AQI
      ↓
Normalization / Scaling
      ↓
Sliding Window Sequences
      ↓
LSTM Network
      ↓
Training
      ↓
Prediction
      ↓
Inverse Scaling
      ↓
Forecast Evaluation
```

A simplified sequence can be represented as:

```text
AQI(t-30)
   ↓
AQI(t-29)
   ↓
  ...
   ↓
AQI(t-1)
   ↓
 LSTM
   ↓
AQI(t)
```

---

# 📈 Model Evaluation

The forecasting models are evaluated using quantitative performance metrics.

### Mean Absolute Error — MAE

Measures the average absolute difference between actual and predicted values.

\[
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y_i}|
\]

### Root Mean Squared Error — RMSE

Penalizes larger prediction errors more strongly.

\[
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y_i})^2}
\]

### Mean Absolute Percentage Error — MAPE

Measures prediction error as a percentage.

\[
MAPE = \frac{100}{n}\sum_{i=1}^{n}
\left|\frac{y_i-\hat{y_i}}{y_i}\right|
\]

### R² Score

Measures the proportion of variance explained by the model.

\[
R^2 =
1-\frac{SS_{res}}{SS_{tot}}
\]

---

## 📊 Model Comparison

The final performance comparison should be reported using the actual experimental results.

| Model | MAE | RMSE | MAPE | R² |
|---|---:|---:|---:|---:|
| Prophet | — | — | — | — |
| LSTM | — | — | — | — |

> **Important:** Performance values should be updated after running the final experiments on the project dataset.

The comparison helps determine whether a statistical forecasting model or a deep-learning sequence model performs better for the selected AQI forecasting task.

---

# 📊 Streamlit Dashboard

The project includes an interactive dashboard developed using **Streamlit**.

The dashboard is intended to make the forecasting system easier to explore without requiring users to execute individual notebooks.

### Dashboard Features

- AQI overview
- City selection
- Historical AQI visualization
- Time-series trends
- Forecast visualization
- Prophet results
- LSTM results
- Model comparison
- Forecast horizon selection
- AQI category interpretation
- Interactive charts

### Dashboard Workflow

```text
User
 │
 ▼
Select City
 │
 ▼
Select Model
 │
 ▼
Select Forecast Horizon
 │
 ▼
Generate Forecast
 │
 ▼
Visualize Results
```

---

# 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib, Plotly |
| Statistical Forecasting | Prophet |
| Deep Learning | TensorFlow / Keras |
| Neural Network | LSTM |
| Dashboard | Streamlit |
| Development | Jupyter Notebook / VS Code |
| Version Control | Git & GitHub |

---

# 📁 Recommended Project Structure

```text
AQI-Forecasting-India/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_prophet_forecasting.ipynb
│   ├── 04_lstm_forecasting.ipynb
│   └── 05_model_comparison.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── prophet_model.py
│   ├── lstm_model.py
│   └── evaluation.py
│
├── models/
│
├── results/
│   ├── figures/
│   ├── forecasts/
│   └── metrics/
│
├── dashboard/
│   └── app.py
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

> The exact directory structure may vary according to the implementation contained in the repository.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Vikas-Yadav-6696/AQI-Forecasting-India.git
cd AQI-Forecasting-India
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available in the current project version, install the required libraries according to the notebooks and dashboard implementation.

---

# ▶️ Running the Project

## Run the Jupyter Notebooks

```bash
jupyter notebook
```

Execute the notebooks in the recommended order:

```text
1. Data Cleaning
       ↓
2. Exploratory Data Analysis
       ↓
3. Prophet Forecasting
       ↓
4. LSTM Forecasting
       ↓
5. Model Evaluation & Comparison
```

---

## Launch the Streamlit Dashboard

From the project root:

```bash
streamlit run dashboard/app.py
```

The application will then be available through the local Streamlit server.

---

# 🔬 Methodology

The complete methodology can be summarized as:

### Phase 1 — Data Collection

Collect historical AQI and air-quality observations.

### Phase 2 — Data Preprocessing

- Clean the dataset
- Handle missing values
- Remove duplicates
- Convert date/time fields
- Sort observations chronologically

### Phase 3 — Exploratory Analysis

Analyze:

- trends
- seasonality
- distributions
- city-level differences
- pollutant relationships

### Phase 4 — Time-Series Preparation

Prepare the dataset for forecasting while maintaining chronological order.

### Phase 5 — Prophet Forecasting

Train and evaluate the Prophet forecasting model.

### Phase 6 — LSTM Forecasting

Create sequential windows, scale the data, train the LSTM network, and generate forecasts.

### Phase 7 — Evaluation

Compare models using:

- MAE
- RMSE
- MAPE
- R²

### Phase 8 — Visualization

Present historical patterns and forecast results.

### Phase 9 — Dashboard

Integrate the analytical results into an interactive Streamlit application.

---

# 📌 Key Expected Outcomes

The project aims to:

- Identify historical AQI trends across Indian cities.
- Understand temporal and seasonal variations.
- Develop forecasting models using two different approaches.
- Quantitatively compare Prophet and LSTM.
- Visualize future AQI estimates.
- Provide an interactive analytical interface.
- Demonstrate an end-to-end Data Science and AI workflow.

---

# ⚠️ Limitations

Forecasting air quality is inherently challenging because AQI is influenced by many external factors.

Potential limitations include:

- Weather conditions may not be fully represented.
- Sudden pollution events can be difficult to forecast.
- Dataset quality and missing observations can affect model performance.
- AQI behavior differs considerably between cities.
- Long-horizon forecasts generally carry greater uncertainty.
- Historical patterns may not always represent future environmental conditions.

Therefore, forecasts should be interpreted as **model-based estimates rather than guaranteed future AQI values**.

---

# 🚀 Future Scope

The project can be extended in several directions.

### Advanced Models

Future versions could evaluate:

- GRU
- Bidirectional LSTM
- CNN-LSTM
- Transformer
- Temporal Fusion Transformer
- XGBoost
- LightGBM

### Additional Features

Weather and environmental variables could be incorporated, such as:

- Temperature
- Humidity
- Wind speed
- Atmospheric pressure
- Rainfall
- Visibility

### Advanced Deployment

The system could be extended with:

- Cloud deployment
- Automated data ingestion
- Scheduled model retraining
- Real-time AQI updates
- API-based forecasting
- Model monitoring

### Geographic Analysis

Future versions could include:

- City-level forecasting
- State-level analysis
- Interactive geographical maps
- Regional pollution comparisons

---

# 🎓 Academic Information

**Project:** AQI Forecasting India  
**Project Category:** Big Data / Data Science / Artificial Intelligence  
**Program:** Master of Computer Applications — Data Science & Artificial Intelligence  
**Project Type:** Academic Group Project  
**Group:** AI — Group G2

---

# 👨‍💻 Contributors

### Vikas Yadav

**MCA — Data Science & Artificial Intelligence**

GitHub: [Vikas-Yadav-6696](https://github.com/Vikas-Yadav-6696)

Additional group members can be added here:

```text
1. Vikas Yadav
2. Vartika Gupta
3. Ashwini Kumar Singh
4. Anshuman Yadav
```

---

# 📚 References

- Central Pollution Control Board (CPCB) — Air Quality Data
- Prophet Documentation
- TensorFlow / Keras Documentation
- Streamlit Documentation
- Pandas Documentation
- NumPy Documentation
- Matplotlib / Plotly Documentation

---

# ⚖️ Disclaimer

This project is developed for **educational and research purposes** as part of an MCA Data Science & Artificial Intelligence academic project.

The forecasts generated by the system are model-based estimates and should not be considered official air-quality measurements, medical advice, environmental regulatory guidance, or emergency warnings.

---

# ⭐ Support the Project

If you find this project useful for learning or research, consider giving the repository a ⭐ on GitHub.

---

## 📄 License

This project is intended for academic and educational use.

If a formal open-source license is included in the repository, refer to the `LICENSE` file for the applicable terms.
