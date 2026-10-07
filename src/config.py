from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "city_day.csv"
OUTPUT_DIR = ROOT / "outputs"

MIN_DAYS = 500          # cities with fewer AQI days are ignored
TEST_DAYS = 60          # hold-out period for evaluation
HORIZON = 30            # days to forecast into the future
WINDOW = 30             # LSTM look-back window (days)
EPOCHS = 60
SEED = 42

# CPCB AQI categories
AQI_BUCKETS = [
    (50, "Good"),
    (100, "Satisfactory"),
    (200, "Moderate"),
    (300, "Poor"),
    (400, "Very Poor"),
    (float("inf"), "Severe"),
]


def aqi_bucket(value: float) -> str:
    for upper, name in AQI_BUCKETS:
        if value <= upper:
            return name
    return "Severe"
