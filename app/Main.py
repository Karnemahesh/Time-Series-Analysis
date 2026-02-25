import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import warnings
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Pro Time Series Forecasting", layout="wide")
st.title("🚀 Intelligent Time Series Forecasting")

# -------------------------------------------------
# NUMERIC CLEANER
# -------------------------------------------------
def clean_column(series):
    def extract_numeric(x):
        if pd.isna(x):
            return np.nan
        x = re.sub(r"[^\d\.\-]", "", str(x))
        try:
            return float(x)
        except:
            return np.nan
    return series.apply(extract_numeric)

# -------------------------------------------------
# FILE LOADER
# -------------------------------------------------
def load_file(uploaded_file):

    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    elif uploaded_file.name.endswith((".xlsx", ".xls")):
        excel_data = pd.ExcelFile(uploaded_file)
        sheet = st.selectbox("Select Excel Sheet", excel_data.sheet_names)
        return pd.read_excel(uploaded_file, sheet_name=sheet)

    elif uploaded_file.name.endswith(".json"):
        raw = pd.read_json(uploaded_file)
        if isinstance(raw.iloc[0], dict):
            return pd.json_normalize(raw)
        return raw

    else:
        st.error("Unsupported file format.")
        st.stop()

# -------------------------------------------------
# DATE DETECTION
# -------------------------------------------------
def detect_date_column(df):
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > len(df) * 0.6:
                return col
        except:
            continue
    return None

# -------------------------------------------------
# SAFE DATETIME PREPARATION
# -------------------------------------------------
def prepare_datetime(df, date_col):

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)

    df = df.set_index(date_col)

    # 🔥 Fix duplicate dates (your previous error)
    if df.index.has_duplicates:
        st.warning("Duplicate timestamps detected. Aggregating by mean.")
        df = df.groupby(df.index).mean()

    df = df.sort_index()

    # Detect frequency
    freq = pd.infer_freq(df.index)

    if not freq:
        diffs = df.index.to_series().diff().dropna()
        if not diffs.empty:
            median_diff = diffs.median()

            if median_diff.days == 1:
                freq = "D"
            elif 28 <= median_diff.days <= 31:
                freq = "M"
            elif 89 <= median_diff.days <= 92:
                freq = "Q"
            elif 364 <= median_diff.days <= 366:
                freq = "Y"
            else:
                freq = "D"
        else:
            freq = "D"

    df = df.asfreq(freq)
    df = df.ffill()

    return df, freq

# -------------------------------------------------
# SEASONALITY DETECTION
# -------------------------------------------------
def detect_seasonality(freq):
    if freq == "D":
        return 7
    elif freq == "M":
        return 12
    elif freq == "Q":
        return 4
    elif freq == "Y":
        return 1
    else:
        return 1

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload dataset (CSV, Excel, JSON)",
    type=["csv", "xlsx", "xls", "json"]
)

if uploaded_file:

    df = load_file(uploaded_file)

    st.subheader("Sample Data")
    st.dataframe(df.head())

    date_col = detect_date_column(df)

    if not date_col:
        st.error("No valid date column detected.")
        st.stop()

    df, freq = prepare_datetime(df, date_col)

    st.success(f"Frequency detected: {freq}")

    # Clean numeric columns
    numeric_cols = []
    for col in df.columns:
        cleaned = clean_column(df[col])
        if cleaned.notna().sum() > 0:
            df[col] = cleaned
            numeric_cols.append(col)

    if not numeric_cols:
        st.error("No numeric columns found.")
        st.stop()

    target_col = st.selectbox("Select Target Column", numeric_cols)

    df = df.dropna(subset=[target_col])

    # -------------------------------------------------
    # TRAIN / TEST SPLIT
    # -------------------------------------------------
    train_size = int(len(df) * 0.8)
    train = df[target_col][:train_size]
    test = df[target_col][train_size:]

    seasonal_period = detect_seasonality(freq)

    # -------------------------------------------------
    # AUTO ARIMA MODEL
    # -------------------------------------------------
    st.subheader("Training Auto ARIMA Model...")

    model = auto_arima(
        train,
        seasonal=True,
        m=seasonal_period,
        stepwise=True,
        suppress_warnings=True
    )

    forecast, conf_int = model.predict(
        n_periods=len(test),
        return_conf_int=True
    )

    forecast_index = test.index

    # -------------------------------------------------
    # METRICS
    # -------------------------------------------------
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    mape = np.mean(np.abs((test - forecast) / test)) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", round(rmse, 2))
    col2.metric("MAE", round(mae, 2))
    col3.metric("MAPE (%)", round(mape, 2))

    # -------------------------------------------------
    # PLOT WITH CONFIDENCE INTERVAL
    # -------------------------------------------------
    fig, ax = plt.subplots()

    train.plot(ax=ax, label="Train")
    test.plot(ax=ax, label="Actual")
    ax.plot(forecast_index, forecast, label="Forecast")

    ax.fill_between(
        forecast_index,
        conf_int[:, 0],
        conf_int[:, 1],
        color="pink",
        alpha=0.3,
        label="Confidence Interval"
    )

    ax.legend()
    ax.set_title("Forecast vs Actual")
    st.pyplot(fig)

    # -------------------------------------------------
    # FUTURE FORECAST
    # -------------------------------------------------
    st.subheader("Future Forecast")

    steps = st.slider("Forecast Steps Ahead", 10, 200, 30)

    future_forecast = model.predict(n_periods=steps)

    future_index = pd.date_range(
        start=df.index[-1],
        periods=steps + 1,
        freq=freq
    )[1:]

    future_df = pd.DataFrame(
        {"Forecast": future_forecast},
        index=future_index
    )

    st.line_chart(pd.concat([df[target_col], future_df]))

    st.download_button(
        "Download Future Forecast CSV",
        future_df.to_csv(),
        file_name="future_forecast.csv",
        mime="text/csv"
    )
