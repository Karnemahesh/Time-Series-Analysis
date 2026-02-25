import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import warnings
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Smart Time Series Forecasting", layout="wide")
st.title("🚀 Smart Time Series Forecasting")

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
# PREPARE DATETIME
# -------------------------------------------------
def prepare_datetime(df, date_col):

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)
    df = df.set_index(date_col)

    # Handle duplicates
    if df.index.has_duplicates:
        st.warning("Duplicate timestamps detected. Aggregating safely...")

        numeric_cols = df.select_dtypes(include=np.number).columns
        non_numeric_cols = df.select_dtypes(exclude=np.number).columns

        df_numeric = df[numeric_cols].groupby(df.index).mean()

        if len(non_numeric_cols) > 0:
            df_non_numeric = df[non_numeric_cols].groupby(df.index).first()
            df = pd.concat([df_numeric, df_non_numeric], axis=1)
        else:
            df = df_numeric

    df = df.sort_index()

    freq = pd.infer_freq(df.index)
    if not freq:
        freq = "D"

    df = df.asfreq(freq)
    df = df.ffill()

    return df, freq

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

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[target_col])

    if len(df) == 0:
        st.error("No usable data after cleaning.")
        st.stop()

    st.write("Rows available for modeling:", len(df))

    # -------------------------------------------------
    # TRAIN TEST SPLIT
    # -------------------------------------------------
    train_size = max(1, int(len(df) * 0.8))
    train = df[target_col].iloc[:train_size]
    test = df[target_col].iloc[train_size:]

    # -------------------------------------------------
    # SMART MODEL SELECTION
    # -------------------------------------------------
    st.subheader("Model Training")

    if len(train) < 2:
        st.warning("Very small dataset. Using naive forecast.")
        forecast = np.repeat(train.iloc[-1], len(test))
        conf_int = None

    elif len(df) >= 20:
        st.info("Using Auto ARIMA")

        try:
            model = auto_arima(
                train,
                seasonal=True,
                m=7,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore"
            )

            forecast, conf_int = model.predict(
                n_periods=len(test),
                return_conf_int=True
            )

        except:
            st.warning("ARIMA failed. Switching to Exponential Smoothing.")
            model = ExponentialSmoothing(train, trend="add")
            fit = model.fit()
            forecast = fit.forecast(len(test))
            conf_int = None

    else:
        st.warning("Small dataset. Using Exponential Smoothing.")
        model = ExponentialSmoothing(train, trend="add")
        fit = model.fit()
        forecast = fit.forecast(len(test))
        conf_int = None

    # -------------------------------------------------
    # METRICS
    # -------------------------------------------------
    if len(test) > 0:
        rmse = np.sqrt(mean_squared_error(test, forecast))
        mae = mean_absolute_error(test, forecast)
        mape = np.mean(np.abs((test - forecast) / test)) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", round(rmse, 2))
        col2.metric("MAE", round(mae, 2))
        col3.metric("MAPE (%)", round(mape, 2))

    # -------------------------------------------------
    # PLOT
    # -------------------------------------------------
    fig, ax = plt.subplots()
    train.plot(ax=ax, label="Train")

    if len(test) > 0:
        test.plot(ax=ax, label="Actual")
        ax.plot(test.index, forecast, label="Forecast")

        if conf_int is not None:
            ax.fill_between(
                test.index,
                conf_int[:, 0],
                conf_int[:, 1],
                alpha=0.3
            )

    ax.legend()
    ax.set_title("Forecast vs Actual")
    st.pyplot(fig)

    # -------------------------------------------------
    # FUTURE FORECAST
    # -------------------------------------------------
    st.subheader("Future Forecast")

    steps = st.slider("Forecast Steps Ahead", 1, 200, 30)

    if len(train) < 2:
        future_forecast = np.repeat(train.iloc[-1], steps)

    elif len(df) >= 20:
        future_forecast = model.predict(n_periods=steps)

    else:
        future_forecast = fit.forecast(steps)

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
