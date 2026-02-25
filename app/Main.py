import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Time Series Forecasting", layout="wide")
st.title("🚀 Advanced Time Series Forecasting")

# -------------------------------------------------
# NUMERIC CLEANER
# -------------------------------------------------
def clean_column(series):
    def extract_numeric(x):
        if pd.isna(x):
            return np.nan
        x = re.sub(r'[^\d\.\-]', '', str(x))
        try:
            return float(x)
        except:
            return np.nan
    return series.apply(extract_numeric)

# -------------------------------------------------
# FILE LOADER (CSV / Excel / JSON)
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

        # Flatten nested JSON automatically
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

    possible = []

    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > len(df) * 0.6:
                possible.append(col)
        except:
            continue

    return possible

def prepare_datetime_index(df, date_col):

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)
    df = df.set_index(date_col)
    df = df.sort_index()

    freq = pd.infer_freq(df.index)

    if freq is None:
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
    df = df.fillna(method="ffill")

    return df, freq

# -------------------------------------------------
# MODEL EVALUATION
# -------------------------------------------------
def evaluate_params(params, df, target, exog_cols):

    try:
        model = SARIMAX(
            df[target],
            exog=df[exog_cols] if exog_cols else None,
            order=(params['p'], params['d'], params['q']),
            seasonal_order=(params['P'], params['D'], params['Q'], params['s']),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        results = model.fit(disp=False)
        return results.aic

    except:
        return np.inf

def fit_best_model(params, df, target, exog_cols):

    model = SARIMAX(
        df[target],
        exog=df[exog_cols] if exog_cols else None,
        order=(params['p'], params['d'], params['q']),
        seasonal_order=(params['P'], params['D'], params['Q'], params['s']),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    return model.fit(disp=False)

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload dataset (CSV, Excel, JSON)",
    type=["csv", "xlsx", "xls", "json"]
)

if uploaded_file:

    try:
        df = load_file(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    st.subheader("Sample Data")
    st.dataframe(df.head())

    # Date column detection
    date_cols = detect_date_column(df)

    if not date_cols:
        st.error("No date column detected.")
        st.stop()

    selected_date = st.selectbox("Select Date Column", date_cols)
    df, freq = prepare_datetime_index(df, selected_date)

    st.success(f"Frequency detected: {freq}")

    # Clean numeric columns
    cleaned_cols = {}
    for col in df.columns:
        cleaned = clean_column(df[col])
        if cleaned.notna().sum() > 0:
            cleaned_cols[col] = cleaned

    if not cleaned_cols:
        st.error("No numeric columns found.")
        st.stop()

    target_col = st.selectbox("Select Target Variable (Y)", list(cleaned_cols.keys()))
    df[target_col] = cleaned_cols[target_col]
    df = df.dropna(subset=[target_col])

    exog_cols = st.multiselect(
        "Select Exogenous Variables (Optional)",
        [c for c in cleaned_cols if c != target_col]
    )

    for ex in exog_cols:
        df[ex] = cleaned_cols[ex]

    if exog_cols:
        df = df.dropna(subset=exog_cols)

    # Plot
    st.subheader("Cleaned Time Series")
    fig, ax = plt.subplots()
    df[target_col].plot(ax=ax)
    ax.set_title("Observed Data")
    st.pyplot(fig)

    # -------------------------------------------------
    # GRID SEARCH
    # -------------------------------------------------
    st.subheader("Model Training")

    grid = [
        {'p': p, 'd': d, 'q': q, 'P': P, 'D': D, 'Q': Q, 's': s}
        for p in [0, 1]
        for d in [0, 1]
        for q in [0, 1]
        for P in [0, 1]
        for D in [0, 1]
        for Q in [0, 1]
        for s in [0, 12]
    ]

    progress = st.progress(0)
    scores = []

    for i, params in enumerate(grid):
        score = evaluate_params(params, df, target_col, exog_cols)
        scores.append(score)
        progress.progress((i + 1) / len(grid))

    best_index = np.argmin(scores)
    best_params = grid[best_index]

    st.success(f"Best Parameters: {best_params}")
    st.write(f"Best AIC: {scores[best_index]}")

    results = fit_best_model(best_params, df, target_col, exog_cols)

    with st.expander("Model Summary"):
        st.text(results.summary())

    # -------------------------------------------------
    # FORECAST
    # -------------------------------------------------
    st.subheader("Forecast")

    steps = st.slider("Forecast Steps", 10, 200, 30)

    if exog_cols:
        exog_forecast = np.tile(
            df[exog_cols].iloc[-1].values,
            (steps, 1)
        )
    else:
        exog_forecast = None

    forecast = results.forecast(steps=steps, exog=exog_forecast)

    forecast_index = pd.date_range(
        start=df.index[-1],
        periods=steps + 1,
        freq=freq
    )[1:]

    forecast.index = forecast_index

    fig2, ax2 = plt.subplots()
    df[target_col].plot(ax=ax2, label="Observed")
    forecast.plot(ax=ax2, label="Forecast")
    ax2.legend()
    ax2.set_title("Forecast vs Observed")
    st.pyplot(fig2)

    forecast_df = pd.DataFrame({
        f"Forecast_{target_col}": forecast
    })

    forecast_df.index.name = "Forecast_Date"

    st.download_button(
        "Download Forecast CSV",
        forecast_df.to_csv(),
        file_name="forecast.csv",
        mime="text/csv"
    )
