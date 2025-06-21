import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import warnings
from joblib import Parallel, delayed
from model_utils import evaluate_params, fit_best_model

# Streamlit page config
st.set_page_config(page_title="Time Series Forecasting", layout="wide")
st.title("🚀 Time Series Forecasting (ARIMA / SARIMAX)")

# Universal numeric cleaner
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

# Upload file
uploaded_file = st.file_uploader("Upload dataset (CSV, Excel, JSON)", type=["csv", "xlsx", "xls", "json"])

if uploaded_file:
    # Load dataset
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".json"):
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file format.")
            st.stop()
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.write("Sample Data:")
    st.dataframe(df.head())

    # Auto-detect date columns
    possible_date_cols = []
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors='coerce')
            if parsed.notna().sum() > 0:
                possible_date_cols.append(col)
        except:
            continue

    # Date selection
    selected_date_col = None
    if possible_date_cols:
        selected_date_col = st.selectbox("Optional: Select Date Column", possible_date_cols)
        df[selected_date_col] = pd.to_datetime(df[selected_date_col], errors='coerce')
        df = df.dropna(subset=[selected_date_col])
        df = df.sort_values(by=selected_date_col)
        df = df.set_index(selected_date_col)

        # Infer frequency
        try:
            freq = pd.infer_freq(df.index)
            if freq:
                df = df.asfreq(freq)
                st.info(f"Frequency inferred and set: {freq}")
        except:
            pass

    # Clean all columns for numeric values
    cleaned_cols = {}
    for col in df.columns:
        cleaned = clean_column(df[col])
        if cleaned.notna().sum() > 0:
            cleaned_cols[col] = cleaned

    if not cleaned_cols:
        st.error("No numeric columns found after cleaning.")
        st.stop()

    # Select target column
    target_col = st.selectbox("Select Target Variable (Y):", list(cleaned_cols.keys()))
    df[target_col] = cleaned_cols[target_col]
    df = df.dropna(subset=[target_col])

    # Select exogenous variables
    exog_cols = st.multiselect("Select Exogenous Variables (optional):", 
                                [col for col in cleaned_cols.keys() if col != target_col])

    for exog in exog_cols:
        df[exog] = cleaned_cols[exog]

    if exog_cols:
        df = df.dropna(subset=exog_cols)

    # Plot clean time series
    st.subheader("Cleaned Target Plot")
    fig, ax = plt.subplots()
    df[target_col].plot(ax=ax, label="Observed", color="blue")
    ax.set_title("Cleaned Time Series Plot")
    ax.legend()
    st.pyplot(fig)

    # Grid Search Parameters
    st.subheader("Model Training - Grid Search")
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

    st.write(f"Total parameter sets to evaluate: {len(grid)}")

    progress = st.progress(0)
    results_list = []
    for idx, params in enumerate(grid):
        score = evaluate_params(params, df, target_col, exog_cols)
        results_list.append(score)
        progress.progress((idx + 1) / len(grid))

    best_idx = np.argmin(results_list)
    best_params = grid[best_idx]
    st.success(f"Best SARIMAX Parameters: {best_params}")
    st.write(f"Best AIC: {results_list[best_idx]}")

    results = fit_best_model(best_params, df, target_col, exog_cols)

    st.subheader("Model Summary")
    with st.expander("Show full model summary"):
        st.text(results.summary())

    # Forecasting
    st.subheader("Forecast")
    forecast_steps = st.slider("Forecast steps:", 10, 200, 30)

    if exog_cols:
        exog_forecast = np.tile(df[exog_cols].iloc[-1].values, (forecast_steps, 1))
    else:
        exog_forecast = None

    forecast = results.forecast(steps=forecast_steps, exog=exog_forecast)

    # Forecast plot
    fig2, ax2 = plt.subplots()
    df[target_col].plot(ax=ax2, label="Observed", color='blue')
    forecast.index = pd.date_range(df.index[-1], periods=forecast_steps+1, freq=df.index.freq or 'D')[1:]
    forecast.plot(ax=ax2, label="Forecast", color='red')
    ax2.set_title("Forecast vs Observed")
    ax2.legend()
    st.pyplot(fig2)

    # Download forecast
    forecast_df = pd.DataFrame({f"Forecast_{target_col}": forecast})
    forecast_df.index.name = 'Forecast_Date'
    st.download_button("Download Forecast CSV", forecast_df.to_csv(), file_name="forecast.csv", mime="text/csv")
