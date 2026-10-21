import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Predictive Analytics Dashboard", page_icon="📈", layout="wide")
st.title("📈 Predictive Analytics Using Historical Data")
st.write("Upload historical data, train a regression model, evaluate it, and forecast future trends.")

@st.cache_data
def load_sample_data():
    dates = pd.date_range(start="2024-01-01", periods=36, freq="MS")
    rng = np.random.default_rng(42)
    values = 120 + np.arange(36) * 3.2 + 18 * np.sin(np.arange(36) * 2 * np.pi / 12) + rng.normal(0, 7, 36)
    return pd.DataFrame({"Date": dates, "Sales": np.round(values, 2)})

uploaded_file = st.file_uploader("Upload your historical dataset (CSV)", type=["csv"])
try:
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data_source = uploaded_file.name
    else:
        data = load_sample_data()
        data_source = "Built-in sample data"
except Exception as exc:
    st.error(f"Could not read this CSV file: {exc}")
    st.stop()

if data.empty or len(data.columns) < 2:
    st.error("Your CSV must contain at least two columns: a date/time column and a numeric value column.")
    st.stop()

st.caption(f"Data source: {data_source} · Original rows: {len(data)}")
left, right = st.columns(2)
with left:
    date_col = st.selectbox("Choose the date/time column", options=list(data.columns), index=0)
with right:
    numeric_cols = [col for col in data.columns if col != date_col and pd.api.types.is_numeric_dtype(data[col])]
    if not numeric_cols:
        for col in data.columns:
            if col != date_col:
                converted = pd.to_numeric(data[col], errors="coerce")
                if converted.notna().any():
                    data[col] = converted
                    numeric_cols.append(col)
    if not numeric_cols:
        st.error("No numeric target column was found. Ensure one column contains numbers, such as Sales, Revenue, or Demand.")
        st.stop()
    target_col = st.selectbox("Choose the value to forecast", options=numeric_cols)

clean = data[[date_col, target_col]].copy()
clean[date_col] = pd.to_datetime(clean[date_col], errors="coerce")
clean[target_col] = pd.to_numeric(clean[target_col], errors="coerce")
clean = clean.dropna().sort_values(date_col)
clean = clean.groupby(date_col, as_index=False)[target_col].mean()

if len(clean) < 6:
    st.error("Please provide at least 6 valid date/value rows to train and evaluate a model.")
    st.stop()

st.subheader("1. Cleaned historical data")
st.write(f"Rows after cleaning: **{len(clean)}** · Removed invalid/missing rows: **{len(data) - len(clean)}**")
st.dataframe(clean, use_container_width=True, hide_index=True)

forecast_periods = st.slider("Number of future periods to forecast", min_value=1, max_value=24, value=6)
train_ratio = st.slider("Training data percentage", min_value=60, max_value=90, value=80, step=5)

# Use chronological order to avoid leaking future observations into training.
clean = clean.reset_index(drop=True)
clean["time_index"] = np.arange(len(clean))
split_at = int(len(clean) * train_ratio / 100)
split_at = min(max(split_at, 3), len(clean) - 2)
train = clean.iloc[:split_at]
test = clean.iloc[split_at:]
model = LinearRegression()
model.fit(train[["time_index"]], train[target_col])
test_predictions = model.predict(test[["time_index"]])

mae = mean_absolute_error(test[target_col], test_predictions)
rmse = float(np.sqrt(mean_squared_error(test[target_col], test_predictions)))
r2 = r2_score(test[target_col], test_predictions) if len(test) >= 2 else float("nan")

st.subheader("2. Model evaluation")
m1, m2, m3 = st.columns(3)
m1.metric("Mean Absolute Error (MAE)", f"{mae:,.2f}")
m2.metric("Root Mean Squared Error (RMSE)", f"{rmse:,.2f}")
m3.metric("R² score", "Not available" if np.isnan(r2) else f"{r2:.3f}")
st.caption("Metrics are calculated on the final chronological test segment. Lower MAE/RMSE generally means smaller prediction errors; R² closer to 1 generally indicates a better fit.")

# Refit using all available history for the final future forecast.
final_model = LinearRegression()
final_model.fit(clean[["time_index"]], clean[target_col])
future_indices = np.arange(len(clean), len(clean) + forecast_periods)
future_values = final_model.predict(future_indices.reshape(-1, 1))
last_date = clean[date_col].iloc[-1]
try:
    inferred_freq = pd.infer_freq(clean[date_col]) if len(clean) >= 3 else None
    if inferred_freq:
        future_dates = pd.date_range(start=last_date, periods=forecast_periods + 1, freq=inferred_freq)[1:]
    else:
        day_gaps = clean[date_col].diff().dropna().dt.days
        step_days = max(1, int(day_gaps.median())) if not day_gaps.empty else 30
        future_dates = pd.DatetimeIndex([last_date + pd.Timedelta(days=step_days * (i + 1)) for i in range(forecast_periods)])
except Exception:
    future_dates = pd.DatetimeIndex([last_date + pd.Timedelta(days=30 * (i + 1)) for i in range(forecast_periods)])

forecast = pd.DataFrame({date_col: future_dates, f"Predicted {target_col}": np.round(future_values, 2)})

st.subheader("3. Historical trend and future forecast")
fig = go.Figure()
fig.add_trace(go.Scatter(x=clean[date_col], y=clean[target_col], mode="lines+markers", name="Historical values"))
fig.add_trace(go.Scatter(x=test[date_col], y=test_predictions, mode="lines+markers", name="Test predictions", line=dict(dash="dash")))
fig.add_trace(go.Scatter(x=forecast[date_col], y=forecast[f"Predicted {target_col}"], mode="lines+markers", name="Future forecast", line=dict(dash="dot")))
fig.update_layout(xaxis_title="Date", yaxis_title=target_col, legend_title="Series", hovermode="x unified", margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(fig, use_container_width=True)

st.subheader("4. Forecast results")
st.dataframe(forecast, use_container_width=True, hide_index=True)
csv_bytes = forecast.to_csv(index=False).encode("utf-8")
st.download_button("Download forecast as CSV", data=csv_bytes, file_name="forecast_results.csv", mime="text/csv")

st.info("This starter project uses Linear Regression on time order. It is suitable for demonstrating a forecasting workflow, but it does not model complex seasonality or external factors. For real business decisions, compare with suitable time-series models and validate on more historical data.")
st.markdown("---\n**Project:** Predictive Analytics Using Historical Data · **Stack:** Python, Pandas, Scikit-learn, Plotly, Streamlit")
