# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Pharmacy Claims Forecasting Model
# MAGIC %md
# MAGIC # Pharmacy Claims Forecasting Model
# MAGIC
# MAGIC Time-series forecasting of monthly pharmacy claims volume using Prophet, trained on `com_edp_prd.com_intgr.claims_pharmacy_events`. Includes EDA, stationarity analysis, model training with MLflow tracking, and evaluation on a held-out test set.

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install prophet mlflow statsmodels --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load and Aggregate Monthly Claims
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')

# Load monthly aggregated claims from the Silver table
# Exclude incomplete current month (Aug 2026) and use data from 2015+
monthly_claims_df = spark.sql("""
    SELECT 
        CAST(DATE_TRUNC('month', FILL_DATE) AS DATE) AS month,
        COUNT(*) AS claim_count,
        COUNT(DISTINCT PATIENT_ID) AS patient_count
    FROM com_edp_prd.com_intgr.claims_pharmacy_events
    WHERE FILL_DATE IS NOT NULL
      AND FILL_DATE >= '2015-01-01'
      AND FILL_DATE < '2026-08-01'
    GROUP BY DATE_TRUNC('month', FILL_DATE)
    ORDER BY month
""").toPandas()

monthly_claims_df['month'] = pd.to_datetime(monthly_claims_df['month'])
print(f"Dataset: {len(monthly_claims_df)} months from {monthly_claims_df['month'].min().date()} to {monthly_claims_df['month'].max().date()}")
print(f"Avg monthly claims: {monthly_claims_df['claim_count'].mean():,.0f}")
print(f"Avg monthly distinct patients: {monthly_claims_df['patient_count'].mean():,.0f}")
display(monthly_claims_df[['claim_count', 'patient_count']].describe())

# COMMAND ----------

# DBTITLE 1,EDA — Monthly Claims Trend
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Plot 1: Monthly claim volume trend
axes[0].plot(monthly_claims_df['month'], monthly_claims_df['claim_count'], 
             color='#1f77b4', linewidth=1.5, label='Monthly Claims')
# Rolling 12-month average
rolling_avg = monthly_claims_df['claim_count'].rolling(window=12, center=True).mean()
axes[0].plot(monthly_claims_df['month'], rolling_avg, 
             color='#d62728', linewidth=2, linestyle='--', label='12-Month Rolling Avg')
axes[0].set_title('Monthly Pharmacy Claims Volume (2015–2026)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Claim Count')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Year-over-year comparison
monthly_claims_df['year'] = monthly_claims_df['month'].dt.year
monthly_claims_df['month_num'] = monthly_claims_df['month'].dt.month
for year in monthly_claims_df['year'].unique():
    year_data = monthly_claims_df[monthly_claims_df['year'] == year]
    axes[1].plot(year_data['month_num'], year_data['claim_count'], 
                 label=str(year), linewidth=1.2, alpha=0.8)
axes[1].set_title('Year-over-Year Monthly Claims Comparison', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Claim Count')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,EDA — Seasonality Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

# Seasonal decomposition (multiplicative)
ts = monthly_claims_df.set_index('month')['claim_count']
decomposition = seasonal_decompose(ts, model='additive', period=12)

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
decomposition.observed.plot(ax=axes[0], color='#1f77b4')
axes[0].set_ylabel('Observed')
axes[0].set_title('Seasonal Decomposition of Monthly Pharmacy Claims', fontsize=14, fontweight='bold')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))

decomposition.trend.plot(ax=axes[1], color='#ff7f0e')
axes[1].set_ylabel('Trend')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))

decomposition.seasonal.plot(ax=axes[2], color='#2ca02c')
axes[2].set_ylabel('Seasonal')
axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))

decomposition.resid.plot(ax=axes[3], color='#d62728')
axes[3].set_ylabel('Residual')
axes[3].set_xlabel('Date')
axes[3].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Data Preprocessing — Time Series Preparation & Stationarity
from statsmodels.tsa.stattools import adfuller

# Create complete monthly time series (fill any gaps)
date_range = pd.date_range(
    start=monthly_claims_df['month'].min(),
    end=monthly_claims_df['month'].max(),
    freq='MS'
)
ts_df = pd.DataFrame({'month': date_range})
ts_df = ts_df.merge(monthly_claims_df[['month', 'claim_count']], on='month', how='left')

# Check for missing months
missing_months = ts_df[ts_df['claim_count'].isna()]
if len(missing_months) > 0:
    print(f"Found {len(missing_months)} missing months — forward filling")
    ts_df['claim_count'] = ts_df['claim_count'].ffill()
else:
    print("No missing months in the time series")

# Stationarity test (Augmented Dickey-Fuller)
result = adfuller(ts_df['claim_count'].dropna(), autolag='AIC')
print(f"\nAugmented Dickey-Fuller Test:")
print(f"  Test Statistic: {result[0]:.4f}")
print(f"  p-value: {result[1]:.4f}")
print(f"  Critical Values: {result[4]}")
if result[1] < 0.05:
    print("  → Series is stationary (reject null hypothesis)")
else:
    print("  → Series is non-stationary (fail to reject null hypothesis)")
    print("  → Prophet handles non-stationarity via trend modeling")

# Prepare Prophet format: ds (datestamp) and y (value)
prophet_df = ts_df[['month', 'claim_count']].rename(columns={'month': 'ds', 'claim_count': 'y'})

# Time-based train/test split: hold out last 12 months for evaluation
train_size = len(prophet_df) - 12
train_df = prophet_df.iloc[:train_size].copy()
test_df = prophet_df.iloc[train_size:].copy()

print(f"\nTrain set: {len(train_df)} months ({train_df['ds'].min().date()} to {train_df['ds'].max().date()})")
print(f"Test set:  {len(test_df)} months ({test_df['ds'].min().date()} to {test_df['ds'].max().date()})")

# COMMAND ----------

# DBTITLE 1,Model Training — Prophet with MLflow
import mlflow
from prophet import Prophet
from mlflow.models import infer_signature
import json

# Set MLflow experiment
experiment_name = "/Users/hembram@dnli.com/Pharmacy_Claims_Forecasting"
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name="prophet_pharmacy_claims_forecast") as run:
    # Model hyperparameters
    params = {
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
        "seasonality_mode": "additive",
        "yearly_seasonality": True,
        "weekly_seasonality": False,
        "daily_seasonality": False,
        "forecast_horizon_months": 12,
        "train_months": len(train_df),
    }
    mlflow.log_params(params)
    
    # Train Prophet model
    model = Prophet(
        changepoint_prior_scale=params["changepoint_prior_scale"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        seasonality_mode=params["seasonality_mode"],
        yearly_seasonality=params["yearly_seasonality"],
        weekly_seasonality=params["weekly_seasonality"],
        daily_seasonality=params["daily_seasonality"],
    )
    model.fit(train_df)
    
    # Forecast on test period
    future = model.make_future_dataframe(periods=12, freq='MS')
    forecast = model.predict(future)
    
    # Evaluate on test set
    test_forecast = forecast.iloc[-12:][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].reset_index(drop=True)
    test_actual = test_df.reset_index(drop=True)
    
    mae = np.mean(np.abs(test_actual['y'].values - test_forecast['yhat'].values))
    rmse = np.sqrt(np.mean((test_actual['y'].values - test_forecast['yhat'].values) ** 2))
    mape = np.mean(np.abs((test_actual['y'].values - test_forecast['yhat'].values) / test_actual['y'].values)) * 100
    
    mlflow.log_metrics({"MAE": mae, "RMSE": rmse, "MAPE": mape})
    
    # Log model with signature and input example
    input_example = train_df.head(3)
    signature = infer_signature(input_example, model.predict(input_example)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    
    mlflow.prophet.log_model(
        model,
        name="prophet_model",
        signature=signature,
        input_example=input_example,
    )
    
    # Log training data summary
    mlflow.log_dict({
        "train_start": str(train_df['ds'].min().date()),
        "train_end": str(train_df['ds'].max().date()),
        "test_start": str(test_df['ds'].min().date()),
        "test_end": str(test_df['ds'].max().date()),
        "total_months": len(prophet_df),
    }, "data_summary.json")
    
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")
    print(f"\nTest Set Metrics:")
    print(f"  MAE:  {mae:,.0f} claims")
    print(f"  RMSE: {rmse:,.0f} claims")
    print(f"  MAPE: {mape:.2f}%")

# COMMAND ----------

# DBTITLE 1,Evaluation — Forecast vs Actuals Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Full forecast with confidence interval
axes[0].plot(train_df['ds'], train_df['y'], color='#1f77b4', linewidth=1.5, label='Training Data')
axes[0].plot(test_actual['ds'], test_actual['y'], color='#2ca02c', linewidth=2, marker='o', label='Actual (Test)')
axes[0].plot(test_forecast['ds'], test_forecast['yhat'], color='#d62728', linewidth=2, marker='s', linestyle='--', label='Forecast')
axes[0].fill_between(test_forecast['ds'], test_forecast['yhat_lower'], test_forecast['yhat_upper'], 
                      color='#d62728', alpha=0.15, label='95% Confidence Interval')
axes[0].axvline(x=train_df['ds'].max(), color='gray', linestyle=':', alpha=0.7, label='Train/Test Split')
axes[0].set_title('Pharmacy Claims Forecast vs Actuals', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Monthly Claim Count')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

# Plot 2: Zoomed-in test period with error bars
month_labels = test_actual['ds'].dt.strftime('%b %Y')
x_pos = np.arange(len(month_labels))
width = 0.35

axes[1].bar(x_pos - width/2, test_actual['y'], width, color='#2ca02c', alpha=0.8, label='Actual')
axes[1].bar(x_pos + width/2, test_forecast['yhat'], width, color='#d62728', alpha=0.8, label='Forecast')
axes[1].set_title('Test Period: Monthly Actual vs Forecast Claims', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Claim Count')
axes[1].set_xlabel('Month')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(month_labels, rotation=45, ha='right')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# Log the evaluation figure to MLflow
with mlflow.start_run(run_id=run_id):
    mlflow.log_figure(fig, "forecast_vs_actuals.png")

plt.show()

# Print detailed comparison table
comparison = pd.DataFrame({
    'Month': test_actual['ds'].dt.strftime('%Y-%m'),
    'Actual': test_actual['y'].values,
    'Forecast': test_forecast['yhat'].values.astype(int),
    'Error': (test_actual['y'].values - test_forecast['yhat'].values).astype(int),
    'Error %': ((test_actual['y'].values - test_forecast['yhat'].values) / test_actual['y'].values * 100).round(2)
})
print("\nDetailed Forecast Comparison:")
display(comparison)

# COMMAND ----------

# DBTITLE 1,Model Summary & MLflow Experiment Link
print("="*60)
print("PHARMACY CLAIMS FORECASTING — MODEL SUMMARY")
print("="*60)
print(f"\nModel: Prophet (additive seasonality)")
print(f"Training period: {train_df['ds'].min().date()} to {train_df['ds'].max().date()} ({len(train_df)} months)")
print(f"Test period: {test_df['ds'].min().date()} to {test_df['ds'].max().date()} ({len(test_df)} months)")
print(f"\nEvaluation Metrics:")
print(f"  MAE:  {mae:>10,.0f} claims")
print(f"  RMSE: {rmse:>10,.0f} claims")
print(f"  MAPE: {mape:>10.2f}%")
print(f"\nMLflow Experiment: {experiment_name}")
print(f"MLflow Run ID: {run_id}")
print(f"\nAll parameters, metrics, model artifact, and visualizations")
print(f"have been logged to MLflow for reproducibility.")
print("="*60)

# COMMAND ----------

# DBTITLE 1,6-Month Forward Forecast (Aug 2026 – Jan 2027)
# Retrain Prophet on ALL available data (Jan 2015 – Jul 2026) for the best forward forecast
full_model = Prophet(
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    seasonality_mode="additive",
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
)
full_model.fit(prophet_df)

# Generate 6-month forward forecast
future_6m = full_model.make_future_dataframe(periods=6, freq='MS')
forecast_6m = full_model.predict(future_6m)

# Extract the 6 new forecast months
fwd_forecast = forecast_6m.iloc[-6:][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].reset_index(drop=True)
fwd_forecast['yhat'] = fwd_forecast['yhat'].round(0).astype(int)
fwd_forecast['yhat_lower'] = fwd_forecast['yhat_lower'].round(0).astype(int)
fwd_forecast['yhat_upper'] = fwd_forecast['yhat_upper'].round(0).astype(int)

# Log the forward-forecast run to MLflow
with mlflow.start_run(run_name="prophet_6m_forward_forecast") as fwd_run:
    mlflow.log_params({
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
        "seasonality_mode": "additive",
        "train_months": len(prophet_df),
        "forecast_horizon_months": 6,
        "retrained_on_full_data": True,
    })
    # Reuse the evaluation metrics from the held-out test as a reference
    mlflow.log_metrics({"reference_MAE": mae, "reference_RMSE": rmse, "reference_MAPE": mape})

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(14, 6))

    # Last 24 months of actuals for context
    recent = prophet_df.tail(24)
    ax.plot(recent['ds'], recent['y'], color='#1f77b4', linewidth=1.5,
            marker='o', markersize=4, label='Actuals (last 24 months)')

    # Forward forecast
    ax.plot(fwd_forecast['ds'], fwd_forecast['yhat'], color='#d62728',
            linewidth=2, marker='s', markersize=6, linestyle='--', label='6-Month Forecast')
    ax.fill_between(fwd_forecast['ds'], fwd_forecast['yhat_lower'], fwd_forecast['yhat_upper'],
                    color='#d62728', alpha=0.15, label='95% Confidence Interval')
    ax.axvline(x=prophet_df['ds'].max(), color='gray', linestyle=':', alpha=0.7, label='Last Actual')

    ax.set_title('6-Month Forward Pharmacy Claims Forecast (Aug 2026 – Jan 2027)',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Monthly Claim Count')
    ax.set_xlabel('Date')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    mlflow.log_figure(fig, "6m_forward_forecast.png")
    fwd_run_id = fwd_run.info.run_id

plt.show()

# Print forecast table
fwd_table = fwd_forecast.copy()
fwd_table['ds'] = fwd_table['ds'].dt.strftime('%Y-%m')
fwd_table.columns = ['Month', 'Forecast', 'Lower 95%', 'Upper 95%']
print("\n6-Month Forward Forecast:")
display(fwd_table)
print(f"\nMLflow Run ID: {fwd_run_id}")

# COMMAND ----------

# DBTITLE 1,Historical Avlayah Claims Proportion
# Avlayah identification: NDC '84976000101' or BRAND_NAME = 'AVLAYAH'
# Query historical monthly Avlayah claim counts and proportions
avlayah_history = spark.sql("""
  SELECT 
    CAST(DATE_TRUNC('month', FILL_DATE) AS DATE) AS month,
    COUNT(*) AS total_claims,
    SUM(CASE WHEN NDC11 = '84976000101' OR UPPER(BRAND_NAME) = 'AVLAYAH' THEN 1 ELSE 0 END) AS avlayah_claims,
    COUNT(DISTINCT PATIENT_ID) AS total_patients,
    COUNT(DISTINCT CASE WHEN NDC11 = '84976000101' OR UPPER(BRAND_NAME) = 'AVLAYAH' THEN PATIENT_ID END) AS avlayah_patients
  FROM com_edp_prd.com_intgr.claims_pharmacy_events
  WHERE FILL_DATE IS NOT NULL
    AND FILL_DATE >= '2024-01-01'
    AND FILL_DATE < '2026-08-01'
  GROUP BY DATE_TRUNC('month', FILL_DATE)
  ORDER BY month
""").toPandas()

avlayah_history['month'] = pd.to_datetime(avlayah_history['month'])
avlayah_history['avlayah_pct'] = (avlayah_history['avlayah_claims'] / avlayah_history['total_claims'] * 100).round(4)

print(f"Avlayah claims summary (Jan 2024 – Jul 2026):")
print(f"  Avg monthly Avlayah claims:   {avlayah_history['avlayah_claims'].mean():.1f}")
print(f"  Avg monthly Avlayah patients: {avlayah_history['avlayah_patients'].mean():.1f}")
print(f"  Avg Avlayah share of total:   {avlayah_history['avlayah_pct'].mean():.4f}%")
display(avlayah_history)

# COMMAND ----------

# DBTITLE 1,Avlayah Claims Forecast (Aug 2026 – Jan 2027)
from sklearn.linear_model import LinearRegression

# Avlayah claims started Apr 2026 — extract the adoption ramp
avlayah_active = avlayah_history[avlayah_history['avlayah_claims'] > 0].copy()
avlayah_active['month_index'] = np.arange(1, len(avlayah_active) + 1)  # 1=Apr, 2=May, 3=Jun, 4=Jul

print("Avlayah Pharmacy Claims Adoption (observed):")
for _, r in avlayah_active.iterrows():
    print(f"  {r['month'].strftime('%b %Y')}: {int(r['avlayah_claims']):>3} claims, "
          f"{int(r['avlayah_patients']):>2} patients (COUNT DISTINCT)")

# Fit a log-linear model to capture rapid early growth
# ln(claims) ~ month_index
avlayah_active['log_claims'] = np.log(avlayah_active['avlayah_claims'])
X_hist = avlayah_active[['month_index']].values
y_hist = avlayah_active['log_claims'].values

growth_model = LinearRegression()
growth_model.fit(X_hist, y_hist)

# Also fit log-linear model for patient counts
patient_model = LinearRegression()
patient_model.fit(X_hist, np.log(avlayah_active['avlayah_patients'].values))

print(f"\nGrowth model: ln(claims) = {growth_model.coef_[0]:.3f} * month + {growth_model.intercept_:.3f}")
print(f"Monthly growth rate: ~{(np.exp(growth_model.coef_[0]) - 1)*100:.0f}%")

# Project forward 6 months (Aug 2026 = index 5, ..., Jan 2027 = index 10)
fwd_months = pd.date_range('2026-08-01', periods=6, freq='MS')
fwd_indices = np.arange(5, 11).reshape(-1, 1)

projected_claims = np.exp(growth_model.predict(fwd_indices)).round(0).astype(int)
projected_patients = np.exp(patient_model.predict(fwd_indices)).round(0).astype(int)

# Cap patients at 42 (the total Avlayah cohort per business definition)
projected_patients = np.minimum(projected_patients, 42)

# Combine with the total claims forecast from the Prophet model
avlayah_fwd = pd.DataFrame({
    'Month': fwd_months.strftime('%Y-%m'),
    'Total Forecast Claims': fwd_forecast['yhat'].values,
    'Projected Avlayah Claims': projected_claims,
    'Projected Avlayah Patients': projected_patients,
    'Avlayah % of Total': (projected_claims / fwd_forecast['yhat'].values * 100).round(4)
})

print("\n" + "="*70)
print("AVLAYAH CLAIMS FORECAST (Aug 2026 – Jan 2027)")
print("="*70)
print("Method: Log-linear growth model fit on Apr–Jul 2026 adoption ramp")
print(f"Observed monthly growth rate: ~{(np.exp(growth_model.coef_[0]) - 1)*100:.0f}%")
print("\nNote: Projections assume the current rapid adoption trajectory")
print("continues. Actual growth may plateau as the eligible patient")
print("pool saturates (42 total Avlayah patients per cohort analysis).")
print("Patient counts are capped at 42 (total known Avlayah cohort).")
print("="*70)
display(avlayah_fwd)

# Visualization
fig, ax1 = plt.subplots(figsize=(12, 6))

# Historical Avlayah claims
ax1.bar(avlayah_active['month'].dt.strftime('%b %Y'), avlayah_active['avlayah_claims'],
        color='#2ca02c', alpha=0.8, label='Actual Avlayah Claims')
# Projected Avlayah claims
ax1.bar(avlayah_fwd['Month'], avlayah_fwd['Projected Avlayah Claims'],
        color='#d62728', alpha=0.7, label='Projected Avlayah Claims')

# Patient counts on secondary axis
ax2 = ax1.twinx()
hist_pts = avlayah_active['avlayah_patients'].values
fwd_pts = avlayah_fwd['Projected Avlayah Patients'].values
all_x = list(avlayah_active['month'].dt.strftime('%b %Y')) + list(avlayah_fwd['Month'])
all_patients = list(hist_pts) + list(fwd_pts)
ax2.plot(all_x, all_patients, color='#ff7f0e', linewidth=2, marker='o', label='Avlayah Patients (distinct)')
ax2.axhline(y=42, color='#ff7f0e', linestyle='--', alpha=0.5, label='Max Cohort (42 patients)')
ax2.set_ylabel('Distinct Patients', color='#ff7f0e')
ax2.tick_params(axis='y', labelcolor='#ff7f0e')

ax1.axvline(x=3.5, color='gray', linestyle=':', alpha=0.7, label='Actual / Projected Split')
ax1.set_title('Avlayah Pharmacy Claims: Actual & Projected (Apr 2026 – Jan 2027)',
              fontsize=14, fontweight='bold')
ax1.set_ylabel('Claim Count')
ax1.set_xlabel('Month')
plt.xticks(rotation=45, ha='right')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.show()