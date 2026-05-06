# Fuel Price Analysis - EDA, ML Modeling & Forecasting

# Dataset: fuel_prices_1970_2026.csv

# 

# What this does:

# - loads and cleans the data

# - explores it with some plots

# - trains a few regression models and picks the best one

# - forecasts the next 24 months

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(“ignore”)

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# —————————————————––

# STEP 1 - Load the data

# —————————————————––

df = pd.read_csv(“fuel_prices_1970_2026.csv”)

print(“shape:”, df.shape)
print(df.head())
print(df.dtypes)
print(”\nmissing values:\n”, df.isnull().sum())

# —————————————————––

# STEP 2 - Figure out which columns are date and price

# —————————————————––

date_col = None
price_col = None

for col in df.columns:
c = col.lower()
if any(x in c for x in [“date”, “year”, “month”, “time”, “period”]):
date_col = col
if any(x in c for x in [“price”, “cost”, “value”, “rate”, “usd”, “fuel”, “avg”, “dollar”]):
if price_col is None:
price_col = col

if price_col is None:
nums = df.select_dtypes(include=np.number).columns.tolist()
price_col = nums[-1] if nums else None

print(f”\nusing date column : {date_col}”)
print(f”using price column: {price_col}”)

# —————————————————––

# STEP 3 - Clean things up

# —————————————————––

# parse dates and sort chronologically

if date_col:
df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True, errors=“coerce”)
df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

```
df["year"]    = df[date_col].dt.year
df["month"]   = df[date_col].dt.month
df["quarter"] = df[date_col].dt.quarter
# months since start - useful as a numeric time feature
df["time_idx"] = (df["year"] - df["year"].min()) * 12 + df["month"]
```

# clean price column if it has dollar signs or commas

if price_col:
if df[price_col].dtype == object:
df[price_col] = (
df[price_col].astype(str)
.str.replace(r”[$,\s]”, “”, regex=True)
)
df[price_col] = pd.to_numeric(df[price_col], errors=“coerce”)
df = df.dropna(subset=[price_col])

# add lag features so the model can learn from past prices

df[“lag_1m”]     = df[price_col].shift(1)
df[“lag_3m”]     = df[price_col].shift(3)
df[“lag_12m”]    = df[price_col].shift(12)
df[“roll_3m”]    = df[price_col].rolling(3).mean()
df[“roll_12m”]   = df[price_col].rolling(12).mean()
df[“yoy_change”] = df[price_col].pct_change(12) * 100   # year over year %

df = df.dropna().reset_index(drop=True)
print(f”\nclean dataset size: {df.shape}”)

# —————————————————––

# STEP 4 - EDA plots

# —————————————————––

fig, axes = plt.subplots(2, 3, figsize=(17, 9))
fig.suptitle(“Fuel Price EDA (1970 - 2026)”, fontsize=14, fontweight=“bold”)

# price over time

ax = axes[0, 0]
ax.plot(df[date_col], df[price_col], linewidth=1.4, color=”#1f77b4”)
ax.set_title(“Price Over Time”)
ax.set_xlabel(“Date”)
ax.set_ylabel(“Price”)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

# distribution

ax = axes[0, 1]
ax.hist(df[price_col], bins=40, color=”#ff7f0e”, edgecolor=“white”, alpha=0.8)
mean_val = df[price_col].mean()
ax.axvline(mean_val, color=“black”, linestyle=”–”, label=f”mean = {mean_val:.2f}”)
ax.set_title(“Price Distribution”)
ax.set_xlabel(“Price”)
ax.legend()

# yearly average

ax = axes[0, 2]
yr = df.groupby(“year”)[price_col].mean()
ax.bar(yr.index, yr.values, color=”#2ca02c”, alpha=0.8)
ax.set_title(“Avg Price by Year”)
ax.set_xlabel(“Year”)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# monthly seasonality

ax = axes[1, 0]
mo = df.groupby(“month”)[price_col].mean()
months = [“Jan”,“Feb”,“Mar”,“Apr”,“May”,“Jun”,“Jul”,“Aug”,“Sep”,“Oct”,“Nov”,“Dec”]
ax.bar(mo.index, mo.values, color=”#9467bd”, alpha=0.8)
ax.set_title(“Seasonality by Month”)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(months, rotation=45)

# year over year change

ax = axes[1, 1]
ax.plot(df[date_col], df[“yoy_change”], linewidth=1.2, color=”#d62728”)
ax.axhline(0, color=“black”, linestyle=”–”, linewidth=0.8)
ax.set_title(“Year-over-Year Change (%)”)
ax.set_xlabel(“Date”)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

# correlation heatmap

ax = axes[1, 2]
num_cols = df.select_dtypes(include=np.number).drop(
columns=[“time_idx”, “year”, “month”, “quarter”], errors=“ignore”
)
sns.heatmap(num_cols.corr(), ax=ax, annot=True, fmt=”.2f”,
cmap=“coolwarm”, linewidths=0.5, annot_kws={“size”: 8})
ax.set_title(“Correlation Heatmap”)

plt.tight_layout()
plt.savefig(“eda_dashboard.png”, dpi=150, bbox_inches=“tight”)
plt.show()
print(“saved eda_dashboard.png”)

# —————————————————––

# STEP 5 - Train the models

# —————————————————––

features = [“time_idx”, “year”, “month”, “quarter”,
“lag_1m”, “lag_3m”, “lag_12m”, “roll_3m”, “roll_12m”]
features = [f for f in features if f in df.columns]
target = price_col

X = df[features]
y = df[target]

# split keeping time order intact (no random shuffle for time series)

cut = int(len(df) * 0.8)
X_train, X_test = X.iloc[:cut], X.iloc[cut:]
y_train, y_test = y.iloc[:cut], y.iloc[cut:]
dates_test = df[date_col].iloc[cut:]

print(f”\ntrain size: {len(X_train)}  |  test size: {len(X_test)}”)

scaler  = StandardScaler()
Xtr     = scaler.fit_transform(X_train)
Xte     = scaler.transform(X_test)

models = {
“linear regression”:   LinearRegression(),
“ridge”:               Ridge(alpha=1.0),
“random forest”:       RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
“gradient boosting”:   GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
max_depth=4, random_state=42),
}

tscv    = TimeSeriesSplit(n_splits=5)
results = {}

print(”\n— model scores —”)
for name, model in models.items():
model.fit(Xtr, y_train)
preds = model.predict(Xte)

```
mae  = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2   = r2_score(y_test, preds)
cv   = cross_val_score(model, Xtr, y_train, cv=tscv, scoring="r2").mean()

results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "CV_R2": cv, "preds": preds}
print(f"  {name:<22}  MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.4f}  CV={cv:.4f}")
```

best = max(results, key=lambda k: results[k][“R2”])
print(f”\nbest model: {best}  (R2={results[best][‘R2’]:.4f})”)

# —————————————————––

# STEP 6 - Plot the results

# —————————————————––

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(“Model Results - Fuel Price Forecasting”, fontsize=13, fontweight=“bold”)

# actual vs predicted

ax = axes[0, 0]
ax.plot(dates_test, y_test.values, label=“actual”,    color=”#1f77b4”, linewidth=1.8)
ax.plot(dates_test, results[best][“preds”], label=“predicted”,
color=”#ff7f0e”, linewidth=1.5, linestyle=”–”)
ax.set_title(f”Actual vs Predicted  ({best})”)
ax.set_xlabel(“Date”)
ax.legend()
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

# r2 comparison

ax = axes[0, 1]
names  = list(results.keys())
r2vals = [results[n][“R2”] for n in names]
bars   = ax.bar(names, r2vals, color=[”#1f77b4”,”#ff7f0e”,”#2ca02c”,”#9467bd”], alpha=0.85)
ax.set_title(“R2 Comparison”)
ax.set_ylim(0, 1.1)
ax.set_xticklabels(names, rotation=15, ha=“right”)
for b, v in zip(bars, r2vals):
ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f”{v:.3f}”, ha=“center”, fontsize=9)

# residuals

ax = axes[1, 0]
resid = y_test.values - results[best][“preds”]
ax.scatter(results[best][“preds”], resid, alpha=0.4, color=”#2ca02c”, s=15)
ax.axhline(0, color=“black”, linestyle=”–”)
ax.set_title(f”Residuals  ({best})”)
ax.set_xlabel(“Predicted”)
ax.set_ylabel(“Error”)

# feature importance from random forest

ax = axes[1, 1]
rf_imp = pd.Series(
models[“random forest”].feature_importances_, index=features
).sort_values()
rf_imp.plot(kind=“barh”, ax=ax, color=”#d62728”, alpha=0.8)
ax.set_title(“Feature Importance (Random Forest)”)

plt.tight_layout()
plt.savefig(“model_results.png”, dpi=150, bbox_inches=“tight”)
plt.show()
print(“saved model_results.png”)

# —————————————————––

# STEP 7 - Forecast next 24 months

# —————————————————––

best_model = models[best]
temp       = df.copy()

future = []
for _ in range(24):
last = temp.iloc[-1]
ny   = int(last[“year”] + (last[“month”] == 12))
nm   = int(last[“month”] % 12) + 1

```
row = {
    "time_idx": last["time_idx"] + 1,
    "year":     ny,
    "month":    nm,
    "quarter":  (nm - 1) // 3 + 1,
    "lag_1m":   temp[price_col].iloc[-1],
    "lag_3m":   temp[price_col].iloc[-3]  if len(temp) >= 3  else temp[price_col].iloc[-1],
    "lag_12m":  temp[price_col].iloc[-12] if len(temp) >= 12 else temp[price_col].iloc[-1],
    "roll_3m":  temp[price_col].iloc[-3:].mean(),
    "roll_12m": temp[price_col].iloc[-12:].mean(),
}

fv   = scaler.transform([[row[f] for f in features]])
pred = best_model.predict(fv)[0]

row[price_col] = pred
row[date_col]  = pd.Timestamp(year=ny, month=nm, day=1)
future.append(row)
temp = pd.concat([temp, pd.DataFrame([row])], ignore_index=True)
```

future_df = pd.DataFrame(future)

fig, ax = plt.subplots(figsize=(13, 5))
n = 60
ax.plot(df[date_col].iloc[-n:], df[price_col].iloc[-n:],
color=”#1f77b4”, linewidth=2, label=“historical”)
ax.plot(future_df[date_col], future_df[price_col],
color=”#ff7f0e”, linewidth=2, linestyle=”–”, marker=“o”, markersize=4, label=“forecast”)
ax.axvline(df[date_col].iloc[-1], color=“gray”, linestyle=”:”, linewidth=1.2, label=“today”)
ax.set_title(f”24-Month Fuel Price Forecast  ({best})”, fontsize=12)
ax.set_xlabel(“Date”)
ax.set_ylabel(“Price”)
ax.legend()
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
plt.tight_layout()
plt.savefig(“forecast.png”, dpi=150, bbox_inches=“tight”)
plt.show()
print(“saved forecast.png”)

# —————————————————––

# final summary

# —————————————————––

print(”\n=== summary ===”)
for name, res in results.items():
print(f”  {name:<22}  R2={res[‘R2’]:.4f}  RMSE={res[‘RMSE’]:.4f}  MAE={res[‘MAE’]:.4f}”)

print(f”\nbest: {best}”)
print(“outputs: eda_dashboard.png | model_results.png | forecast.png”)