from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import scipy.stats as stats
from itertools import product
import time
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.metrics import r2_score


# ============================================================
# Global configuration
# ============================================================

SAVE_FILES = False
SHOW_PLOTS = True
BLOCK_PLOTS = False
AUTO_CLOSE_SECONDS = 0.5

SEED = 42
tf.random.set_seed(SEED)

TARGET = "Global_active_power"
LOOKBACK = 24
HORIZON = 24

NUMERIC_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

BASE_EXOG_COLS = ["is_weekend", "is_work_hours", "is_morning", "is_evening"]


# ============================================================
# Plot helper
# ============================================================

def _finalize_plot():
    if not SHOW_PLOTS:
        plt.close()
        return
    if BLOCK_PLOTS:
        plt.show()
    else:
        plt.show(block=False)
        plt.pause(AUTO_CLOSE_SECONDS)
        plt.close()

# ============================================================
# Metrics
# ============================================================

def calc_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(100.0 * np.mean(2.0 * np.abs(y_true - y_pred) / denom))

def mase(y_true, y_pred, train_series, season_lag=24, eps=1e-8):
    train_series = pd.Series(train_series).dropna().astype(float)
    diff = train_series[season_lag:] - train_series.shift(season_lag)[season_lag:]
    scale = float(np.mean(np.abs(diff.dropna())))
    model_mae = float(mean_absolute_error(y_true, y_pred))
    return model_mae / (scale + eps)

# ============================================================
# Unified Evaluation Function
# ============================================================

def evaluate_model(y_true, y_pred, train_series, residuals=None, model_name="Model"):

    rmse_val = calc_rmse(y_true, y_pred)
    mae_val = mean_absolute_error(y_true, y_pred)
    mase_val = mase(y_true, y_pred, train_series)
    r2_val = r2_score(y_true, y_pred)
    smape_val = smape(y_true, y_pred)

    lb_pvalue = np.nan

    if residuals is not None:
        residuals = pd.Series(residuals).dropna()
        lb_test = acorr_ljungbox(residuals, lags=[24], return_df=True)
        lb_pvalue = float(lb_test["lb_pvalue"].iloc[0])

    return {
        "Model": model_name,
        "RMSE": rmse_val,
        "MAE": mae_val,
        "MASE": mase_val,
        "R2": r2_val,
        "SMAPE": smape_val,
        "LjungBox_pvalue": lb_pvalue
    }

# ============================================================
# Preprocessing
# ============================================================

def preprocess_household_power(input_path, resample_rule="1h"):
    df = pd.read_csv(input_path, sep=";", na_values=["?"], low_memory=False)
    df.columns = df.columns.str.strip()

    df["datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    df = df.dropna(subset=["datetime"])

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop(columns=["Date", "Time"]).set_index("datetime").sort_index()
    df = df.resample(resample_rule).mean().asfreq(resample_rule)

    return df

def add_calendar_features(df):
    out = df.copy()
    idx = out.index
    out["hour"] = idx.hour
    out["day_of_week"] = idx.dayofweek
    out["month"] = idx.month
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_morning"] = ((out["hour"] >= 6) & (out["hour"] <= 11)).astype(int)
    out["is_evening"] = ((out["hour"] >= 18) & (out["hour"] <= 22)).astype(int)
    out["is_work_hours"] = ((out["day_of_week"] <= 4) & (out["hour"] >= 9) & (out["hour"] <= 17)).astype(int)

    season_map = {
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "fall", 10: "fall", 11: "fall",
    }
    out["season"] = out["month"].map(season_map)

    return out

def add_cyclical_time_features(df):
    out = df.copy()
    h = out.index.hour
    d = out.index.dayofweek
    out["hour_sin"] = np.sin(2 * np.pi * h / 24)
    out["hour_cos"] = np.cos(2 * np.pi * h / 24)
    out["dow_sin"] = np.sin(2 * np.pi * d / 7)
    out["dow_cos"] = np.cos(2 * np.pi * d / 7)
    return out

def add_target_lags(df):
    out = df.copy()
    out["y_lag_1"] = out[TARGET].shift(1)
    out["y_lag_24"] = out[TARGET].shift(24)
    out["y_lag_168"] = out[TARGET].shift(168)
    out["y_rollmean_24"] = out[TARGET].shift(1).rolling(24).mean()
    out["y_rollmean_168"] = out[TARGET].shift(1).rolling(168).mean()
    return out

def seasonal_naive_forecast(train_ts, test_ts, season_lag=24):

    history = pd.concat([train_ts, test_ts])
    preds = []

    for i in range(len(test_ts)):
        idx = len(train_ts) + i
        preds.append(history.iloc[idx - season_lag])

    return pd.Series(preds, index=test_ts.index)

# ============================================================
# Pre-Model Diagnostics (ADF + ACF/PACF)
# ============================================================

def run_pre_model_diagnostics(train_ts, max_lags=168):

    print("\n=== PRE-MODEL DIAGNOSTICS ===")

    # --------------------------------------------------
    # 1️⃣ ADF Test
    # --------------------------------------------------
    adf_result = adfuller(train_ts.dropna())

    print("\nADF Test Results:")
    print("ADF Statistic :", adf_result[0])
    print("p-value       :", adf_result[1])
    print("Critical Values:")
    for key, value in adf_result[4].items():
        print(f"   {key}: {value}")

    if adf_result[1] < 0.05:
        print("→ Series is likely stationary (reject H0).")
    else:
        print("→ Series is likely non-stationary (fail to reject H0).")

    # --------------------------------------------------
    # 2️⃣ ACF Plot
    # --------------------------------------------------
    plt.figure(figsize=(10, 4))
    plot_acf(train_ts.dropna(), lags=max_lags)
    plt.title("ACF Plot (Training Series)")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # 3️⃣ PACF Plot
    # --------------------------------------------------
    plt.figure(figsize=(10, 4))
    plot_pacf(train_ts.dropna(), lags=max_lags, method="ywm")
    plt.title("PACF Plot (Training Series)")
    plt.tight_layout()
    plt.show()

# ============================================================
# SARIMA MODEL
# ============================================================

def add_weekly_fourier(df, K=2):
    """
    Adds weekly Fourier terms.
    K = number of harmonics (2 is usually enough)
    """
    out = df.copy()
    t = np.arange(len(out))

    period = 24 * 7  # 168 hours

    for k in range(1, K + 1):
        out[f"week_sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        out[f"week_cos_{k}"] = np.cos(2 * np.pi * k * t / period)

    return out

def run_sarimax_dual_seasonal(train_ts, val_ts, test_ts,
                               order=(2,1,2),
                               seasonal_order=(1,1,1,24),
                               weekly_period=168,
                               fourier_k=3):

    print("\n=== SARIMAX (Daily SARIMA + Weekly Fourier Improved) ===")

    # -----------------------------
    # Build Fourier Terms
    # -----------------------------
    def make_fourier(index):
        t = np.arange(len(index))
        X = []
        for k in range(1, fourier_k + 1):
            X.append(np.sin(2 * np.pi * k * t / weekly_period))
            X.append(np.cos(2 * np.pi * k * t / weekly_period))
        return np.column_stack(X)

    exog_train = make_fourier(train_ts.index)
    exog_val   = make_fourier(val_ts.index)
    exog_test  = make_fourier(test_ts.index)

    # -----------------------------
    # Fit on train
    # -----------------------------
    model = SARIMAX(
        train_ts,
        order=order,
        seasonal_order=seasonal_order,
        exog=exog_train,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted = model.fit(disp=False, maxiter=100)

    # -----------------------------
    # Validation
    # -----------------------------
    val_pred = fitted.get_forecast(
        steps=len(val_ts),
        exog=exog_val
    ).predicted_mean

    val_mase = mase(val_ts, val_pred, train_ts)

    print("Validation MASE:", val_mase)

    # -----------------------------
    # Refit on train+val
    # -----------------------------
    train_full = pd.concat([train_ts, val_ts])
    exog_full  = np.vstack([exog_train, exog_val])

    model_full = SARIMAX(
        train_full,
        order=order,
        seasonal_order=seasonal_order,
        exog=exog_full,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted_full = model_full.fit(disp=False, maxiter=100)

    # -----------------------------
    # Test Forecast
    # -----------------------------
    test_pred = fitted_full.get_forecast(
        steps=len(test_ts),
        exog=exog_test
    ).predicted_mean

    test_rmse = calc_rmse(test_ts, test_pred)
    test_mae  = mean_absolute_error(test_ts, test_pred)
    test_mase = mase(test_ts, test_pred, train_full)

    print("\n=== SARIMAX TEST RESULTS ===")
    print("RMSE :", test_rmse)
    print("MAE  :", test_mae)
    print("MASE :", test_mase)

    return test_pred, fitted_full.resid


def analyze_sarima_residuals(residuals, lags=50):

    residuals = pd.Series(residuals).dropna()

    print("\n=== SARIMA Residual Diagnostics ===")
    print("Mean of residuals:", residuals.mean())
    print("Std  of residuals:", residuals.std())

    # ---------------------------------------------------
    # 1️⃣ Residual Time Plot
    # ---------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(residuals)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("SARIMA Residuals Over Time")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------
    # 2️⃣ Histogram + Normal Curve
    # ---------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=50, density=True, alpha=0.6)

    mu, std = residuals.mean(), residuals.std()
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, linewidth=2)

    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------
    # 3️⃣ ACF Plot
    # ---------------------------------------------------
    plt.figure(figsize=(8, 4))
    plot_acf(residuals, lags=lags)
    plt.title("Residual ACF")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------
    # 4️⃣ Ljung-Box Test
    # ---------------------------------------------------
    lb_test = acorr_ljungbox(residuals, lags=[24, 48, 72, 168], return_df=True)
    print(lb_test)
    print("\nLjung-Box Test (lag=24):")
    print(lb_test)

def sarima_grid_search(train_ts, val_ts):

    print("\n=== SARIMA GRID SEARCH (Validation MASE) ===")

    p = range(0, 3)
    d = [1]
    q = range(0, 3)

    P = range(0, 2)
    D = [1]
    Q = range(0, 2)
    s = 24

    results = []

    for (pi, di, qi), (Pi, Di, Qi) in product(
        product(p, d, q),
        product(P, D, Q)
    ):

        try:
            model = SARIMAX(
                train_ts,
                order=(pi, di, qi),
                seasonal_order=(Pi, Di, Qi, s),
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            fitted = model.fit(disp=False)

            forecast = fitted.get_forecast(steps=len(val_ts))
            pred = forecast.predicted_mean
            pred.index = val_ts.index

            aligned = pd.concat([val_ts, pred], axis=1).dropna()

            val_mase = mase(
                aligned.iloc[:, 0],
                aligned.iloc[:, 1],
                train_ts,
                season_lag=24
            )

            print(
                f"({pi},{di},{qi}) x ({Pi},{Di},{Qi},24) "
                f"→ MASE={val_mase:.4f}"
            )

            results.append({
                "order": (pi,di,qi),
                "seasonal": (Pi,Di,Qi,24),
                "mase": val_mase
            })

        except:
            continue

    results_df = pd.DataFrame(results).sort_values("mase")

    print("\n=== BEST MODELS ===")
    print(results_df.head())

    return results_df

def sarima_window_sensitivity(
        full_train_ts,
        val_ts,
        test_ts,
        order=(2,1,2),
        seasonal_order=(1,1,1,24),
        window_days_list=[30, 90, 180, 365]
):

    print("\n=== SARIMA Window Sensitivity Check ===")

    results = []

    for days in window_days_list:

        print(f"\n--- Training window: {days} days ---")

        hours = days * 24
        train_window = full_train_ts.iloc[-hours:]

        start_time = time.time()

        try:
            model = SARIMAX(
                train_window,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            fitted = model.fit(disp=False)

            runtime = time.time() - start_time

            dynamic_start = len(train_window)

            test_pred = fitted.predict(
                start=dynamic_start,
                end=dynamic_start + len(test_ts) - 1,
                dynamic=True
            )

            test_pred.index = test_ts.index

            mae = mean_absolute_error(test_ts, test_pred)

            converged = fitted.mle_retvals.get("converged", False)

            print(f"MAE       : {mae:.4f}")
            print(f"Runtime   : {runtime:.2f} sec")
            print(f"Converged : {converged}")

            results.append({
                "days": days,
                "mae": mae,
                "runtime_sec": runtime,
                "converged": converged
            })

        except Exception as e:
            print("Failed:", e)

    results_df = pd.DataFrame(results)

    print("\n=== Summary ===")
    print(results_df)

    return results_df

# ============================================================
# LSTM
# ============================================================

def make_windows(X, y):
    """
    Creates multi-step windows:
    Input  : LOOKBACK hours
    Output : next HORIZON hours (vector)
    """
    Xw, yw = [], []
    n = len(y)

    for i in range(n - LOOKBACK - HORIZON + 1):
        Xw.append(X[i:i + LOOKBACK])
        yw.append(y[i + LOOKBACK : i + LOOKBACK + HORIZON])

    return np.array(Xw), np.array(yw)

def build_lstm_model(n_features):
    from tensorflow import keras
    from tensorflow.keras import layers
    import tensorflow as tf

    tf.random.set_seed(SEED)

    model = keras.Sequential([
        layers.Input(shape=(LOOKBACK, n_features)),

        layers.LSTM(
            64,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.1   # slightly reduced
        ),

        layers.LSTM(
            32,
            dropout=0.2,
            recurrent_dropout=0.1  # slightly reduced
        ),

        layers.Dense(48, activation="relu"),

        layers.Dense(HORIZON)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss=weighted_mse
    )

    return model


def weighted_mse(y_true, y_pred):
    horizon = tf.shape(y_true)[1]
    weights = tf.linspace(1.0, 1.5, horizon)  # increasing weight
    weights = tf.reshape(weights, (1, -1))
    error = tf.square(y_true - y_pred)
    weighted_error = error * weights
    return tf.reduce_mean(weighted_error)

def build_exog_for_lstm(df, include_electrical=True):
    cols = BASE_EXOG_COLS + [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "y_lag_1", "y_lag_24", "y_lag_168",
        "y_rollmean_24", "y_rollmean_168",
    ]

    if include_electrical:
        cols += [
            "Global_reactive_power",
            "Voltage",
            "Global_intensity",
            "Sub_metering_1",
            "Sub_metering_2",
            "Sub_metering_3",
        ]

    exog = df[cols].astype(float).copy()
    season_dummies = pd.get_dummies(df["season"], prefix="season").astype(float)
    season_dummies = season_dummies.drop(columns=["season_winter"], errors="ignore")

    return pd.concat([exog, season_dummies], axis=1)

def run_lstm_model(df, train_ts, val_ts, test_ts, include_electrical=True):

    # ---------------------------------------------------------
    # 1️⃣ Restrict dataframe to ONLY allowed timeline
    # ---------------------------------------------------------
    allowed_index = pd.concat([train_ts, val_ts, test_ts]).index
    df = df.loc[allowed_index].copy()

    # ---------------------------------------------------------
    # 2️⃣ Feature engineering AFTER restriction (no leakage)
    # ---------------------------------------------------------
    df = add_target_lags(df)

    X_all = build_exog_for_lstm(df, include_electrical)
    y_all = df[[TARGET]].astype(float)

    df_model = pd.concat([y_all, X_all], axis=1).dropna().sort_index()

    feature_cols = [c for c in df_model.columns if c != TARGET]

    # ---------------------------------------------------------
    # 3️⃣ Explicit split using SAME timestamps
    # ---------------------------------------------------------
    train_df = df_model[
        (df_model.index >= train_ts.index.min()) &
        (df_model.index <= train_ts.index.max())
        ]

    val_df = df_model[
        (df_model.index >= val_ts.index.min()) &
        (df_model.index <= val_ts.index.max())
        ]

    test_df = df_model[
        (df_model.index >= test_ts.index.min()) &
        (df_model.index <= test_ts.index.max())
        ]

    # ---------------------------------------------------------
    # 4️⃣ Scaling (fit ONLY on training)
    # ---------------------------------------------------------
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = X_scaler.fit_transform(train_df[feature_cols])
    y_train = y_scaler.fit_transform(train_df[[TARGET]]).reshape(-1)

    X_val = X_scaler.transform(val_df[feature_cols])
    y_val = y_scaler.transform(val_df[[TARGET]]).reshape(-1)

    X_test = X_scaler.transform(test_df[feature_cols])
    y_test = y_scaler.transform(test_df[[TARGET]]).reshape(-1)

    # ---------------------------------------------------------
    # 5️⃣ Windowing
    # ---------------------------------------------------------
    X_train_w, y_train_w = make_windows(X_train, y_train)
    X_val_w, y_val_w = make_windows(X_val, y_val)
    X_test_w, y_test_w = make_windows(X_test, y_test)

    if len(X_test_w) == 0:
        raise ValueError("Test window empty — not enough data after split.")

    # ---------------------------------------------------------
    # 6️⃣ Build model
    # ---------------------------------------------------------
    model = build_lstm_model(X_train_w.shape[2])

    from tensorflow import keras

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        min_delta=1e-4
    )

    # ---------------------------------------------------------
    # 7️⃣ Train
    # ---------------------------------------------------------
    history = model.fit(
        X_train_w, y_train_w,
        validation_data=(X_val_w, y_val_w),
        epochs=25,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    # ---------------------------------------------------------
    # 8️⃣ Predict
    # ---------------------------------------------------------
    y_pred_scaled = model.predict(X_test_w, verbose=0)

    y_pred = y_scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).reshape(y_pred_scaled.shape)

    y_true = y_scaler.inverse_transform(
        y_test_w.reshape(-1, 1)
    ).reshape(y_test_w.shape)

    # ---------------------------------------------------------
    # 9️⃣ Rebuild datetime index
    # ---------------------------------------------------------
    pred_start = LOOKBACK
    pred_index = test_df.index[pred_start: pred_start + len(y_pred)]

    y_true_df = pd.DataFrame(
        y_true,
        index=pred_index,
        columns=[f"t+{i+1}" for i in range(HORIZON)]
    )

    y_pred_df = pd.DataFrame(
        y_pred,
        index=pred_index,
        columns=[f"t+{i+1}" for i in range(HORIZON)]
    )

    print("Train max:", train_df.index.max())
    print("Val min  :", val_df.index.min())
    print("Test min :", test_df.index.min())

    return y_true_df, y_pred_df, history


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    base_dir = Path(__file__).resolve().parent
    input_file = base_dir / "household_power_consumption.txt"

    df = preprocess_household_power(input_file)
    df = add_calendar_features(df)
    df = add_cyclical_time_features(df)
    df = add_weekly_fourier(df, K=2)

    # ============================================================
    # 1️⃣ Canonical Time-Series Split (NO LEAKAGE)
    # ============================================================

    ts = df[TARGET].dropna().astype(float)
    ts = ts.asfreq("1h")

    n = len(ts)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_ts = ts.iloc[:train_end].copy()
    val_ts   = ts.iloc[train_end:val_end].copy()
    test_ts  = ts.iloc[val_end:].copy()

    # Forward fill INSIDE each split only
    train_ts = train_ts.ffill()
    val_ts   = val_ts.ffill()
    test_ts  = test_ts.ffill()

    print("Train range:", train_ts.index.min(), "→", train_ts.index.max())
    print("Val range  :", val_ts.index.min(), "→", val_ts.index.max())
    print("Test range :", test_ts.index.min(), "→", test_ts.index.max())

    # ============================================================
    # Pre-Model Diagnostics (ADF + ACF/PACF)
    # ============================================================

    run_pre_model_diagnostics(train_ts, max_lags=168)

    # ============================================================
    # 2️⃣ Seasonal Naive Baseline
    # ============================================================

    naive_pred = seasonal_naive_forecast(train_ts, test_ts)

    naive_rmse = calc_rmse(test_ts, naive_pred)
    naive_mae  = mean_absolute_error(test_ts, naive_pred)
    naive_mase = mase(test_ts, naive_pred, train_ts)

    naive_results = evaluate_model(
        test_ts,
        naive_pred,
        train_ts,
        residuals=(test_ts - naive_pred),
        model_name="Seasonal Naive"
    )

    print("\n=== Seasonal Naive Baseline (y_t = y_{t-24}) ===")
    print("RMSE :", naive_rmse)
    print("MAE  :", naive_mae)
    print("MASE :", naive_mase)

    # ============================================================
    # 3️⃣ SARIMAX
    # ============================================================

    sarima_pred, sarima_resid = run_sarimax_dual_seasonal(
        train_ts=train_ts,
        val_ts=val_ts,
        test_ts=test_ts,
        order=(2, 1, 2),
        seasonal_order=(1,1,1,24),
        weekly_period=168,
        fourier_k=5
    )

    analyze_sarima_residuals(sarima_resid)

    window_results = sarima_window_sensitivity(
        full_train_ts=train_ts,
        val_ts=val_ts,
        test_ts=test_ts,
        order=(2, 1, 2),
        seasonal_order=(1, 1, 1, 24)
    )

    sarima_results = evaluate_model(
        test_ts,
        sarima_pred,
        pd.concat([train_ts, val_ts]),
        residuals=sarima_resid,
        model_name="SARIMA"
    )

    # ============================================================
    # 4️⃣ LSTM
    # ============================================================

    train_full = pd.concat([train_ts, val_ts])

    lstm_true_multi, lstm_pred_multi, history = run_lstm_model(
        df,
        train_ts,
        val_ts,
        test_ts,
        include_electrical=True
    )

    rmse_by_horizon = []
    mae_by_horizon = []
    mase_by_horizon = []

    for i in range(HORIZON):
        col = f"t+{i + 1}"

        rmse_by_horizon.append(
            calc_rmse(lstm_true_multi[col], lstm_pred_multi[col])
        )

        mae_by_horizon.append(
            mean_absolute_error(lstm_true_multi[col], lstm_pred_multi[col])
        )

        mase_by_horizon.append(
            mase(
                lstm_true_multi[col].values,
                lstm_pred_multi[col].values,
                train_full,
                season_lag=24
            )
        )

    rmse_avg_24h = np.mean(rmse_by_horizon)
    mae_avg_24h = np.mean(mae_by_horizon)
    mase_avg_24h = np.mean(mase_by_horizon)

    rmse_t24 = rmse_by_horizon[-1]
    mae_t24 = mae_by_horizon[-1]
    mase_t24 = mase_by_horizon[-1]

    lstm_results = evaluate_model(
        lstm_true_multi["t+24"],
        lstm_pred_multi["t+24"],
        train_full,
        residuals=(lstm_true_multi["t+24"] - lstm_pred_multi["t+24"]),
        model_name="LSTM (t+24)"
    )

    print("\n=== 24-Hour Multi-Step LSTM Results ===")
    print(f"Average RMSE (1-24h): {rmse_avg_24h:.4f}")
    print(f"Average MAE  (1-24h): {mae_avg_24h:.4f}")
    print(f"Average MASE (1-24h): {mase_avg_24h:.4f}")
    print(f"RMSE at t+24       : {rmse_t24:.4f}")
    print(f"MAE  at t+24       : {mae_t24:.4f}")
    print(f"MASE at t+24       : {mase_t24:.4f}")
    plt.figure(figsize=(10, 5))

    plt.plot(range(1, HORIZON + 1), rmse_by_horizon, marker="o", label="RMSE")
    plt.plot(range(1, HORIZON + 1), mae_by_horizon, marker="s", label="MAE")

    plt.xlabel("Forecast horizon (hours ahead)")
    plt.ylabel("Error")
    plt.title("Multi-Step Forecast Error vs Horizon")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # -------------------------------
    # Clean Raw Training Curve
    # -------------------------------
    train_loss = np.array(history.history["loss"])
    val_loss = np.array(history.history["val_loss"])

    plt.figure(figsize=(8, 4))

    plt.plot(train_loss, linewidth=2, label="Training Loss")
    plt.plot(val_loss, linewidth=2, label="Validation Loss")

    # Tight y-axis limits (removes empty space)
    y_min = min(val_loss.min(), train_loss.min())
    y_max = max(val_loss.max(), train_loss.max())
    plt.ylim(y_min * 0.98, y_max * 1.02)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Training Curve (Raw Loss)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ============================================================
    # Final Model Comparison Box
    # ============================================================

    results_df = pd.DataFrame([
        naive_results,
        sarima_results,
        lstm_results
    ])

    results_df = results_df.set_index("Model")

    print("\n" + "=" * 70)
    print("FINAL MODEL DIAGNOSTIC COMPARISON")
    print("=" * 70)
    print(results_df.round(4))
    print("=" * 70)
