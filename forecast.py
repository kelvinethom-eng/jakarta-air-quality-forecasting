"""
Forecasting PM2.5 harian pakai SARIMA (baseline) dan bandingkan dengan naive
baseline (nilai kemarin = prediksi hari ini).

Output: plot actual vs predicted + metrik RMSE/MAE.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

DATA_DIR = "data"
ASSETS_DIR = "assets"
TEST_DAYS = 30  # jumlah hari terakhir yang dipakai untuk evaluasi


def load_series():
    df = pd.read_csv(os.path.join(DATA_DIR, "daily_merged.csv"), parse_dates=["day"])
    df = df.set_index("day")
    df = df.asfreq("D")

    pm25 = df["pm25"].interpolate(limit=7)

    # Ambil blok tanggal berurutan (tanpa bolong) TERPANJANG dimana PM2.5 ada,
    # supaya time series benar-benar kontinu harian - syarat SARIMA.
    valid = pm25.dropna()
    is_break = valid.index.to_series().diff().dt.days.fillna(1) > 1
    group_id = is_break.cumsum()
    longest_group = group_id.value_counts().idxmax()
    keep_index = valid[group_id == longest_group].index

    df = df.loc[keep_index]
    df["pm25"] = df["pm25"].interpolate(limit=7)

    print(f"Memakai blok kontinu terpanjang: {df.index.min().date()} s/d {df.index.max().date()} ({len(df)} hari)")
    return df


def run_univariate(train, test):
    """SARIMA tanpa variabel eksternal (baseline dari analisis sebelumnya)."""
    model = SARIMAX(
        train["pm25"], order=(2, 0, 0), seasonal_order=(1, 0, 0, 7),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    return fitted.forecast(steps=len(test))


def run_sarimax_exog(train, test, exog_cols):
    """SARIMAX dengan cuaca sebagai exogenous variable."""
    model = SARIMAX(
        train["pm25"], exog=train[exog_cols],
        order=(2, 0, 0), seasonal_order=(1, 0, 0, 7),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    return fitted.forecast(steps=len(test), exog=test[exog_cols])


def main():
    df = load_series()

    exog_cols = ["temperature_c", "humidity_pct", "wind_speed", "wind_direction"]
    has_exog = df[exog_cols].notna().all(axis=1)
    print(f"Hari dengan data cuaca lengkap dalam blok ini: {has_exog.sum()} / {len(df)}")

    if has_exog.sum() < TEST_DAYS + 60:
        print("Data cuaca terlalu sedikit dalam blok kontinu PM2.5 ini untuk SARIMAX,")
        print("lanjut evaluasi univariat saja.")
        df_exog = None
    else:
        df_exog = df[has_exog].copy()

    train, test = df[:-TEST_DAYS], df[-TEST_DAYS:]

    print(f"Data latih: {len(train)} hari, data uji: {len(test)} hari")

    # --- Baseline: naive forecast (nilai hari sebelumnya) ---
    naive_pred = test["pm25"].shift(1)
    naive_pred.iloc[0] = train["pm25"].iloc[-1]
    naive_rmse = np.sqrt(mean_squared_error(test["pm25"], naive_pred))
    naive_mae = mean_absolute_error(test["pm25"], naive_pred)

    # --- SARIMA univariat ---
    sarima_pred = run_univariate(train, test)
    sarima_rmse = np.sqrt(mean_squared_error(test["pm25"], sarima_pred))
    sarima_mae = mean_absolute_error(test["pm25"], sarima_pred)

    print("\n=== Hasil Evaluasi (30 hari terakhir) ===")
    print(f"Naive baseline        -> RMSE: {naive_rmse:.2f}, MAE: {naive_mae:.2f}")
    print(f"SARIMA univariat      -> RMSE: {sarima_rmse:.2f}, MAE: {sarima_mae:.2f}")

    sarimax_pred = None
    if df_exog is not None:
        train_e, test_e = df_exog[:-TEST_DAYS], df_exog[-TEST_DAYS:]
        if len(test_e) == TEST_DAYS:
            # Evaluasi ADIL: naive & SARIMA univariat dihitung ULANG di periode
            # yang sama dengan test SARIMAX (subset 2024), bukan pakai window
            # test utama yang beda periode.
            naive_pred_e = test_e["pm25"].shift(1)
            naive_pred_e.iloc[0] = train_e["pm25"].iloc[-1]
            naive_rmse_e = np.sqrt(mean_squared_error(test_e["pm25"], naive_pred_e))

            sarima_pred_e = run_univariate(train_e, test_e)
            sarima_rmse_e = np.sqrt(mean_squared_error(test_e["pm25"], sarima_pred_e))

            sarimax_pred = run_sarimax_exog(train_e, test_e, exog_cols)
            sarimax_rmse = np.sqrt(mean_squared_error(test_e["pm25"], sarimax_pred))
            sarimax_mae = mean_absolute_error(test_e["pm25"], sarimax_pred)

            print(f"\n=== Evaluasi terpisah di periode {test_e.index.min().date()} - {test_e.index.max().date()} (ada data cuaca) ===")
            print(f"Naive baseline        -> RMSE: {naive_rmse_e:.2f}")
            print(f"SARIMA univariat      -> RMSE: {sarima_rmse_e:.2f}")
            print(f"SARIMAX + cuaca       -> RMSE: {sarimax_rmse:.2f}, MAE: {sarimax_mae:.2f}")
            best_rmse = min(naive_rmse_e, sarima_rmse_e, sarimax_rmse)
            best_name = {naive_rmse_e: "Naive", sarima_rmse_e: "SARIMA univariat", sarimax_rmse: "SARIMAX + cuaca"}[best_rmse]
            print(f"Model terbaik di periode ini: {best_name}")
        else:
            print("Blok data cuaca lengkap tidak mencakup 30 hari terakhir data PM2.5.")

    # --- Plot 1: SARIMA univariat vs Naive (test window utama) ---
    plt.figure(figsize=(12, 5))
    plt.plot(train.index[-60:], train["pm25"].values[-60:], label="Data latih (60 hari terakhir)", color="gray")
    plt.plot(test.index, test["pm25"].values, label="Actual", color="black", linewidth=2)
    plt.plot(test.index, sarima_pred, label="Prediksi SARIMA (univariat)", color="#c0392b", linestyle="--")
    plt.plot(test.index, naive_pred, label="Prediksi Naive", color="#2980b9", linestyle=":")
    plt.title(f"Forecast PM2.5: SARIMA vs Naive Baseline ({test.index.min().date()} - {test.index.max().date()})")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(ASSETS_DIR, exist_ok=True)
    plt.savefig(os.path.join(ASSETS_DIR, "forecast_comparison.png"), dpi=120)
    plt.close()
    print(f"\nPlot univariat tersimpan di {ASSETS_DIR}/forecast_comparison.png")

    # --- Plot 2: SARIMAX+cuaca vs SARIMA vs Naive (test window periode 2024, ada cuaca) ---
    if sarimax_pred is not None and len(test_e) == TEST_DAYS:
        plt.figure(figsize=(12, 5))
        plt.plot(train_e.index[-60:], train_e["pm25"].values[-60:], label="Data latih (60 hari terakhir)", color="gray")
        plt.plot(test_e.index, test_e["pm25"].values, label="Actual", color="black", linewidth=2)
        plt.plot(test_e.index, sarima_pred_e, label="Prediksi SARIMA (univariat)", color="#c0392b", linestyle="--")
        plt.plot(test_e.index, naive_pred_e, label="Prediksi Naive", color="#2980b9", linestyle=":")
        plt.plot(test_e.index, sarimax_pred, label="Prediksi SARIMAX + cuaca", color="#27ae60", linestyle="-.")
        plt.title(f"Forecast PM2.5 dengan Exogenous Cuaca ({test_e.index.min().date()} - {test_e.index.max().date()})")
        plt.ylabel("PM2.5 (µg/m³)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(ASSETS_DIR, "forecast_exog_comparison.png"), dpi=120)
        plt.close()
        print(f"Plot exogenous tersimpan di {ASSETS_DIR}/forecast_exog_comparison.png")


if __name__ == "__main__":
    main()
