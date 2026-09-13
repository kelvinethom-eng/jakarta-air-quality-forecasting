"""
EDA: time series PM2.5, korelasi dengan cuaca, dan scatter plot.
Output disimpan sebagai PNG di folder assets/.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data"
ASSETS_DIR = "assets"


def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(DATA_DIR, "daily_merged.csv"), parse_dates=["day"])
    df = df.sort_values("day")

    # 1. Time series PM2.5
    plt.figure(figsize=(12, 4))
    plt.plot(df["day"], df["pm25"], linewidth=0.8, color="#c0392b")
    plt.axhline(55, color="orange", linestyle="--", label="Batas 'Tidak Sehat' (55 µg/m³)")
    plt.title("Tren PM2.5 Harian - Jakarta (semua stasiun OpenAQ)")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "pm25_timeseries.png"), dpi=120)
    plt.close()

    # 2. Correlation heatmap (hanya baris yang punya data cuaca lengkap)
    weather_cols = ["pm25", "temperature_c", "humidity_pct", "precipitation_mm",
                     "wind_speed", "wind_direction", "surface_pressure"]
    df_weather = df.dropna(subset=["temperature_c", "humidity_pct"])
    corr = df_weather[weather_cols].corr()

    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title(f"Korelasi PM2.5 vs Cuaca (n={len(df_weather)} hari)")
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "correlation_heatmap.png"), dpi=120)
    plt.close()

    # 3. Scatter plot PM2.5 vs variabel paling berkorelasi
    corr_with_pm25 = corr["pm25"].drop("pm25").abs().sort_values(ascending=False)
    top_var = corr_with_pm25.index[0]

    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df_weather, x=top_var, y="pm25", alpha=0.5)
    sns.regplot(data=df_weather, x=top_var, y="pm25", scatter=False, color="red")
    plt.title(f"PM2.5 vs {top_var} (korelasi = {corr['pm25'][top_var]:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "scatter_top_correlation.png"), dpi=120)
    plt.close()

    print("Insight korelasi PM2.5 dengan variabel cuaca:")
    print(corr["pm25"].drop("pm25").sort_values(key=abs, ascending=False))
    print(f"\nVariabel paling berkorelasi: {top_var}")
    print(f"3 file PNG tersimpan di folder {ASSETS_DIR}/")


if __name__ == "__main__":
    main()
