import pandas as pd
import os

DATA_DIR = "data"

WEATHER_PARAMS = ["temperature", "relativehumidity", "wind_speed", "wind_direction"]


def load_raw():
    df = pd.read_csv(os.path.join(DATA_DIR, "openaq_jakarta.csv"))
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["day"] = df["date"].dt.date
    return df


def build_daily_dataset(df):
    # Rata-rata harian per lokasi per parameter
    daily = df.groupby(["day", "parameter"])["value"].mean().unstack()

    # Rename supaya jelas
    daily = daily.rename(
        columns={
            "pm25": "pm25",
            "temperature": "temperature_c",
            "relativehumidity": "humidity_pct",
            "wind_speed": "wind_speed",
            "wind_direction": "wind_direction",
        }
    )

    keep_cols = [c for c in ["pm25", "temperature_c", "humidity_pct", "wind_speed", "wind_direction"] if c in daily.columns]
    daily = daily[keep_cols].reset_index()
    daily["day"] = pd.to_datetime(daily["day"])
    daily = daily.sort_values("day")
    return daily


def aggregate_weather_daily():
    """Agregasi cuaca Open-Meteo (per jam) jadi harian."""
    w = pd.read_csv(os.path.join(DATA_DIR, "weather_jakarta.csv"))
    w["datetime"] = pd.to_datetime(w["datetime"])
    w["day"] = w["datetime"].dt.date

    daily_w = w.groupby("day").agg(
        temperature_c=("temperature_2m", "mean"),
        humidity_pct=("relative_humidity_2m", "mean"),
        precipitation_mm=("precipitation", "sum"),
        wind_speed=("wind_speed_10m", "mean"),
        wind_direction=("wind_direction_10m", "mean"),
        surface_pressure=("surface_pressure", "mean"),
    ).reset_index()
    daily_w["day"] = pd.to_datetime(daily_w["day"])
    return daily_w


def main():
    df = load_raw()

    # Stasiun online di periode berbeda-beda (ada yang mulai 2016, ada yang 2023).
    # Fokus ke periode terbaru dimana beberapa stasiun overlap, biar tren harian
    # merepresentasikan kondisi Jakarta pada rentang waktu yang sama, bukan
    # gado-gado dari era yang berbeda.
    cutoff = pd.Timestamp("2023-01-01", tz="UTC")
    df = df[df["date"] >= cutoff]

    daily = build_daily_dataset(df)

    # Hanya simpan baris yang punya PM2.5 (target utama). Buang kolom cuaca
    # co-located lama - akan diganti dengan data Open-Meteo yang jauh lebih lengkap.
    pm25_daily = daily[["day", "pm25"]].dropna(subset=["pm25"])

    # Buang outlier ekstrem yang kemungkinan besar error sensor (>500 ug/m3
    # sangat jarang terjadi bahkan saat kabut asap parah)
    pm25_daily = pm25_daily[pm25_daily["pm25"] <= 500]

    # Gabungkan dengan cuaca Open-Meteo (lengkap, tanpa gap, tahun 2024)
    weather_daily = aggregate_weather_daily()
    merged = pm25_daily.merge(weather_daily, on="day", how="left")

    out_path = os.path.join(DATA_DIR, "daily_merged.csv")
    merged.to_csv(out_path, index=False)
    print(f"Tersimpan {len(merged)} baris ke {out_path}")
    print(f"Baris dengan data cuaca lengkap: {merged['temperature_c'].notna().sum()}")
    print(merged.describe())


if __name__ == "__main__":
    main()
