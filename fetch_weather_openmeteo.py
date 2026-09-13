"""
Ambil data cuaca historis (suhu, kelembapan, angin, curah hujan) dari Open-Meteo.
"""
import os
import requests
import pandas as pd

from config import (
    JAKARTA_LAT,
    JAKARTA_LON,
    OPENMETEO_ARCHIVE_URL,
    START_DATE,
    END_DATE,
    DATA_DIR,
)

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    params = {
        "latitude": JAKARTA_LAT,
        "longitude": JAKARTA_LON,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Asia/Jakarta",
    }

    print("Mengambil data cuaca historis dari Open-Meteo...")
    resp = requests.get(OPENMETEO_ARCHIVE_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()["hourly"]

    df = pd.DataFrame(data)
    df.rename(columns={"time": "datetime"}, inplace=True)

    out_path = os.path.join(DATA_DIR, "weather_jakarta.csv")
    df.to_csv(out_path, index=False)
    print(f"Selesai. {len(df)} baris disimpan ke {out_path}")


if __name__ == "__main__":
    main()
