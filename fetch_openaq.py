"""
Ambil data kualitas udara (PM2.5, PM10, dll) dari OpenAQ v3 untuk area Jakarta.
"""

import os
import time
import requests
import pandas as pd

from config import JAKARTA_LAT, JAKARTA_LON, RADIUS_METERS, OPENAQ_BASE_URL, DATA_DIR


def get_headers():
    api_key = os.environ.get("OPENAQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAQ_API_KEY belum di-set. Jalankan: export OPENAQ_API_KEY='api_key_kamu'"
        )
    return {"X-API-Key": api_key}


def find_locations():
    """Cari stasiun monitoring dalam radius sekitar Jakarta."""
    url = f"{OPENAQ_BASE_URL}/locations"
    params = {
        "coordinates": f"{JAKARTA_LAT},{JAKARTA_LON}",
        "radius": RADIUS_METERS,
        "limit": 100,
    }
    resp = requests.get(url, headers=get_headers(), params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()["results"]


def get_sensor_measurements(sensor_id, date_from, date_to):
    """Ambil data historis (agregasi harian) untuk satu sensor."""
    url = f"{OPENAQ_BASE_URL}/sensors/{sensor_id}/measurements/daily"
    params = {
        "date_from": date_from,
        "date_to": date_to,
        "limit": 1000,
    }
    resp = requests.get(url, headers=get_headers(), params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()["results"]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Mencari stasiun monitoring di sekitar Jakarta...")
    locations = find_locations()
    print(f"Ditemukan {len(locations)} lokasi.")

    all_rows = []
    for loc in locations:
        loc_name = loc.get("name")
        for sensor in loc.get("sensors", []):
            sensor_id = sensor["id"]
            param = sensor["parameter"]["name"]
            print(f"  Fetching {loc_name} - {param} (sensor {sensor_id})...")
            try:
                measurements = get_sensor_measurements(
                    sensor_id, "2024-01-01", "2024-12-31"
                )
            except requests.HTTPError as e:
                print(f"    Skip (error: {e})")
                continue

            for m in measurements:
                all_rows.append(
                    {
                        "location": loc_name,
                        "parameter": param,
                        "date": m["period"]["datetimeFrom"]["utc"],
                        "value": m["value"],
                        "unit": sensor["parameter"]["units"],
                    }
                )
            time.sleep(0.5)  # sopan ke rate limit

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(DATA_DIR, "openaq_jakarta.csv")
    df.to_csv(out_path, index=False)
    print(f"Selesai. {len(df)} baris disimpan ke {out_path}")


if __name__ == "__main__":
    main()
