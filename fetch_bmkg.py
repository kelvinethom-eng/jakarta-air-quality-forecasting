"""
Ambil data prakiraan cuaca 3 hari ke depan dari BMKG (opsional, sifatnya real-time
forecast, bukan historis - cocok untuk fitur tambahan atau live dashboard).
Tidak butuh API key.

Cara pakai:
    python fetch_bmkg.py
"""

import os
import requests
import pandas as pd

from config import BMKG_ADM4_CODE, BMKG_FORECAST_URL, DATA_DIR


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    params = {"adm4": BMKG_ADM4_CODE}
    print("Mengambil prakiraan cuaca BMKG...")
    resp = requests.get(BMKG_FORECAST_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    rows = []
    for entry in payload.get("data", []):
        lokasi = entry.get("lokasi", {})
        for cuaca_per_hari in entry.get("cuaca", []):
            for item in cuaca_per_hari:
                rows.append(
                    {
                        "kelurahan": lokasi.get("desa"),
                        "waktu": item.get("local_datetime"),
                        "suhu_c": item.get("t"),
                        "kelembapan_persen": item.get("hu"),
                        "cuaca": item.get("weather_desc"),
                        "kecepatan_angin_kmh": item.get("ws"),
                    }
                )

    df = pd.DataFrame(rows)
    out_path = os.path.join(DATA_DIR, "bmkg_forecast_jakarta.csv")
    df.to_csv(out_path, index=False)
    print(f"Selesai. {len(df)} baris disimpan ke {out_path}")


if __name__ == "__main__":
    main()
