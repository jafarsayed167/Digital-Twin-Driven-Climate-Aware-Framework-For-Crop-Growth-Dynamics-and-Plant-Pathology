import requests
import pandas as pd
import time

lat = 16.3067
lon = 80.4365

# ============================================================
# SOURCE 1: Open-Meteo Soil Temperature & Moisture API
# Free, no API key needed, very reliable
# Docs: https://open-meteo.com/en/docs/soil-temperature-api
# ============================================================

def try_open_meteo():
    print("🔄 Trying Open-Meteo Soil API...")
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=soil_temperature_0cm,soil_temperature_6cm,"
        f"soil_moisture_0_to_1cm,soil_moisture_1_to_3cm,"
        f"soil_moisture_3_to_9cm"
        f"&forecast_days=1"
        f"&timezone=Asia/Kolkata"
    )
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            hourly = data["hourly"]

            # Take latest available value (last non-null)
            def latest(lst):
                vals = [v for v in lst if v is not None]
                return round(vals[-1], 2) if vals else None

            soil_temp_surface = latest(hourly["soil_temperature_0cm"])
            soil_temp_6cm     = latest(hourly["soil_temperature_6cm"])
            moisture_0_1      = latest(hourly["soil_moisture_0_to_1cm"])
            moisture_1_3      = latest(hourly["soil_moisture_1_to_3cm"])
            moisture_3_9      = latest(hourly["soil_moisture_3_to_9cm"])

            # Average moisture across layers → convert to % (m³/m³ * 100)
            moisture_vals = [v for v in [moisture_0_1, moisture_1_3, moisture_3_9] if v is not None]
            avg_moisture = round((sum(moisture_vals) / len(moisture_vals)) * 100, 1) if moisture_vals else None

            if avg_moisture is not None:
                df = pd.DataFrame([{
                    "SoilMoisture":  avg_moisture,
                    "SoilTemp_0cm":  soil_temp_surface,
                    "SoilTemp_6cm":  soil_temp_6cm,
                    # Open-Meteo does not provide NPK — use Guntur regional values
                    "pH":            6.5,
                    "Nitrogen":      72.0,
                    "Phosphorus":    46.0,
                    "Potassium":     63.0,
                    "Source":        "Open-Meteo"
                }])
                print(f"✅ Open-Meteo: Moisture={avg_moisture}%, SoilTemp={soil_temp_surface}°C")
                return df
    except Exception as e:
        print(f"⚠ Open-Meteo failed: {e}")
    return None


# ============================================================
# SOURCE 2: NASA POWER API
# Free, no API key, NASA satellite-based soil/climate data
# Docs: https://power.larc.nasa.gov/
# ============================================================

def try_nasa_power():
    print("🔄 Trying NASA POWER API...")
    from datetime import datetime, timedelta
    today     = datetime.now().strftime("%Y%m%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=GWETROOT,GWETPROF,GWETTOP,T2M,PRECTOTCORR"
        f"&community=AG"
        f"&longitude={lon}&latitude={lat}"
        f"&start={yesterday}&end={today}"
        f"&format=JSON"
    )
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            props = data["properties"]["parameter"]

            def last_val(param_dict):
                vals = list(param_dict.values())
                valid = [v for v in vals if v != -999.0]
                return round(valid[-1], 2) if valid else None

            gwet_top  = last_val(props.get("GWETTOP",  {}))   # Surface soil wetness 0-1
            gwet_root = last_val(props.get("GWETROOT", {}))   # Root zone wetness 0-1

            if gwet_top is not None:
                moisture_pct = round(gwet_top * 100, 1)       # Convert to %

                df = pd.DataFrame([{
                    "SoilMoisture":  moisture_pct,
                    "SoilTemp_0cm":  None,
                    "SoilTemp_6cm":  None,
                    "pH":            6.5,
                    "Nitrogen":      72.0,
                    "Phosphorus":    46.0,
                    "Potassium":     63.0,
                    "Source":        "NASA POWER"
                }])
                print(f"✅ NASA POWER: Moisture={moisture_pct}%")
                return df
    except Exception as e:
        print(f"⚠ NASA POWER failed: {e}")
    return None


# ============================================================
# SOURCE 3: SoilGrids (original — kept as 3rd fallback)
# ============================================================

def try_soilgrids():
    print("🔄 Trying SoilGrids API...")
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lat={lat}&lon={lon}"
    for i in range(2):
        try:
            r = requests.get(url, timeout=12)
            if r.status_code == 200:
                data   = r.json()
                layers = data["properties"]["layers"]
                raw    = {}
                for layer in layers:
                    raw[layer["name"]] = layer["depths"][0]["values"]["mean"]

                df = pd.DataFrame([{
                    "SoilMoisture":  round(raw.get("soc", 1.2), 2),
                    "SoilTemp_0cm":  None,
                    "SoilTemp_6cm":  None,
                    "pH":            round(raw.get("phh2o", 65) / 10, 1),
                    "Nitrogen":      round(raw.get("nitrogen", 70), 1),
                    "Phosphorus":    round(raw.get("p", 46), 1),
                    "Potassium":     round(raw.get("k", 63), 1),
                    "Source":        "SoilGrids"
                }])
                print("✅ SoilGrids: Data fetched")
                return df
        except Exception as e:
            print(f"⚠ SoilGrids attempt {i+1}: {e}")
            time.sleep(2)
    return None


# ============================================================
# FALLBACK: Guntur Region Realistic Values
# Based on ICAR / Andhra Pradesh Agricultural Dept data
# Black cotton soil + red loamy soil — Guntur district
# ============================================================

def use_fallback():
    print("⚠ All APIs failed. Using Guntur region standard values...")
    df = pd.DataFrame([{
        "SoilMoisture":  48.0,   # % — typical Guntur black cotton soil
        "SoilTemp_0cm":  28.5,   # °C — avg surface temp Guntur
        "SoilTemp_6cm":  26.0,   # °C
        "pH":            6.8,    # Slightly acidic — ideal for rice
        "Nitrogen":      72.0,   # kg/ha — moderate
        "Phosphorus":    46.0,   # kg/ha
        "Potassium":     63.0,   # kg/ha
        "Source":        "Fallback (Guntur Regional)"
    }])
    return df


# ============================================================
# MAIN — Try APIs in order
# ============================================================

df = try_open_meteo()

if df is None:
    df = try_nasa_power()

if df is None:
    df = try_soilgrids()

if df is None:
    df = use_fallback()

df.to_csv("data/soil_data.csv", index=False)
print(f"\n✅ Soil data saved → Source: {df['Source'].iloc[0]}")
print(df.to_string(index=False))
