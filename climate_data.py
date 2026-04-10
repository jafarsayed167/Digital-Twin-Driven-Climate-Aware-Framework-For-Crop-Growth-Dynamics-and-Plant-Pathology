import requests
import pandas as pd
from datetime import datetime, timedelta
import math

API_KEY = "4e7816313caf0d3485492daceb12fd30"
city = "Guntur"

rows = []

# -------- FORECAST API --------
forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
forecast_data = requests.get(forecast_url).json()

if "list" not in forecast_data:
    print("❌ Forecast Error:", forecast_data)
    exit()

# -------- FUNCTION: SMART RAIN HANDLING --------
def get_rain(data, is_current=False):
    rain = 0

    if "rain" in data:
        if is_current:
            rain = data["rain"].get("1h", 0)
        else:
            rain = data["rain"].get("3h", 0)

    elif "weather" in data:
        weather_type = data["weather"][0]["main"].lower()

        if "drizzle" in weather_type:
            rain = 0.5
        elif "rain" in weather_type:
            rain = 2

    return rain

# -------- FUNCTION: SOLAR RADIATION ESTIMATE --------
def estimate_solar(clouds, hour):
    base = 800  # Clear sky max W/m²
    cloud_factor = 1 - (clouds / 100) * 0.75
    if 6 <= hour <= 18:
        time_factor = math.sin(math.pi * (hour - 6) / 12)
    else:
        time_factor = 0
    return round(base * cloud_factor * time_factor, 1)

# -------- PAST 5 DAYS (SIMULATED) --------
now = datetime.now()

for i in range(5):
    sample = forecast_data["list"][i]
    fake_date = now - timedelta(days=(5 - i))
    clouds = sample.get("clouds", {}).get("all", 50)

    rows.append({
        "Date": fake_date,
        "Temperature": sample["main"]["temp"],
        "Humidity": sample["main"]["humidity"],
        "WindSpeed": sample["wind"]["speed"],
        "Rainfall": get_rain(sample),
        "SolarRadiation": estimate_solar(clouds, fake_date.hour)
    })

# -------- CURRENT DATA --------
current_url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
current_data = requests.get(current_url).json()

if "main" not in current_data:
    print("❌ Current Error:", current_data)
    exit()

clouds = current_data.get("clouds", {}).get("all", 50)

rows.append({
    "Date": now,
    "Temperature": current_data["main"]["temp"],
    "Humidity": current_data["main"]["humidity"],
    "WindSpeed": current_data["wind"]["speed"],
    "Rainfall": get_rain(current_data, is_current=True),
    "SolarRadiation": estimate_solar(clouds, now.hour)
})

# -------- FUTURE 3 DAYS --------
for item in forecast_data["list"][:24]:
    dt = datetime.strptime(item["dt_txt"], "%Y-%m-%d %H:%M:%S")
    clouds = item.get("clouds", {}).get("all", 50)

    rows.append({
        "Date": item["dt_txt"],
        "Temperature": item["main"]["temp"],
        "Humidity": item["main"]["humidity"],
        "WindSpeed": item["wind"]["speed"],
        "Rainfall": get_rain(item),
        "SolarRadiation": estimate_solar(clouds, dt.hour)
    })

# -------- FINAL DATAFRAME --------
df = pd.DataFrame(rows)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(by="Date")
df.to_csv("data/climate_data.csv", index=False)

print("✅ Final Data Ready (Past + Present + Future)")