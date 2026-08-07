import pandas as pd
import numpy as np
from datetime import datetime

# ================================================================
# PLANT PATHOLOGY ENGINE — Digital Twin
# Parameters: Leaf Color, Leaf Spots, Chlorophyll Index,
#             Disease Probability, Fungal Infection Rate
# ================================================================

# -------- LEAF COLOR INDEX (LCI) --------
# Based on temperature + humidity + solar radiation
# Green=Healthy, Yellow=Stress, Brown=Diseased
def compute_leaf_color_index(temp, humidity, solar, nitrogen):
    """
    Returns:
        lci_score: 0-100 (100=perfectly green)
        color_status: "Green / Healthy", "Yellow / Stressed", "Brown / Diseased"
        hex_color: visual hex
    """
    # Nitrogen deficiency → yellowing
    n_factor = np.clip(nitrogen / 80, 0.5, 1.0)

    # Heat stress → browning
    heat_penalty = max(0, (temp - 35) * 3)

    # Low solar → pale color
    solar_factor = np.clip(solar / 600, 0.6, 1.0)

    # High humidity → dark spots (not yellow)
    hum_factor = 1.0 if humidity < 90 else 0.88

    lci_score = round(
        100 * n_factor * solar_factor * hum_factor - heat_penalty,
        1
    )
    lci_score = float(np.clip(lci_score, 0, 100))

    if lci_score >= 70:
        color_status = "🟢 Green / Healthy"
        hex_color    = "#4ade80"
    elif lci_score >= 45:
        color_status = "🟡 Yellow / Stressed"
        hex_color    = "#fbbf24"
    elif lci_score >= 20:
        color_status = "🟠 Pale / Moderate Stress"
        hex_color    = "#fb923c"
    else:
        color_status = "🟤 Brown / Diseased"
        hex_color    = "#a16207"

    return lci_score, color_status, hex_color


# -------- LEAF SPOT DETECTION (LSD) --------
# Based on humidity, rain, temperature cycles
def compute_leaf_spot_index(humidity, rain, temp, wind):
    """
    Returns:
        spot_score: 0-100 (100 = max spotting)
        spot_level: severity label
        spot_type: likely pathogen type
    """
    # Moisture on leaf surface
    wetness = (humidity / 100) * 0.5 + (min(rain, 20) / 20) * 0.5

    # Optimal fungal temp 20–30°C
    if 20 <= temp <= 30:
        temp_fav = 1.0
    elif temp < 20:
        temp_fav = (temp - 10) / 10 if temp > 10 else 0.0
    else:
        temp_fav = max(0, 1.0 - (temp - 30) / 10)

    # Low wind = spores stay on leaf
    wind_factor = max(0, 1.0 - wind / 8)

    spot_score = round(wetness * temp_fav * wind_factor * 100, 1)
    spot_score = float(np.clip(spot_score, 0, 100))

    if spot_score < 20:
        spot_level = "✅ None / Minimal"
        spot_type  = "No significant pathogen activity"
    elif spot_score < 40:
        spot_level = "🟡 Low"
        spot_type  = "Early fungal spore germination possible"
    elif spot_score < 60:
        spot_level = "🟠 Moderate"
        spot_type  = "Likely: Brown Spot / Early Blight"
    elif spot_score < 80:
        spot_level = "🔴 High"
        spot_type  = "Active: Rice Blast / Sheath Blight"
    else:
        spot_level = "🔴 Severe"
        spot_type  = "Critical: Multiple fungal diseases active"

    return spot_score, spot_level, spot_type


# -------- CHLOROPHYLL INDEX (CI) --------
# SPAD-based estimation from climate + soil
def compute_chlorophyll_index(temp, solar, nitrogen, ph, humidity):
    """
    Returns estimated SPAD chlorophyll index (0–80 typical range)
    Healthy rice: 35–55 SPAD
    """
    # Nitrogen is primary driver of chlorophyll
    n_base = np.clip(nitrogen / 100, 0.3, 1.0) * 50

    # Solar drives photosynthesis
    solar_effect = np.clip(solar / 800, 0.5, 1.1)

    # pH affects nutrient uptake
    if 6.0 <= ph <= 7.5:
        ph_factor = 1.0
    else:
        ph_factor = 1.0 - abs(ph - 6.75) * 0.08

    # Heat stress reduces chlorophyll
    heat_penalty = max(0, (temp - 35) * 0.5)

    # High humidity can cause slight boost (less evaporation)
    hum_factor = 1.02 if 60 <= humidity <= 85 else 0.97

    spad = round(n_base * solar_effect * ph_factor * hum_factor - heat_penalty, 1)
    spad = float(np.clip(spad, 5, 80))

    if spad >= 45:
        ci_status = "🟢 Excellent (High Photosynthesis)"
    elif spad >= 35:
        ci_status = "🟢 Good (Normal Range)"
    elif spad >= 25:
        ci_status = "🟡 Moderate (Mild Deficiency)"
    elif spad >= 15:
        ci_status = "🟠 Low (Nitrogen Deficient)"
    else:
        ci_status = "🔴 Critical (Severe Chlorosis)"

    return spad, ci_status


# -------- DISEASE PROBABILITY (DP) --------
# Bayesian-style weighted probability
def compute_disease_probability(temp, humidity, rain, wind, lci, spot_score):
    """
    Returns overall disease outbreak probability 0-100%
    """
    # Environmental conduciveness
    env_score = 0.0

    # Temperature in fungal range 20-32°C
    if 20 <= temp <= 32:
        env_score += 30 * (1.0 - abs(temp - 26) / 6)

    # High humidity is biggest driver
    if humidity > 80:
        env_score += 35 * ((humidity - 80) / 20)

    # Rainfall creates wet conditions
    if rain > 5:
        env_score += 20 * min(rain / 20, 1.0)

    # Low wind = spores accumulate
    if wind < 3:
        env_score += 10 * (1.0 - wind / 3)

    # Leaf health modifies probability
    leaf_penalty = (100 - lci) * 0.05
    spot_contrib = spot_score * 0.15

    dp = round(min(env_score + leaf_penalty + spot_contrib, 100), 1)

    if dp < 20:
        dp_level = "✅ Very Low"
        dp_color = "#4ade80"
        dp_action = "No intervention needed. Continue regular monitoring."
    elif dp < 40:
        dp_level = "🟡 Low"
        dp_color = "#86efac"
        dp_action = "Preventive fungicide spray recommended within 7 days."
    elif dp < 60:
        dp_level = "🟠 Moderate"
        dp_color = "#fbbf24"
        dp_action = "Apply curative fungicide within 3 days. Improve drainage."
    elif dp < 80:
        dp_level = "🔴 High"
        dp_color = "#f97316"
        dp_action = "Immediate fungicide application needed. Scout field daily."
    else:
        dp_level = "🔴 Critical"
        dp_color = "#ef4444"
        dp_action = "Emergency treatment required. Consider crop insurance."

    return dp, dp_level, dp_color, dp_action


# -------- FUNGAL INFECTION RATE (FIR) --------
# Models spore germination and infection spread rate
def compute_fungal_infection_rate(temp, humidity, rain, days_wet):
    """
    Returns:
        fir: infection rate % per day
        spore_viability: % (how viable spores are)
        spread_risk: spread risk level
    """
    # Spore germination requires: 20-30°C + >90% RH or wet surface
    if 20 <= temp <= 30 and humidity >= 90:
        germination = 1.0
    elif 18 <= temp <= 33 and humidity >= 80:
        germination = 0.6
    elif humidity >= 70:
        germination = 0.3
    else:
        germination = 0.1

    # Wet days multiply infection
    wet_multiplier = min(1.0 + days_wet * 0.15, 3.0)

    # Rainfall disperses spores
    rain_spread = np.clip(rain / 10, 0, 1.5)

    fir = round(germination * wet_multiplier * rain_spread * 20, 1)
    fir = float(np.clip(fir, 0, 100))

    # Spore viability
    if 20 <= temp <= 28:
        spore_viability = round(85 + humidity * 0.1, 1)
    elif temp > 35:
        spore_viability = round(max(10, 85 - (temp - 35) * 8), 1)
    else:
        spore_viability = round(max(20, 70 - abs(temp - 24) * 3), 1)
    spore_viability = float(np.clip(spore_viability, 0, 100))

    if fir < 10:
        spread_risk = "✅ Negligible"
    elif fir < 25:
        spread_risk = "🟡 Low Spread"
    elif fir < 50:
        spread_risk = "🟠 Moderate Spread"
    elif fir < 75:
        spread_risk = "🔴 Rapid Spread"
    else:
        spread_risk = "🔴 Epidemic Risk"

    return fir, spore_viability, spread_risk


# ================================================================
# MAIN RUN FUNCTION
# ================================================================
def run_pathology(climate_csv="data/climate_data.csv",
                  soil_csv="data/soil_data.csv"):

    df = pd.read_csv(climate_csv)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Use recent 3h average (last 8 rows ~ 24h)
    recent  = df.tail(8)
    temp    = float(recent["Temperature"].mean())
    hum     = float(recent["Humidity"].mean())
    rain    = float(recent["Rainfall"].sum())
    wind    = float(recent["WindSpeed"].mean())
    solar   = float(recent["SolarRadiation"].mean()) if "SolarRadiation" in recent else 500.0

    # Days with humidity > 80 (wet days estimate)
    days_wet = int((df["Humidity"] > 80).sum() // 8)

    # Soil
    try:
        soil     = pd.read_csv(soil_csv).iloc[0]
        nitrogen = float(soil.get("Nitrogen",  72))
        ph       = float(soil.get("pH",        6.5))
    except Exception:
        nitrogen, ph = 72.0, 6.5

    # Compute all indices
    lci, lci_status, lci_hex = compute_leaf_color_index(temp, hum, solar, nitrogen)
    spot, spot_level, spot_type = compute_leaf_spot_index(hum, rain, temp, wind)
    spad, ci_status = compute_chlorophyll_index(temp, solar, nitrogen, ph, hum)
    dp, dp_level, dp_color, dp_action = compute_disease_probability(temp, hum, rain, wind, lci, spot)
    fir, spore_v, spread_risk = compute_fungal_infection_rate(temp, hum, rain, days_wet)

    result = {
        "Timestamp":           datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Temperature":         round(temp, 1),
        "Humidity":            round(hum, 1),
        "Rainfall":            round(rain, 1),

        "LeafColorIndex":      lci,
        "LeafColorStatus":     lci_status,
        "LeafColorHex":        lci_hex,

        "LeafSpotScore":       spot,
        "LeafSpotLevel":       spot_level,
        "LeafSpotType":        spot_type,

        "ChlorophyllSPAD":     spad,
        "ChlorophyllStatus":   ci_status,

        "DiseaseProbability":  dp,
        "DiseaseLevel":        dp_level,
        "DiseaseColor":        dp_color,
        "DiseaseAction":       dp_action,

        "FungalInfectionRate": fir,
        "SporeViability":      spore_v,
        "SpreadRisk":          spread_risk,
    }

    pd.DataFrame([result]).to_csv("data/plant_pathology.csv", index=False)

    print("✅ Plant Pathology Analysis Complete")
    print(f"   🍃 Leaf Color Index   : {lci} — {lci_status}")
    print(f"   🔴 Leaf Spot Score    : {spot} — {spot_level}")
    print(f"   🌿 Chlorophyll (SPAD) : {spad} — {ci_status}")
    print(f"   🦠 Disease Probability: {dp}% — {dp_level}")
    print(f"   🍄 Fungal Infect Rate : {fir}%/day — {spread_risk}")
    print(f"   💡 Action             : {dp_action}")

    return result


if __name__ == "__main__":
    run_pathology()
