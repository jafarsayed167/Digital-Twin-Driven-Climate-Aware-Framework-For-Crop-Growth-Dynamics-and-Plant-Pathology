"""
RICE CROP GROWTH — Guntur, Andhra Pradesh
==========================================
LOCATION FIXED: Guntur, AP (lat=16.3067, lon=80.4365)
Data source: OpenWeatherMap API → climate_data.csv (Guntur only)

GDD METHOD (ICAR/FAO Standard):
  GDD = max(0, daily_avg_temp − 10°C)
  Rice needs ~1750–2200 GDD to mature depending on variety.
  Each day contributes GDD based on actual Guntur temperature.

WHAT THE GRAPHS SHOW:
  - Growth Index: 0–100% daily score = temp × water × solar combined effect
  - Biomass: accumulated plant material (g/m²) over time
  - Leaf Area Index: how much leaf area per ground area (peak at heading)
  - Estimated Yield: increases as crop matures, peaks at maturity

SOLID vs DASHED:
  - Solid = days that already happened (actual Guntur climate data)
  - Dashed = future projected days (estimated from recent trends)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json, os

# ================================================================
# GUNTUR FIXED — do not change
# ================================================================
GUNTUR = {"city":"Guntur","state":"Andhra Pradesh","lat":16.3067,"lon":80.4365}

VARIETIES = {
    "MTU 1010 (Rajendra)":    {"duration":125,"yield_pot":7000,"temp_opt":29,"gdd_total":1800,"blast_res":"High",    "drought_tol":"Moderate","desc":"Short duration, high yield, blast resistant"},
    "Swarna (MTU7029)":       {"duration":145,"yield_pot":6500,"temp_opt":28,"gdd_total":2100,"blast_res":"Moderate","drought_tol":"Low",      "desc":"Most popular in AP/Telangana, fine grain"},
    "BPT 5204 (Samba Masuri)":{"duration":150,"yield_pot":5500,"temp_opt":27,"gdd_total":2200,"blast_res":"Moderate","drought_tol":"Low",      "desc":"Fine grain, high market demand"},
    "IR 64":                   {"duration":110,"yield_pot":6000,"temp_opt":28,"gdd_total":1600,"blast_res":"High",    "drought_tol":"Moderate","desc":"Disease resistant, widely grown"},
    "Tellahamsa":              {"duration":130,"yield_pot":5800,"temp_opt":27,"gdd_total":1900,"blast_res":"Low",     "drought_tol":"Low",      "desc":"Traditional AP variety"},
    "Other":                   {"duration":120,"yield_pot":5500,"temp_opt":28,"gdd_total":1750,"blast_res":"Moderate","drought_tol":"Moderate","desc":"Generic rice variety"},
}

# Growth stages as % of total GDD
STAGES = [
    (0.00, 0.06, "🌱 Germination"),
    (0.06, 0.18, "🌿 Seedling"),
    (0.18, 0.35, "🌾 Tillering"),
    (0.35, 0.48, "🌾 Stem Elongation"),
    (0.48, 0.60, "🌸 Booting"),
    (0.60, 0.68, "🌸 Heading"),
    (0.68, 0.80, "🌼 Flowering"),
    (0.80, 1.00, "🍚 Grain Filling → Maturity"),
]

# Stage multiplier for biomass accumulation
STAGE_MULT = [0.4, 1.0, 2.0, 2.8, 3.2, 2.8, 2.2, 3.8]

FERT  = {"None":0.70,"Low (1 bag NPK)":0.85,"Medium (2 bags)":1.00,"High (3+ bags)":1.12,"Organic only":0.90}
IRR   = {"Flood":1.00,"Drip":1.15,"Sprinkler":1.10,"Rainfed":0.85}

CONFIG_FILE = "data/crop_config.json"


def load_config():
    base = {
        "sowing_date":"2026-04-01","variety":"MTU 1010 (Rajendra)",
        "field_size_acres":1.0,"irrigation":"Flood","fertilizer":"Medium (2 bags)",
        "transplanted":True,"seedling_age":25,
        # GUNTUR FIXED
        "location":GUNTUR["city"]+" "+GUNTUR["state"],
        "lat":GUNTUR["lat"],"lon":GUNTUR["lon"],
    }
    if os.path.exists(CONFIG_FILE):
        try:
            saved=json.load(open(CONFIG_FILE)); base.update(saved)
        except: pass
    # Always enforce Guntur
    base["lat"]=GUNTUR["lat"]; base["lon"]=GUNTUR["lon"]
    base["location"]=GUNTUR["city"]+", "+GUNTUR["state"]
    return base


def save_config(cfg):
    cfg["lat"]=GUNTUR["lat"]; cfg["lon"]=GUNTUR["lon"]
    cfg["location"]=GUNTUR["city"]+", "+GUNTUR["state"]
    os.makedirs("data",exist_ok=True)
    with open(CONFIG_FILE,"w") as f: json.dump(cfg,f,indent=2)


def get_stage(cum_gdd, gdd_total):
    pct = cum_gdd / gdd_total
    for i,(s,e,name) in enumerate(STAGES):
        if s <= pct < e: return i,name
    return len(STAGES)-1, STAGES[-1][2]


# ================================================================
# MAIN SIMULATION
# ================================================================
def simulate_growth(climate_csv="data/climate_data.csv", config=None):
    if config is None:
        config = load_config()

    vi  = VARIETIES.get(config["variety"], VARIETIES["Other"])
    dur = vi["duration"]; gdd_t=vi["gdd_total"]; topt=vi["temp_opt"]
    fm  = FERT.get(config.get("fertilizer","Medium (2 bags)"),1.0)
    im  = IRR.get(config.get("irrigation","Flood"),1.0)

    # Load Guntur climate data
    df=pd.read_csv(climate_csv); df["Date"]=pd.to_datetime(df["Date"])
    df=df.sort_values("Date").reset_index(drop=True)
    df["_d"]=df["Date"].dt.date
    daily=df.groupby("_d").agg({
        "Temperature":"mean","Humidity":"mean","Rainfall":"sum",
        "WindSpeed":"mean","SolarRadiation":"mean"
    }).reset_index()
    daily.columns=["Date","Temperature","Humidity","Rainfall","WindSpeed","SolarRadiation"]
    daily["Date"]=pd.to_datetime(daily["Date"])

    sowing  = datetime.strptime(config["sowing_date"],"%Y-%m-%d")
    today   = datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)
    cur_day = max(0,(today-sowing).days)

    rows=[]; biomass=0.; lai=0.05; tillers=1; cum_gdd=0.

    for d in range(dur+1):
        sim_date = sowing + timedelta(days=d)

        # Get actual Guntur climate for this date
        dc = daily[daily["Date"].dt.date == sim_date.date()]
        if len(dc)>0:
            rc=dc.iloc[0]; is_actual=True
        else:
            rc=daily.iloc[d%len(daily)]; is_actual=False

        temp=float(rc["Temperature"]); hum=float(rc["Humidity"])
        rain=float(rc["Rainfall"]);    solar=float(rc["SolarRadiation"])

        # GDD (ICAR base temp 10°C)
        gdd_today = max(0., temp-10.)
        cum_gdd  += gdd_today

        # Growth factors
        if temp<12: tf=0.
        elif temp<=topt: tf=(temp-12)/(topt-12)
        else: tf=max(0., 1.-((temp-topt)/(42-topt))*.65)
        tf=float(np.clip(tf,0,1))

        water=(rain+(hum/100)*2.5); wf=float(np.clip(water/5.*im,0,1))
        sf=float(np.clip(solar/650,0,1))

        gi=round((tf*.38+wf*.35+sf*.27)*100*fm,1)
        gi=float(np.clip(gi,0,100))

        si,sname=get_stage(cum_gdd,gdd_t)
        mult=STAGE_MULT[min(si,len(STAGE_MULT)-1)]

        # Biomass — GDD driven RUE approach
        biomass += (gdd_today/18.)*sf*mult*fm*3.5
        biomass  = round(min(biomass, vi["yield_pot"]*0.25),1)

        # LAI
        if si<=2:   lai=min(lai+.12*tf*wf*fm,6.5)
        elif si==3: lai=min(lai+.05*tf,7.0)
        elif si>=7: lai=max(lai-.06,.3)
        lai=round(lai,2)

        # Tillers
        if si==2:   tillers=min(tillers+max(0,int(tf*wf*4)),30)
        elif si>=5: tillers=max(tillers-1,12)

        # Stress
        hs=bool(temp>35); cs=bool(temp<16)
        ws=bool(rain<1.5 and hum<55 and config.get("irrigation","Flood")=="Rainfed")
        dr=bool(hum>85 and 20<=temp<=32)

        # Yield (harvest index by stage)
        hi={0:0.,1:0.,2:.04,3:.08,4:.14,5:.24,6:.35,7:.45}
        ey=round(biomass*hi.get(si,.45)*.1,1)

        rows.append({
            "Day":d,"Date":sim_date.strftime("%Y-%m-%d"),"Stage":sname,
            "IsActual":is_actual,"IsCurrent":bool(d==cur_day),
            "IsPast":bool(d<cur_day),"IsFuture":bool(d>cur_day),
            "Temperature":round(temp,1),"Humidity":round(hum,1),
            "Rainfall":round(rain,1),"SolarRad":round(solar,1),
            "GDD_today":round(gdd_today,1),"CumGDD":round(cum_gdd,1),
            "TempFactor":round(tf,3),"WaterFactor":round(wf,3),"SolarFactor":round(sf,3),
            "GrowthIndex":gi,"Biomass_gm2":biomass,"LeafAreaIndex":lai,
            "TillerCount":tillers,"EstYield_kgha":ey,
            "HeatStress":hs,"ColdStress":cs,"WaterStress":ws,"DiseaseRisk":dr,
        })

    out=pd.DataFrame(rows)
    out.to_csv("data/crop_growth.csv",index=False)

    cr=out[out["IsCurrent"]==True]
    cr=cr.iloc[0] if len(cr) else out.iloc[min(cur_day,len(out)-1)]
    past=out[out["IsPast"]==True]
    stress=int(past[["HeatStress","ColdStress","WaterStress"]].any(axis=1).sum()) if len(past) else 0
    avg_gi=round(float(past["GrowthIndex"].mean()),1) if len(past) else float(cr["GrowthIndex"])

    print(f"✅ Crop Growth [Guntur,AP] | {config['variety']} | Day {cur_day}/{dur}")
    print(f"   Stage: {cr['Stage']} | CumGDD: {cr['CumGDD']} | GI: {cr['GrowthIndex']}%")
    print(f"   Biomass: {cr['Biomass_gm2']} g/m² | Yield: {cr['EstYield_kgha']} kg/ha")
    return out, cur_day, config


if __name__ == "__main__":
    simulate_growth()

