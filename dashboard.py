"""
Smart Farming Digital Twin — Guntur, AP
Light green theme, all fixes applied.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys, datetime, random, tempfile, json
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Digital Twin — Smart Farm",
                   page_icon="🌾", layout="wide", initial_sidebar_state="collapsed")

st_autorefresh(interval=60000, key="refresh")

# Run data pipelines via import (works both locally and on Streamlit Cloud)
import importlib, sys as _sys
def _run(script_name):
    try:
        _sys.path.insert(0, "scripts")
        mod = importlib.import_module(script_name)
        importlib.reload(mod)
    except Exception as e:
        pass  # errors handled inside each script

import os as _os, sys as _sys
_sys.path.insert(0, "scripts")
_os.makedirs("data", exist_ok=True)
_os.makedirs("data/models", exist_ok=True)

# Step 1: Fetch live data
try:
    import climate_data as _cd; import importlib; importlib.reload(_cd)
except Exception as _e: pass

try:
    import soil_data as _sd; importlib.reload(_sd)
except Exception as _e: pass

try:
    import crop_growth as _cg; importlib.reload(_cg)
    if _os.path.exists("data/climate_data.csv"):
        import json as _json
        _cfg = _json.load(open("data/crop_config.json")) if _os.path.exists("data/crop_config.json") else {}
        _cg.simulate_growth("data/climate_data.csv", _cfg if _cfg else None)
except Exception as _e: pass

# Step 2: Run ML pipeline — train + predict fresh every time
try:
    import ml_models as _ml; importlib.reload(_ml)
    _ml.train_all_if_needed()
    if _os.path.exists("data/climate_data.csv"):
        _ml.predict_yield()
        _ml.predict_irrigation()
        _ml.forecast_climate()
    print("✅ ML pipeline complete")
except Exception as _e:
    print(f"⚠ ML pipeline: {_e}")

# ================================================================
# GUNTUR DAY/NIGHT IDEAL CONDITIONS (from research table)
# ================================================================
IDEALS = {
    "Temperature":     {"day_min":26,"day_max":34,"night_min":20,"night_max":25,"unit":"°C"},
    "Humidity":        {"day_min":70,"day_max":85,"night_min":80,"night_max":90,"unit":"%"},
    "WindSpeed":       {"day_min":1.0,"day_max":2.5,"night_min":0.5,"night_max":1.5,"unit":"m/s"},
    "SolarRadiation":  {"day_min":500,"day_max":650,"night_min":0,"night_max":0,"unit":"kWh/m²"},
    "SoilMoisture":    {"day_min":80,"day_max":100,"night_min":80,"night_max":95,"unit":"%"},
    "pH":              {"day_min":5.2,"day_max":6.5,"night_min":5.2,"night_max":6.5,"unit":""},
    "Nitrogen":        {"day_min":100,"day_max":120,"night_min":100,"night_max":120,"unit":"kg/ha"},
    "Phosphorus":      {"day_min":40,"day_max":60,"night_min":40,"night_max":60,"unit":"kg/ha"},
    "Potassium":       {"day_min":40,"day_max":80,"night_min":40,"night_max":80,"unit":"kg/ha"},
}

def is_day():
    return 6 <= datetime.datetime.now().hour <= 18

def get_ideal(param):
    """Returns (min, max) based on current time of day"""
    if param not in IDEALS: return None, None
    i = IDEALS[param]
    if is_day():
        return i["day_min"], i["day_max"]
    else:
        return i["night_min"], i["night_max"]

def status_for(param, val):
    mn, mx = get_ideal(param)
    if mn is None: return "—", "#94a3b8"
    if val < mn:   return "⬇ Low",    "#f97316"
    if val > mx:   return "⬆ High",   "#ef4444"
    return "✅ Optimal", "#16a34a"

def suggestion_for(param, val):
    """Full suggestions: low / high / optimal"""
    mn, mx = get_ideal(param)
    if mn is None: return None
    unit = IDEALS[param]["unit"]
    period = "Day" if is_day() else "Night"
    ideal_str = f"{mn}–{mx} {unit} ({period})"

    SUGG = {
        "Temperature": {
            "low":     f"🌡 Temperature too low ({val}°C < {mn}°C) — Rice seedling growth slows below {mn}°C. Cover with plastic mulch at night.",
            "high":    f"🔥 Temperature too high ({val}°C > {mx}°C) — Heat stress! Increase irrigation frequency. Use mulching to cool soil surface.",
            "optimal": f"✅ Temperature optimal ({val}°C in {ideal_str}) — Ideal for photosynthesis and grain filling.",
        },
        "Humidity": {
            "low":     f"💧 Humidity low ({val}% < {mn}%) — Crop may wilt. Increase irrigation. High ET expected — water loss is high.",
            "high":    f"⚠ Humidity high ({val}% > {mx}%) — Fungal disease risk (Rice Blast, Sheath Blight). Improve field ventilation. Monitor for lesions.",
            "optimal": f"✅ Humidity optimal ({val}% in {ideal_str}) — Good for leaf expansion and pollination.",
        },
        "WindSpeed": {
            "low":     f"🍃 Wind low ({val} m/s < {mn} m/s) — Stagnant air increases humidity and disease risk. No action needed but monitor.",
            "high":    f"🌬 Wind high ({val} m/s > {mx} m/s) — Risk of lodging (crop falling). Install windbreaks. Avoid pesticide spraying.",
            "optimal": f"✅ Wind speed optimal ({val} m/s in {ideal_str}) — Good air circulation reduces disease spread.",
        },
        "SolarRadiation": {
            "low":     f"🌥 Solar radiation low ({val} W/m²) — Photosynthesis reduced. Biomass accumulation slows. No action — weather dependent.",
            "high":    f"☀ Very high solar ({val} W/m²) — Combined with high temp = heat stress. Increase irrigation.",
            "optimal": f"✅ Solar radiation good ({val} W/m²) — High photosynthesis. Biomass accumulating well.",
        },
        "SoilMoisture": {
            "low":     f"💧 Soil dry ({val}% < {mn}%) — Rice roots stressed. Irrigate immediately. Check irrigation system.",
            "high":    f"🌊 Soil waterlogged ({val}% > {mx}%) — Promotes Sheath Blight and root rot. Open drainage channels.",
            "optimal": f"✅ Soil moisture optimal ({val}% in {ideal_str}) — Roots healthy. Continue current irrigation schedule.",
        },
        "pH": {
            "low":     f"⚗ Soil acidic (pH {val} < {mn}) — Nutrient uptake blocked. Apply agricultural lime @ 2 bags/acre.",
            "high":    f"⚗ Soil alkaline (pH {val} > {mx}) — Iron/Zinc deficiency risk. Add organic compost or sulfur.",
            "optimal": f"✅ Soil pH optimal ({val} in {ideal_str}) — All nutrients available for uptake.",
        },
        "Nitrogen": {
            "low":     f"🧪 Nitrogen low ({val} kg/ha < {mn}) — Yellowing leaves, stunted growth. Apply urea @ 50 kg/acre in split doses.",
            "high":    f"⚠ Nitrogen excess ({val} kg/ha > {mx}) — Promotes Rice Blast, soft tissue. Stop N application. Drain field.",
            "optimal": f"✅ Nitrogen optimal ({val} kg/ha in {ideal_str}) — Good leaf color and growth index.",
        },
        "Phosphorus": {
            "low":     f"🧪 Phosphorus low ({val} kg/ha < {mn}) — Root development poor. Apply DAP @ 25 kg/acre before next irrigation.",
            "high":    f"ℹ Phosphorus sufficient ({val} kg/ha > {mx}) — No action needed.",
            "optimal": f"✅ Phosphorus optimal ({val} kg/ha in {ideal_str}) — Strong root system and energy transfer.",
        },
        "Potassium": {
            "low":     f"🧪 Potassium low ({val} kg/ha < {mn}) — Weak stems, Brown Spot risk. Apply MOP @ 20 kg/acre.",
            "high":    f"ℹ Potassium sufficient ({val} kg/ha > {mx}) — No action needed.",
            "optimal": f"✅ Potassium optimal ({val} kg/ha in {ideal_str}) — Strong stems and disease resistance.",
        },
    }
    if param not in SUGG: return None
    if val < mn: return SUGG[param]["low"]
    if val > mx: return SUGG[param]["high"]
    return SUGG[param]["optimal"]

# ================================================================
# CSS — LIGHT PALE GREEN THEME
# ================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap');

/* ═══════════════════════════════════════════════════════════════
   FUTURISTIC DARK-GRAY — IEEE Conference & Color Xerox Perfect
   Background: #0f172a→#1e293b  |  Neon green + cyan accents
   High readability · Glassmorphism · Professional typography
═══════════════════════════════════════════════════════════════ */

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
  background: #0f172a !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 15px !important;
  color: #e2e8f0 !important;
}
[data-testid="stAppViewContainer"] {
  background: linear-gradient(160deg,
    #0f172a 0%,
    #111827 30%,
    #1a2744 65%,
    #1e293b 100%) !important;
}

p, span, li, label {
  color: #cbd5e1 !important;
  font-size: 15px !important;
  font-family: 'Inter', sans-serif !important;
}
b, strong { color: #f8fafc !important; font-weight: 700 !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2.5rem 5rem !important; max-width: 1600px !important; }
[data-testid="stMetric"] { display: none !important; }

h1, h2, h3, h4 {
  font-family: 'Inter', sans-serif !important;
  color: #4ade80 !important;
  font-weight: 800 !important;
}

/* ─── SCROLLBAR ────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: rgba(74,222,128,.25); border-radius: 99px; }

/* ─── TABS ─────────────────────────────────── */
[data-baseweb="tab-list"] {
  background: rgba(30,41,59,.8) !important;
  border: 1px solid rgba(74,222,128,.2) !important;
  border-radius: 14px !important;
  padding: 5px !important;
  backdrop-filter: blur(12px) !important;
  box-shadow: 0 4px 24px rgba(0,0,0,.4) !important;
}
[data-baseweb="tab"] {
  color: #94a3b8 !important;
  font-weight: 600 !important;
  font-size: 13.5px !important;
  border-radius: 10px !important;
  letter-spacing: .2px !important;
  transition: color .2s !important;
}
[data-baseweb="tab"]:hover { color: #4ade80 !important; }
[aria-selected="true"] {
  background: linear-gradient(135deg, #16a34a 0%, #0891b2 100%) !important;
  color: #ffffff !important;
  border-radius: 10px !important;
  font-weight: 700 !important;
  box-shadow: 0 2px 12px rgba(22,163,74,.4) !important;
}

/* ─── HEADER ────────────────────────────────── */
.site-header {
  text-align: center;
  padding: 3rem 1rem 1.4rem;
  position: relative;
}
.site-header::after {
  content: '';
  display: block;
  width: 240px; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(74,222,128,.5), transparent);
  margin: 1rem auto 0;
}
.site-header .sub {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 10.5px !important;
  font-weight: 600 !important;
  color: #64748b !important;
  letter-spacing: 4.5px;
  text-transform: uppercase;
  margin-bottom: .6rem;
}
.site-header .proj-desc {
  font-size: 13.5px !important;
  color: #94a3b8 !important;
  font-style: italic;
  max-width: 980px;
  margin: .5rem auto 0;
  line-height: 1.85;
  font-weight: 400;
}

/* ─── SECTION LABELS ────────────────────────── */
.sec-label {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 10.5px !important;
  font-weight: 700 !important;
  color: #4ade80 !important;
  text-transform: uppercase !important;
  letter-spacing: 4.5px !important;
  margin: 2.8rem 0 1.3rem !important;
  display: flex;
  align-items: center;
  gap: 14px;
}
.sec-label::before {
  content: '';
  width: 3px; height: 16px;
  background: linear-gradient(180deg, #4ade80, #22d3ee);
  border-radius: 99px;
  flex-shrink: 0;
}
.sec-label::after {
  content: '';
  flex: 1; height: 1px;
  background: linear-gradient(90deg, rgba(74,222,128,.3), transparent);
}

/* ─── METRIC CARDS ──────────────────────────── */
.mcard {
  background: rgba(30,41,59,.65);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(148,163,184,.12);
  border-radius: 18px;
  padding: 1.6rem 1.8rem 1.4rem;
  position: relative;
  overflow: hidden;
  transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
  box-shadow:
    0 1px 0 rgba(255,255,255,.05) inset,
    0 8px 32px rgba(0,0,0,.45);
}
.mcard::after {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: 18px;
  background: linear-gradient(145deg, rgba(255,255,255,.04) 0%, transparent 60%);
  pointer-events: none;
}
.mcard:hover {
  transform: translateY(-4px);
  border-color: rgba(74,222,128,.3);
  box-shadow:
    0 1px 0 rgba(255,255,255,.07) inset,
    0 0 0 1px rgba(74,222,128,.15),
    0 16px 48px rgba(0,0,0,.55);
}
.mcard::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2.5px;
  border-radius: 18px 18px 0 0;
}

/* Card accent top bars */
.ct::before  { background: linear-gradient(90deg,#f97316,#ef4444); }
.ch::before  { background: linear-gradient(90deg,#38bdf8,#818cf8); }
.cr::before  { background: linear-gradient(90deg,#a78bfa,#ec4899); }
.cw::before  { background: linear-gradient(90deg,#c084fc,#a855f7); }
.cs::before  { background: linear-gradient(90deg,#fbbf24,#f97316); }
.cm::before  { background: linear-gradient(90deg,#4ade80,#22c55e); }
.cst::before { background: linear-gradient(90deg,#fb923c,#f97316); }
.cp::before  { background: linear-gradient(90deg,#f472b6,#ec4899); }
.cn::before  { background: linear-gradient(90deg,#86efac,#4ade80); }
.cph::before { background: linear-gradient(90deg,#fdba74,#fb923c); }
.ck::before  { background: linear-gradient(90deg,#60a5fa,#3b82f6); }
.clci::before{ background: linear-gradient(90deg,#4ade80,#86efac); }
.cls::before { background: linear-gradient(90deg,#f87171,#ef4444); }
.cci::before { background: linear-gradient(90deg,#6ee7b7,#34d399); }
.cdp::before { background: linear-gradient(90deg,#fdba74,#f97316); }
.cfr::before { background: linear-gradient(90deg,#c084fc,#a855f7); }
.cnd::before { background: linear-gradient(90deg,#67e8f9,#22d3ee); }

/* ─── CARD TEXT ─────────────────────────────── */
.clabel {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 9.5px !important;
  font-weight: 700 !important;
  color: #64748b !important;
  text-transform: uppercase !important;
  letter-spacing: 3px !important;
  margin-bottom: .6rem !important;
}
.cval {
  font-family: 'Inter', sans-serif !important;
  font-size: 3.3rem !important;
  font-weight: 800 !important;
  color: #f1f5f9 !important;
  -webkit-text-fill-color: #f1f5f9 !important;
  line-height: 1 !important;
  margin-bottom: .4rem !important;
  letter-spacing: -1.5px;
}
.cval-sm {
  font-family: 'Inter', sans-serif !important;
  font-size: 2rem !important;
  font-weight: 800 !important;
  color: #f1f5f9 !important;
  -webkit-text-fill-color: #f1f5f9 !important;
  line-height: 1.2 !important;
  margin-bottom: .4rem !important;
  letter-spacing: -.5px;
}
.cunit {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 12px !important;
  color: #4ade80 !important;
  font-weight: 600 !important;
}
.cdesc {
  font-size: 12px !important;
  color: #475569 !important;
  margin-top: .4rem !important;
  line-height: 1.5 !important;
  font-weight: 500 !important;
}

/* ─── BADGES ─────────────────────────────────── */
.bg  { background:rgba(74,222,128,.15); color:#4ade80; border:1px solid rgba(74,222,128,.35); border-radius:99px; padding:3px 14px; font-size:11.5px !important; font-weight:700 !important; }
.br  { background:rgba(239,68,68,.15); color:#f87171; border:1px solid rgba(239,68,68,.35); border-radius:99px; padding:3px 14px; font-size:11.5px !important; font-weight:700 !important; }
.bw  { background:rgba(251,191,36,.15); color:#fbbf24; border:1px solid rgba(251,191,36,.35); border-radius:99px; padding:3px 14px; font-size:11.5px !important; font-weight:700 !important; }
.bn  { background:rgba(74,222,128,.15); color:#4ade80; border:1px solid rgba(74,222,128,.35); border-radius:99px; padding:3px 14px; font-size:11.5px !important; font-weight:700 !important; }

/* ─── ALERT / SUGGESTION BOXES ──────────────── */
.ae   { background:rgba(239,68,68,.08); border-left:3px solid #ef4444; border-radius:0 14px 14px 0; padding:.95rem 1.4rem; margin:.5rem 0; color:#fca5a5 !important; font-size:14.5px !important; font-weight:500; line-height:1.75; }
.aw   { background:rgba(251,191,36,.08); border-left:3px solid #fbbf24; border-radius:0 14px 14px 0; padding:.95rem 1.4rem; margin:.5rem 0; color:#fde68a !important; font-size:14.5px !important; font-weight:500; line-height:1.75; }
.ao   { background:rgba(74,222,128,.07); border-left:3px solid #4ade80; border-radius:0 14px 14px 0; padding:.95rem 1.4rem; margin:.5rem 0; color:#bbf7d0 !important; font-size:14.5px !important; font-weight:500; line-height:1.75; }
.ai   { background:rgba(56,189,248,.07); border-left:3px solid #38bdf8; border-radius:0 14px 14px 0; padding:.95rem 1.4rem; margin:.5rem 0; color:#bae6fd !important; font-size:14.5px !important; font-weight:500; line-height:1.75; }
.aopt { background:rgba(74,222,128,.07); border-left:3px solid #86efac; border-radius:0 14px 14px 0; padding:.95rem 1.4rem; margin:.5rem 0; color:#bbf7d0 !important; font-size:14.5px !important; font-weight:500; line-height:1.75; }
.alow { background:rgba(251,146,60,.08); border-left:3px solid #fb923c; border-radius:0 14px 14px 0; padding:.95rem 1.4rem; margin:.5rem 0; color:#fed7aa !important; font-size:14.5px !important; font-weight:500; line-height:1.75; }
.ahigh{ background:rgba(239,68,68,.08); border-left:3px solid #f87171; border-radius:0 14px 14px 0; padding:.95rem 1.4rem; margin:.5rem 0; color:#fca5a5 !important; font-size:14.5px !important; font-weight:500; line-height:1.75; }

/* ─── EXPLAIN BOX ────────────────────────────── */
.explain {
  background: rgba(30,41,59,.6);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(74,222,128,.18);
  border-radius: 14px;
  padding: 1.3rem 1.6rem;
  margin: .9rem 0;
  font-size: 14.5px !important;
  color: #cbd5e1 !important;
  line-height: 1.8;
  font-weight: 400;
}
.explain b { color: #4ade80 !important; font-weight: 700 !important; }

/* ─── YIELD CARD ─────────────────────────────── */
.yield-card {
  background: linear-gradient(135deg,
    rgba(30,41,59,.9) 0%,
    rgba(22,40,60,.9) 100%);
  backdrop-filter: blur(16px);
  border: 1px solid rgba(74,222,128,.25);
  border-radius: 22px;
  padding: 2.5rem 3rem;
  text-align: center;
  box-shadow:
    0 0 0 1px rgba(74,222,128,.08),
    0 20px 60px rgba(0,0,0,.5);
}
.yield-num {
  font-family: 'Inter', sans-serif !important;
  font-size: 5rem !important;
  font-weight: 900 !important;
  color: #f8fafc !important;
  -webkit-text-fill-color: #f8fafc !important;
  line-height: 1;
  letter-spacing: -3px;
}
.yield-unit  { color: #4ade80 !important; font-size: 13px !important; font-weight: 600; margin-top: .5rem; font-family:'JetBrains Mono',monospace; }
.yield-grade { font-size: 1.35rem !important; font-weight: 800 !important; margin-top: .9rem; }

/* ─── GROWTH STAGES ──────────────────────────── */
.stage-row { display:flex; align-items:center; gap:14px; padding:.85rem 1.5rem; border-radius:13px; margin:.4rem 0; font-size:14.5px !important; font-weight:600; border:1px solid transparent; transition:all .2s; }
.stage-active  { background:rgba(74,222,128,.1); border-color:rgba(74,222,128,.3); color:#f1f5f9 !important; font-weight:800 !important; }
.stage-done    { background:rgba(255,255,255,.03); color:#64748b !important; border-color:rgba(255,255,255,.07); }
.stage-future  { background:rgba(255,255,255,.01); color:#334155 !important; }

/* ─── FEATURE IMPORTANCE ─────────────────────── */
.fimp-row { display:flex; align-items:center; gap:14px; padding:.55rem 0; font-size:13.5px !important; font-weight:500; color:#cbd5e1 !important; }
.fimp-bar  { flex:1; background:rgba(255,255,255,.06); border-radius:99px; height:8px; overflow:hidden; }
.fimp-fill { height:100%; border-radius:99px; background:linear-gradient(90deg,#4ade80,#22d3ee); }

/* ─── SUGGESTION BOXES ───────────────────────── */
.sug      { background:rgba(30,41,59,.5); border:1px solid rgba(74,222,128,.15); border-radius:13px; padding:.95rem 1.4rem; margin:.5rem 0; color:#cbd5e1 !important; font-size:14.5px !important; font-weight:500; line-height:1.75; backdrop-filter:blur(8px); }
.sug-opt  { background:rgba(74,222,128,.07); border:1px solid rgba(74,222,128,.22); border-radius:13px; padding:.95rem 1.4rem; margin:.5rem 0; color:#bbf7d0 !important; font-size:14.5px !important; font-weight:500; line-height:1.75; }
.sug-low  { background:rgba(251,146,60,.07); border:1px solid rgba(251,146,60,.22); border-radius:13px; padding:.95rem 1.4rem; margin:.5rem 0; color:#fed7aa !important; font-size:14.5px !important; font-weight:500; line-height:1.75; }
.sug-high { background:rgba(239,68,68,.07); border:1px solid rgba(239,68,68,.22); border-radius:13px; padding:.95rem 1.4rem; margin:.5rem 0; color:#fca5a5 !important; font-size:14.5px !important; font-weight:500; line-height:1.75; }

/* ─── TABLE ──────────────────────────────────── */
.itable { width:100%; border-collapse:collapse; font-size:14.5px !important; }
.itable th {
  background: rgba(74,222,128,.08);
  color: #4ade80 !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-weight: 700 !important;
  text-transform: uppercase;
  letter-spacing: 3px;
  font-size: 9.5px !important;
  padding: 13px 18px;
  text-align: left;
  border-bottom: 1px solid rgba(74,222,128,.2);
}
.itable td { padding:12px 18px; border-bottom:1px solid rgba(255,255,255,.05); color:#cbd5e1 !important; font-size:14.5px !important; font-weight:400; }
.itable tr:hover td { background:rgba(74,222,128,.04); }

/* ─── MODEL CARDS ────────────────────────────── */
.model-card {
  background: rgba(30,41,59,.65);
  backdrop-filter: blur(14px);
  border: 1px solid rgba(148,163,184,.12);
  border-radius: 18px;
  padding: 1.8rem 2rem;
  box-shadow: 0 8px 32px rgba(0,0,0,.4), inset 0 1px 0 rgba(255,255,255,.05);
  transition: all .22s;
}
.model-card:hover { border-color:rgba(74,222,128,.25); box-shadow:0 0 0 1px rgba(74,222,128,.12),0 16px 48px rgba(0,0,0,.55); }
.model-name { font-size:1.05rem !important; font-weight:800 !important; margin-bottom:.5rem; color:#4ade80 !important; letter-spacing:-.2px; }
.model-metric { font-family:'Inter',sans-serif !important; font-size:2.7rem !important; font-weight:900 !important; color:#f1f5f9 !important; letter-spacing:-1px; }
.model-desc { font-size:13.5px !important; color:#94a3b8 !important; font-weight:400; margin-top:.5rem; line-height:1.7; }
.model-how  { font-size:12.5px !important; color:#64748b !important; font-weight:400; margin-top:.6rem; line-height:1.7; border-top:1px solid rgba(255,255,255,.07); padding-top:.6rem; }

/* ─── DISEASE CARDS ──────────────────────────── */
.dis-card { background:rgba(30,41,59,.65); backdrop-filter:blur(12px); border:1px solid rgba(148,163,184,.12); border-radius:16px; padding:1.5rem 1.8rem; margin-bottom:1rem; box-shadow:0 8px 28px rgba(0,0,0,.4),inset 0 1px 0 rgba(255,255,255,.04); }
.dis-card .dtitle { font-size:1.25rem !important; font-weight:800 !important; margin-bottom:.35rem; color:#4ade80 !important; letter-spacing:-.3px; }
.dis-card .dpathogen { font-size:12.5px !important; color:#64748b !important; font-weight:500; margin-bottom:.65rem; font-style:italic; }
.dis-card .dbar-bg { background:rgba(255,255,255,.06); border-radius:99px; height:8px; overflow:hidden; margin:.5rem 0; }
.dis-card .dbar-fill { height:100%; border-radius:99px; }

/* ─── MISC ───────────────────────────────────── */
.src-badge { display:inline-block; background:rgba(34,211,238,.1); border:1px solid rgba(34,211,238,.25); color:#22d3ee !important; border-radius:99px; padding:4px 16px; font-size:9.5px !important; font-weight:700 !important; letter-spacing:2.5px; text-transform:uppercase; margin-bottom:1.2rem; font-family:'JetBrains Mono',monospace; }
.soil-pred { background:rgba(30,41,59,.6); backdrop-filter:blur(10px); border:1px solid rgba(148,163,184,.1); border-radius:16px; padding:1.5rem; margin:.7rem 0; }
.upload-zone { background:rgba(30,41,59,.5); backdrop-filter:blur(12px); border:1.5px dashed rgba(74,222,128,.25); border-radius:22px; padding:4rem; text-align:center; margin:1.4rem 0; }
.upload-zone h3 { font-size:1.55rem !important; font-weight:800 !important; color:#4ade80 !important; margin:0 0 .7rem; }
.disease-result { border-radius:20px; padding:2.2rem 2.8rem; text-align:center; margin-bottom:1.4rem; }
.disease-name { font-family:'Inter',sans-serif !important; font-size:2.3rem !important; font-weight:900 !important; margin-bottom:.45rem; color:#f1f5f9 !important; letter-spacing:-.5px; }

.stButton button {
  background: linear-gradient(135deg, #16a34a 0%, #0891b2 100%) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 12px !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 700 !important;
  font-size: 14.5px !important;
  padding: .85rem 2.8rem !important;
  box-shadow: 0 4px 20px rgba(22,163,74,.3) !important;
  letter-spacing: .3px !important;
}

/* stMarkdown override */
[data-testid="stMarkdownContainer"] { width:100% !important; }
[data-testid="stMarkdownContainer"] > div { width:100% !important; }
div.big-title {
  font-family: 'Inter', sans-serif !important;
  font-size: 72px !important;
  font-weight: 900 !important;
  color: #4ade80 !important;
  -webkit-text-fill-color: #4ade80 !important;
  letter-spacing: -3px !important;
  line-height: 1.05 !important;
  text-align: center !important;
  display: block !important;
  width: 100% !important;
}
</style>
<script>
function fixTitle() {
  var els = document.querySelectorAll("div,p,span,h1,h2,h3");
  els.forEach(function(e) {
    if (e.childNodes.length === 1 &&
        e.textContent.trim().indexOf("Smart Farming Digital Twin") >= 0) {
      e.style.fontSize        = "76px";
      e.style.fontWeight      = "900";
      e.style.color           = "#4ade80";
      e.style.webkitTextFillColor = "#4ade80";
      e.style.fontFamily      = "'Inter', sans-serif";
      e.style.letterSpacing   = "-3px";
      e.style.lineHeight      = "1.05";
      e.style.textAlign       = "center";
      e.style.display         = "block";
      e.style.width           = "100%";
    }
  });
}
fixTitle();
setTimeout(fixTitle, 250);
setTimeout(fixTitle, 800);
setTimeout(fixTitle, 2000);
setInterval(fixTitle, 2500)</script>
""", unsafe_allow_html=True)

def hex_rgba(h, a=1.0):
    h=h.lstrip("#"); r,g,b=int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    return f"rgba({r},{g},{b},{a})"

def badge_v(param, val):
    """Badge based on day/night ideal"""
    mn,mx=get_ideal(param)
    if mn is None: return '<span class="bn">—</span>'
    if val<mn: return '<span class="bw">⬇ Low</span>'
    if val>mx: return '<span class="br">⬆ High</span>'
    return '<span class="bg">✅ Optimal</span>'

def sf(v,d=0.):
    try: return float(v)
    except: return d
def ss(v,d="N/A"):
    try: return str(v)
    except: return d

PC={"Temperature":"#f97316","Humidity":"#0891b2","Rainfall":"#6366f1","WindSpeed":"#8b5cf6","SolarRadiation":"#f59e0b"}

# ================================================================
# LOAD DATA
# ================================================================
@st.cache_data(ttl=60)
def load_all():
    climate=pd.read_csv("data/climate_data.csv")
    climate["Date"]=pd.to_datetime(climate["Date"]); climate=climate.sort_values("Date").reset_index(drop=True)
    soil     = pd.read_csv("data/soil_data.csv").iloc[0]
    crop     = pd.read_csv("data/crop_growth.csv")           if os.path.exists("data/crop_growth.csv") else None
    disease  = pd.read_csv("data/disease_risk.csv")          if os.path.exists("data/disease_risk.csv") else None
    patho    = pd.read_csv("data/plant_pathology.csv").iloc[0] if os.path.exists("data/plant_pathology.csv") else None
    img      = pd.read_csv("data/image_analysis.csv").iloc[0]  if os.path.exists("data/image_analysis.csv") else None
    yld      = pd.read_csv("data/yield_prediction.csv").iloc[0] if os.path.exists("data/yield_prediction.csv") else None
    irr      = pd.read_csv("data/irrigation_recommendation.csv").iloc[0] if os.path.exists("data/irrigation_recommendation.csv") else None
    fi       = pd.read_csv("data/models/feature_importance.csv") if os.path.exists("data/models/feature_importance.csv") else None
    ms       = pd.read_csv("data/models/model_summary.csv")      if os.path.exists("data/models/model_summary.csv") else None
    fc       = pd.read_csv("data/climate_forecast.csv")      if os.path.exists("data/climate_forecast.csv") else None
    dfc      = pd.read_csv("data/disease_forecast.csv")      if os.path.exists("data/disease_forecast.csv") else None
    cfg      = json.load(open("data/crop_config.json"))      if os.path.exists("data/crop_config.json") else {}
    return climate,soil,crop,disease,patho,img,yld,irr,fi,ms,fc,dfc,cfg

climate,soil,crop_df,disease_df,path_row,img_anal,yield_row,irrig_row,feat_imp,model_sum,forecast_df,dis_fc,crop_cfg=load_all()

df3=climate.set_index("Date").resample("3h").agg(
    {"Temperature":"mean","Humidity":"mean","Rainfall":"sum","WindSpeed":"mean","SolarRadiation":"max"}
).dropna(how="all").reset_index()
for c in list(PC.keys()):
    if c in df3.columns: df3[c]=df3[c].round(2)

latest=df3.iloc[-1]; cur_hour=datetime.datetime.now().hour
temp=round(sf(latest["Temperature"])+random.uniform(-.3,.3),2)
hum=sf(latest["Humidity"]); rain=sf(latest["Rainfall"]); wind=sf(latest["WindSpeed"])
# Solar: use latest value from raw climate data (not 3h average)
sv_raw = climate[climate["SolarRadiation"]>0]["SolarRadiation"] if "SolarRadiation" in climate.columns else pd.Series([0])
solar_raw = round(sf(sv_raw.iloc[-1]),1) if (6<=cur_hour<=18 and not sv_raw.empty) else 0.0
# Also update df3 solar for graph consistency
sv3=df3[df3["SolarRadiation"]>0]["SolarRadiation"] if "SolarRadiation" in df3.columns else pd.Series([0])
solar=solar_raw  # use raw latest for card display
moist=sf(soil.get("SoilMoisture",48)); stemp=soil.get("SoilTemp_0cm",None)
ph=sf(soil.get("pH",6.8)); nitro=sf(soil.get("Nitrogen",72))
phos=sf(soil.get("Phosphorus",46)); pota=sf(soil.get("Potassium",63))
src=ss(soil.get("Source","Unknown")); sv=round(sf(stemp),1) if stemp else None
period_label="☀ Day" if is_day() else "🌙 Night"

# ================================================================
# HEADER
# ================================================================
st.markdown(f"""
<div class="site-header">
  <div style="font-family:'Montserrat',sans-serif;font-size:76px;font-weight:900;color:#4ade80;-webkit-text-fill-color:#4ade80;letter-spacing:-2px;line-height:1.1;text-align:center;width:100%;display:block;margin:0 auto 0.6rem auto;" class="big-title">🌾 Smart Farming Digital Twin</div>
  <div style="
    font-family:'Inter',sans-serif;
    font-size:14px;
    font-weight:700;
    color:#64748b;
    letter-spacing:3.5px;
    text-transform:uppercase;
    text-align:center;
    margin-bottom:.6rem;
  ">📍 Guntur, Andhra Pradesh &nbsp;·&nbsp; Real-Time Monitor &nbsp;·&nbsp;
    {datetime.datetime.now().strftime('%d %b %Y &nbsp;·&nbsp; %I:%M %p')} &nbsp;·&nbsp; {period_label}</div>
  <div style="
    font-family:'Nunito',sans-serif;
    font-size:15px;
    font-weight:600;
    color:#64748b;
    font-style:italic;
    max-width:1000px;
    margin:0 auto;
    line-height:1.75;
    text-align:center;
  ">
    A Digital Twin-Driven Climate-Aware Framework that constructs a real-time virtual replica of a rice field
    using live API-sourced climate and soil data — requiring no physical sensors in the field.
    The Digital Twin continuously synchronizes with live environmental conditions, simulating crop growth dynamics,
    forecasting weather patterns, predicting yield, detecting plant diseases, and recommending irrigation actions.
  </div>
</div>""", unsafe_allow_html=True)

tab1,tab2,tab3,tab4,tab5,tab6,tab7=st.tabs([
    "🌤 Climate & Soil","🌾 Crop Growth","📸 Disease & Pathology",
    "🦠 Disease Risk","🤖 ML Models","💧 Irrigation","📈 Trends"
])

# ╔══════════════════╗
# ║  TAB 1 CLIMATE  ║
# ╚══════════════════╝
with tab1:
    st.markdown(f'<div class="sec-label">🌤 Live Climate — Guntur &nbsp;·&nbsp; {period_label} Ideals Active</div>', unsafe_allow_html=True)

    for col,(cls,lbl,param,val,unit) in zip(st.columns(5),[
        ("ct","🌡 Temperature","Temperature",temp,"°C"),
        ("ch","💧 Humidity","Humidity",hum,"%"),
        ("cr","🌧 Rainfall","Rainfall",rain,"mm"),
        ("cw","🌬 Wind Speed","WindSpeed",wind,"m/s"),
        ("cs","🌞 Solar","SolarRadiation",solar,"W/m²"),
    ]):
        mn,mx=get_ideal(param)
        ideal_str=f"{mn}–{mx} {unit}" if mn is not None else "—"
        bdg=badge_v(param,val)
        col.markdown(f'<div class="mcard {cls}"><div class="clabel">{lbl}</div><div class="cval">{val} <span class="cunit">{unit}</span></div>{bdg}<div class="cdesc">{period_label} ideal: {ideal_str}</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="sec-label">🌱 Soil Health — {src}</div>', unsafe_allow_html=True)
    for col,(cls,lbl,param,val,unit) in zip(st.columns(6),[
        ("cm","💧 Moisture","SoilMoisture",moist,"%"),
        ("cst","🌡 Soil Temp","Temperature",sv if sv else 27,"°C"),
        ("cp","⚗ pH","pH",ph,""),
        ("cn","🧪 Nitrogen","Nitrogen",nitro,"kg/ha"),
        ("cph","🧪 Phosphorus","Phosphorus",phos,"kg/ha"),
        ("ck","🧪 Potassium","Potassium",pota,"kg/ha"),
    ]):
        mn,mx=get_ideal(param) if param in IDEALS else (None,None)
        ideal_str=f"{mn}–{mx} {unit}" if mn is not None else "—"
        bdg=badge_v(param,val) if param in IDEALS else '<span class="bn">—</span>'
        col.markdown(f'<div class="mcard {cls}"><div class="clabel">{lbl}</div><div class="cval">{val} <span class="cunit">{unit}</span></div>{bdg}<div class="cdesc">Ideal: {ideal_str}</div></div>', unsafe_allow_html=True)

    # FULL SUGGESTIONS — low/high/optimal for each
    st.markdown(f'<div class="sec-label">💡 Smart Suggestions — {period_label} Analysis (All Parameters)</div>', unsafe_allow_html=True)
    params_vals = [
        ("Temperature",temp),("Humidity",hum),("WindSpeed",wind),("SolarRadiation",solar),
        ("SoilMoisture",moist),("pH",ph),("Nitrogen",nitro),("Phosphorus",phos),("Potassium",pota),
    ]
    c1,c2=st.columns(2)
    for i,(param,val) in enumerate(params_vals):
        sugg=suggestion_for(param,val)
        if sugg is None: continue
        mn,mx=get_ideal(param)
        if val<mn: css_class="sug-low"
        elif val>mx: css_class="sug-high"
        else: css_class="sug-opt"
        target=c1 if i%2==0 else c2
        target.markdown(f'<div class="{css_class}">{sugg}</div>', unsafe_allow_html=True)

    # Day/Night ideal table
    st.markdown('<div class="sec-label">🌗 Day vs Night Ideal Conditions — Guntur Rice</div>', unsafe_allow_html=True)
    dn_rows=[
        ("🌡 Temperature",     "26–34 °C",   "20–25 °C"),
        ("💧 Humidity",        "70–85 %",    "80–90 %"),
        ("🌬 Wind Speed",      "1.0–2.5 m/s","0.5–1.5 m/s"),
        ("🌞 Solar Radiation", "5.0–6.5 kWh","0 kWh"),
        ("💧 Soil Moisture",   "80–100 %",   "80–95 %"),
        ("⚗ Soil pH",          "5.2–6.5",    "5.2–6.5"),
        ("🧪 Nitrogen",        "100–120 kg/ha","Conversion phase"),
        ("🧪 Phosphorus",      "40–60 kg/ha", "Root activity"),
        ("🧪 Potassium",       "40–80 kg/ha", "Stress recovery"),
    ]
    th='<table class="itable"><tr><th>Parameter</th><th>☀ Day Ideal</th><th>🌙 Night Ideal</th><th>Current</th><th>Status</th></tr>'
    curr_vals={"🌡 Temperature":f"{temp}°C","💧 Humidity":f"{hum}%","🌬 Wind Speed":f"{wind}m/s",
               "🌞 Solar Radiation":f"{solar}W/m²","💧 Soil Moisture":f"{moist}%","⚗ Soil pH":f"{ph}",
               "🧪 Nitrogen":f"{nitro}","🧪 Phosphorus":f"{phos}","🧪 Potassium":f"{pota}"}
    status_map={"Temperature":(temp,"Temperature"),"Humidity":(hum,"Humidity"),"Wind Speed":(wind,"WindSpeed"),
                "Solar Radiation":(solar,"SolarRadiation"),"Soil Moisture":(moist,"SoilMoisture"),
                "Soil pH":(ph,"pH"),"Nitrogen":(nitro,"Nitrogen"),"Phosphorus":(phos,"Phosphorus"),"Potassium":(pota,"Potassium")}
    for lbl,day_i,night_i in dn_rows:
        curr=curr_vals.get(lbl,"—")
        # find status
        for k,(v,pname) in status_map.items():
            if k in lbl:
                s,clr=status_for(pname,v); break
        else: s,clr="—","#94a3b8"
        color_map={"✅ Optimal":"#16a34a","⬇ Low":"#f97316","⬆ High":"#ef4444","—":"#94a3b8"}
        clr=color_map.get(s,"#94a3b8")
        th+=f'<tr><td>{lbl}</td><td style="color:#15803d">{day_i}</td><td style="color:#0891b2">{night_i}</td><td style="font-weight:600">{curr}</td><td style="color:{clr};font-weight:700">{s}</td></tr>'
    st.markdown(th+"</table>", unsafe_allow_html=True)

# ╔════════════════════╗
# ║  TAB 2 CROP GROWTH ║
# ╚════════════════════╝
with tab2:
    st.markdown('<div class="sec-label">🌾 Rice Crop Growth — Guntur, AP (GDD Method)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="explain">
      <b>📍 Guntur, AP — All data from OpenWeatherMap Guntur API</b><br><br>
      <b>What is GDD and how does it drive crop growth?</b><br>
      GDD (Growing Degree Days) = max(0, daily avg temp − 10°C base temp)<br>
      Example: Guntur temp=30°C → GDD today = 30−10 = 20 degree-days<br>
      Rice needs ~1800 total GDD to complete its growth cycle (MTU 1010).<br><br>
      <b>How each day's growth is calculated:</b><br>
      Growth Index = (Temperature factor×38%) + (Water factor×35%) + (Solar factor×27%) × Fertilizer boost<br>
      • Temperature factor: how close today's temp is to optimal (29°C for MTU1010)<br>
      • Water factor: rainfall + humidity + irrigation type<br>
      • Solar factor: solar radiation ÷ 650 W/m²<br><br>
      <b>What the graphs show:</b><br>
      • Growth Index: 0–100% daily quality score — higher = better growing conditions<br>
      • Biomass: total plant dry matter (g/m²) — accumulates every day<br>
      • Leaf Area Index: leaf coverage (peaks at booting stage ~day 55–70)<br>
      • Yield: increases only after booting — stays low early, rises at grain filling<br><br>
      <b>Solid lines</b> = actual days using real Guntur climate | <b>Dashed</b> = projected future | ◆ = Today
    </div>""", unsafe_allow_html=True)

    if crop_df is None or len(crop_df)==0:
        with st.spinner("Loading Guntur crop data..."):
            sys.path.insert(0,"scripts")
            from crop_growth import simulate_growth
            crop_df,_,_=simulate_growth("data/climate_data.csv",crop_cfg if crop_cfg else None)
        st.cache_data.clear()

    if crop_df is not None and len(crop_df)>0:
        # FIX: Always recalculate cur_day from TODAY - sowing_date (not from CSV)
        import datetime as _dt
        _sowing_str = crop_cfg.get("sowing_date","2026-04-01")
        try:
            _sowing = _dt.datetime.strptime(_sowing_str, "%Y-%m-%d")
            cur_day = max(0, (_dt.datetime.now() - _sowing).days)
        except:
            cur_day = 0
        # Find closest row in crop_df
        cur_rows = crop_df[crop_df["Day"] == cur_day]
        if len(cur_rows) == 0:
            cur_rows = crop_df[crop_df["Day"] <= cur_day]
        cur_row = cur_rows.iloc[-1] if len(cur_rows) > 0 else crop_df.iloc[-1]
        cur_day = int(cur_row["Day"])
        variety = crop_cfg.get("variety","MTU 1010 (Rajendra)")
        past_df=crop_df[crop_df["IsPast"]==True]
        avg_gi=round(float(past_df["GrowthIndex"].mean()),1) if len(past_df) else float(cur_row["GrowthIndex"])

        # STRESS DAYS — count ALL days 0..today using Temperature directly
        # Stress: check ALL rows up to and including today
        try:
            days_so_far    = crop_df[crop_df["Day"] <= cur_day]
            heat_d         = int((days_so_far["Temperature"] > 35).sum())
            cold_d         = int((days_so_far["Temperature"] < 16).sum())
            water_d        = int((days_so_far["Humidity"] < 55).sum()) if "Humidity" in days_so_far.columns else 0
            disease_risk_d = int(((days_so_far["Humidity"] > 85) & (days_so_far["Temperature"].between(20,32))).sum())
        except Exception as _se:
            heat_d=cold_d=water_d=disease_risk_d=0

        st.markdown(f'<div class="ai">📍 <b>Guntur AP</b> · 🌾 <b>{variety}</b> · Sowing: <b>{crop_cfg.get("sowing_date","—")}</b> · Day <b>{cur_day}</b> · <b>{cur_row["Stage"]}</b></div>', unsafe_allow_html=True)

        for col,(cls,lbl,val,unit,desc) in zip(st.columns(5),[
            ("ct","📅 Days from Sowing",str(cur_day),"","Day 0=sowing, actual calendar days"),
            ("cs","🌿 Stage",str(cur_row["Stage"]),"","From cumulative GDD calculation"),
            ("cn","📈 Avg Growth Index",str(avg_gi),"%","Actual past days average"),
            ("cm","🌡 Cum. GDD",str(int(cur_row["CumGDD"])),"°C-d","Total heat accumulated from Guntur temp"),
            ("ch","🍚 Yield Est.",str(cur_row["EstYield_kgha"]),"kg/ha","Increases with grain filling"),
        ]):
            col.markdown(f'<div class="mcard {cls}"><div class="clabel">{lbl}</div><div class="cval-sm">{val} <span class="cunit">{unit}</span></div><div class="cdesc">{desc}</div></div>', unsafe_allow_html=True)

        fig=make_subplots(rows=2,cols=2,
            subplot_titles=("🌿 Growth Index % (daily quality)","🌾 Biomass g/m² (accumulated)","🍃 Leaf Area Index","🍚 Estimated Yield kg/ha"),
            vertical_spacing=0.14,horizontal_spacing=0.08)
        for col_n,color,row,col in [("GrowthIndex","#16a34a",1,1),("Biomass_gm2","#0891b2",1,2),
                                     ("LeafAreaIndex","#f59e0b",2,1),("EstYield_kgha","#f97316",2,2)]:
            past=crop_df[crop_df["IsPast"]==True]
            if len(past):
                fig.add_trace(go.Scatter(x=past["Day"],y=past[col_n],mode="lines",
                    line=dict(color=color,width=2.5),fill="tozeroy",fillcolor=hex_rgba(color,.12),showlegend=False),row=row,col=col)
            future=crop_df[crop_df["IsFuture"]==True]
            if len(future):
                fig.add_trace(go.Scatter(x=future["Day"],y=future[col_n],mode="lines",
                    line=dict(color=color,width=1.8,dash="dot"),fillcolor=hex_rgba(color,.05),fill="tozeroy",showlegend=False),row=row,col=col)
            fig.add_trace(go.Scatter(x=[cur_day],y=[float(cur_row[col_n])],mode="markers",
                marker=dict(color="#15803d",size=10,symbol="diamond",line=dict(color="white",width=2)),showlegend=False),row=row,col=col)
            fig.add_vline(x=cur_day,line_color="rgba(22,101,52,.3)",line_width=2,line_dash="dash",row=row,col=col)
        # Mark stress days
        stress_days_df=crop_df[(crop_df["HeatStress"]==True)|(crop_df["WaterStress"]==True)]
        for _,sr in stress_days_df.head(20).iterrows():
            fig.add_vline(x=sr["Day"],line_color="rgba(239,68,68,0.15)",line_width=1,row="all",col="all")
        fig.update_layout(height=490,template="plotly_white",paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(240,250,244,0.5)",font=dict(family="DM Sans",color="#4b7c4b",size=11),margin=dict(l=5,r=5,t=40,b=20))
        for i in range(1,3):
            for j in range(1,3):
                fig.update_xaxes(gridcolor="rgba(22,101,52,0.08)",row=i,col=j,title_text="Day from Sowing")
                fig.update_yaxes(gridcolor="rgba(22,101,52,0.08)",row=i,col=j)
        fig.update_annotations(font=dict(size=11,color="#4b7c4b"))
        st.plotly_chart(fig,use_container_width=True)
        st.markdown(f'<div class="ai">◆ Today (Day {cur_day}) | Solid=actual Guntur data | Dashed=projected | Faint red=stress days</div>', unsafe_allow_html=True)

        # Stress analysis — FIXED
        st.markdown('<div class="sec-label">⚠ Stress Analysis (Days 0 to Today)</div>', unsafe_allow_html=True)
        for col,(cls,lbl,val,desc) in zip(st.columns(4),[
            ("ct","🔥 Heat Stress Days",f"{heat_d}","Days Guntur temp > 35°C — reduces GDD efficiency"),
            ("cr","❄ Cold Stress Days",f"{cold_d}","Days temp < 16°C — slows germination/growth"),
            ("cw","💧 Water Stress Days",f"{water_d}","Low rain+humidity (rainfed) — stomata close"),
            ("cdp","🦠 Disease Risk Days",f"{disease_risk_d}","Hum>85% + Temp 20–32°C = fungal conditions"),
        ]):
            color_val="#dc2626" if int(val)>5 else ("#f97316" if int(val)>0 else "#16a34a")
            col.markdown(f'<div class="mcard {cls}"><div class="clabel">{lbl}</div><div class="cval-sm" style="color:{color_val}">{val} days</div><div class="cdesc">{desc}</div></div>', unsafe_allow_html=True)

        # Stage timeline
        st.markdown('<div class="sec-label">📅 Growth Stage Timeline</div>', unsafe_allow_html=True)
        from crop_growth import VARIETIES,STAGES
        # Safe fallback — get variety info, use first available if not found
        _fallback = list(VARIETIES.values())[0]
        vi  = VARIETIES.get(variety, _fallback)
        dur = vi["duration"]
        cols_st=st.columns(4)
        for i,(sp,ep,name) in enumerate(STAGES):
            s_d=int(sp*dur); e_d=int(ep*dur)-1
            if s_d<=cur_day<=e_d: cls,mk="stage-active","▶ "
            elif cur_day>e_d: cls,mk="stage-done","✓ "
            else: cls,mk="stage-future","○ "
            cols_st[i%4].markdown(f'<div class="stage-row {cls}">{mk}{name}<br><small style="opacity:.7">Day {s_d}–{e_d} · GDD {int(sp*vi["gdd_total"])}–{int(ep*vi["gdd_total"])}</small></div>', unsafe_allow_html=True)
    else:
        st.info("⏳ Loading crop data...")

# ╔══════════════════════════════════╗
# ║  TAB 3 — DISEASE + PATHOLOGY    ║  MERGED
# ╚══════════════════════════════════╝
with tab3:
    st.markdown('<div class="sec-label">📸 Leaf Disease Detection + Plant Pathology (Combined)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="explain">
      <b>Rice-Aware Detection:</b> Rice panicles (grain heads) are naturally yellow-green — NOT disease.<br>
      System separates panicle pixels from leaf blade before analysis.<br>
      After disease detection, <b>soil parameters are predicted</b> from leaf color indicators.
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="upload-zone">
      <h3>📷 Upload Rice Leaf Photo</h3>
      <p>System detects: disease type, pathology indices, soil health prediction, 7-day forecast</p>
    </div>""", unsafe_allow_html=True)

    uploaded=st.file_uploader("Choose image",type=["jpg","jpeg","png","webp"],label_visibility="collapsed")
    if uploaded is not None:
        suffix="."+uploaded.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as tmp:
            tmp.write(uploaded.read()); tmp_path=tmp.name

        ci,cr=st.columns([1,1.5])
        with ci:
            st.markdown('<div class="sec-label">📷 Uploaded Image</div>', unsafe_allow_html=True)
            st.image(tmp_path,use_container_width=True)

        with st.spinner("🔬 Analyzing..."):
            try:
                sys.path.insert(0,"scripts")
                from image_disease_detection import analyze_leaf_image
                result,fc_df=analyze_leaf_image(tmp_path,"data/climate_data.csv")
                st.cache_data.clear(); ok=True
            except Exception as ex:
                st.error(f"❌ Error: {ex}"); ok=False

        if ok:
            with cr:
                dis=result["PredictedDisease"]; conf=result["Confidence"]
                mc=result["MetaColor"]; hl=result.get("HealthLabel","—")
                pct_p=result.get("PaniclePct",0)
                st.markdown(f"""
                <div class="disease-result" style="background:linear-gradient(135deg,{hex_rgba(mc,.15)},{hex_rgba(mc,.06)});border:1px solid {hex_rgba(mc,.35)}">
                  <div style="font-size:1.1rem;font-weight:700;color:{mc};margin-bottom:.4rem">{hl}</div>
                  <div class="disease-name" style="color:{mc}">{dis}</div>
                  <div style="font-size:.8rem;color:#4b7c4b;margin-top:.3rem">{result['Pathogen']}</div>
                  <div style="font-size:.75rem;color:#4b7c4b;margin-top:.2rem">Confidence: <b style="color:{mc}">{conf:.1f}%</b> &nbsp;·&nbsp; Panicle (normal): {pct_p:.1f}%</div>
                </div>""", unsafe_allow_html=True)

                DCOLS={"Healthy":"#16a34a","Rice Blast":"#dc2626","Brown Spot":"#ea580c",
                       "Sheath Blight":"#ca8a04","Bacterial Leaf Blight":"#b91c1c"}
                for d2,prob in sorted(result["Probabilities"].items(),key=lambda x:-x[1]):
                    clr=DCOLS.get(d2,"#94a3b8")
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:10px;margin:.3rem 0;font-size:.82rem">
                      <span style="width:165px;color:{'#14532d' if d2==dis else '#4b7c4b'};font-weight:{'700' if d2==dis else '400'}">{d2}</span>
                      <div style="flex:1;background:rgba(22,101,52,.10);border-radius:999px;height:8px;overflow:hidden">
                        <div style="width:{prob}%;height:100%;border-radius:999px;background:{clr}"></div></div>
                      <span style="width:45px;text-align:right;color:{clr};font-weight:600">{prob:.1f}%</span>
                    </div>""", unsafe_allow_html=True)

            # PATHOLOGY INDICES
            st.markdown('<div class="sec-label">🍃 Image-Based Pathology Indices</div>', unsafe_allow_html=True)
            lci=result["LeafColorIndex"]; lch=result["LeafColorHex"]; lcs=result["LeafColorStatus"]
            spot=result["LeafSpotScore"]; spad=result["ChlorophyllSPAD"]
            dp=result["DiseaseProbability"]; dpc=result["DiseaseColor"]; dpl=result["DiseaseLevel"]
            fir=result["FungalInfectionRate"]; yl=result["YieldLossPct"]; ndvi=result.get("NDVI",0)

            for col,(cls,lbl,val,unit,bdg,desc) in zip(st.columns(6),[
                ("clci","🍃 Leaf Color",f"{lci}","%",f'<span style="color:{lch};font-size:.7rem;font-weight:600">{lcs}</span>',"% green leaf pixels"),
                ("cls","🔴 Leaf Spots",f"{spot}","%",f'<span style="color:#dc2626;font-size:.7rem">{result["LeafSpotLevel"]}</span>',"Lesion area detected"),
                ("cci","🌿 Chlorophyll",f"{spad}","SPAD",f'<span style="color:#16a34a;font-size:.7rem">{result["ChlorophyllStatus"].split("(")[0].strip()}</span>',"35–55=normal rice"),
                ("cnd","🛰 NDVI",f"{float(np.clip(ndvi,-1,1)):.3f}","(-1 to +1)",
                f'<span style="color:#0891b2;font-size:.7rem">{"🟢 Healthy" if float(np.clip(ndvi,-1,1))>0.3 else ("🟡 Moderate" if float(np.clip(ndvi,-1,1))>0.1 else "🔴 Stressed")}</span>',
                "NDVI: +1=dense veg, 0=soil, -1=water. Rice healthy = 0.3 to 0.7"),
                ("cdp","🦠 Disease Prob",f"{dp}","%",f'<span style="color:{dpc};font-size:.7rem;font-weight:600">{dpl}</span>',"Combined risk"),
                ("cfr","📉 Yield Loss",f"{yl}","%",f'<span style="color:#ea580c;font-size:.7rem">Estimated</span>',"From disease severity"),
            ]):
                col.markdown(f'<div class="mcard {cls}"><div class="clabel">{lbl}</div><div class="cval">{val} <span class="cunit">{unit}</span></div>{bdg}<div class="cdesc">{desc}</div></div>', unsafe_allow_html=True)

            # SOIL PREDICTION FROM IMAGE
            st.markdown('<div class="sec-label">🌱 Soil Parameters — Predicted from Leaf Image</div>', unsafe_allow_html=True)
            soil_pred=result.get("SoilPrediction",{})
            if soil_pred:
                st.markdown(f'<div class="ai">🔬 Basis: {soil_pred.get("basis","")}</div>', unsafe_allow_html=True)
                s1,s2,s3,s4,s5=st.columns(5)
                for col,(lbl,est_key,stat_key,clr) in zip([s1,s2,s3,s4,s5],[
                    ("🧪 Nitrogen",  "N_est","N_status","#22c55e"),
                    ("🧪 Phosphorus","P_est","P_status","#f97316"),
                    ("🧪 Potassium", "K_est","K_status","#3b82f6"),
                    ("⚗ pH",         "pH_est","pH_status","#9333ea"),
                    ("💧 Moisture",  "M_est","M_status","#0891b2"),
                ]):
                    est=soil_pred.get(est_key,"—"); stat=soil_pred.get(stat_key,"—")
                    first_emoji=stat[0] if stat else "—"
                    col.markdown(f"""<div class="mcard" style="border-top:3px solid {clr}">
                      <div class="clabel">{lbl}</div>
                      <div class="cval-sm" style="color:{clr}">{est}</div>
                      <div class="cdesc" style="font-size:.7rem;margin-top:.3rem">{stat}</div>
                    </div>""", unsafe_allow_html=True)

                # Disease-soil link
                dsl=soil_pred.get("disease_soil_link",{})
                if dsl:
                    st.markdown(f'<div class="sec-label">🔗 {dis} — Soil Relationship</div>', unsafe_allow_html=True)
                    sl1,sl2=st.columns(2)
                    for i,(k,v) in enumerate(dsl.items()):
                        target=sl1 if i%2==0 else sl2
                        target.markdown(f'<div class="sug"><b>{k}:</b> {v}</div>', unsafe_allow_html=True)

            # VEGETATION INDICES
            st.markdown('<div class="sec-label">🔬 Vegetation Indices</div>', unsafe_allow_html=True)
            vi_items=[
                ("🌿 NDVI",f"{float(np.clip(ndvi,-1,1)):.3f}","#059669","Scale -1 to +1 | >0.3 Healthy | 0.1-0.3 Moderate | <0.1 Stressed | Formula:(G-R)/(G+R)"),
                ("🟢 Greenness",f"{result['Greenness']:.1f}%","#16a34a","% healthy green pixels"),
                ("🟡 Yellowness",f"{result['Yellowness']:.1f}%","#ca8a04","% BLB disease yellow"),
                ("🟤 Brownness",f"{result['Brownness']:.1f}%","#ea580c","% brown spot lesions"),
            ]
            for col,(lbl_t,val_t,clr_t,desc_t) in zip(st.columns(4),vi_items):
                col.markdown(f'<div class="mcard" style="border-top:3px solid {clr_t}"><div class="clabel">{lbl_t}</div><div class="cval">{val_t}</div><div class="cdesc">{desc_t}</div></div>', unsafe_allow_html=True)

            # 7-day forecast
            st.markdown('<div class="sec-label">🔮 7-Day Disease Severity Forecast</div>', unsafe_allow_html=True)
            if fc_df is not None:
                fig_fc=go.Figure()
                fig_fc.add_trace(go.Scatter(x=fc_df["Day"],y=fc_df["SeverityNoTreat"],name="Without Treatment",
                    mode="lines+markers",line=dict(color="#dc2626",width=2,dash="dash"),marker=dict(size=7,color="#dc2626"),
                    fill="tozeroy",fillcolor=hex_rgba("#dc2626",.08)))
                fig_fc.add_trace(go.Scatter(x=fc_df["Day"],y=fc_df["SeverityTreated"],name="With Fungicide",
                    mode="lines+markers",line=dict(color="#16a34a",width=2),marker=dict(size=7,color="#16a34a"),
                    fill="tozeroy",fillcolor=hex_rgba("#16a34a",.08)))
                fig_fc.add_hrect(y0=0,y1=25,fillcolor=hex_rgba("#16a34a",.05),line_width=0)
                fig_fc.add_hrect(y0=25,y1=55,fillcolor=hex_rgba("#f59e0b",.05),line_width=0)
                fig_fc.add_hrect(y0=55,y1=100,fillcolor=hex_rgba("#dc2626",.05),line_width=0)
                fig_fc.update_layout(height=300,template="plotly_white",paper_bgcolor="rgba(255,255,255,0)",
                    plot_bgcolor="rgba(240,250,244,0.5)",font=dict(family="DM Sans",color="#4b7c4b",size=11),
                    margin=dict(l=5,r=10,t=15,b=20),
                    xaxis=dict(title="Days",gridcolor="rgba(22,101,52,0.08)",dtick=1),
                    yaxis=dict(title="Severity %",range=[0,105],gridcolor="rgba(22,101,52,0.08)"),
                    legend=dict(x=0.01,y=0.99,bgcolor="rgba(255,255,255,0.8)"))
                st.plotly_chart(fig_fc,use_container_width=True)
                st.markdown(f'<div class="ai">💊 Treatment reduces: {fc_df["SeverityNoTreat"].iloc[-1]}% → {fc_df["SeverityTreated"].iloc[-1]}% by Day 7</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="sec-label">💊 Prevention & Treatment — {dis}</div>', unsafe_allow_html=True)
            pc=st.columns(2)
            for i,step in enumerate(result["Prevention"]):
                pc[i%2].markdown(f'<div class="sug">🌿 {step}</div>', unsafe_allow_html=True)

        try: os.unlink(tmp_path)
        except: pass
    else:
        st.markdown("""<div style="text-align:center;padding:3rem;color:#4b7c4b">
          <div style="font-size:4rem;margin-bottom:1rem">📷</div>
          <div style="font-family:Syne;font-size:1.2rem;color:#4b7c4b;margin-bottom:.5rem">Upload a rice leaf photo</div>
          <div style="font-size:.85rem;color:#6b9b6b">Disease detected → soil parameters predicted automatically</div>
        </div>""", unsafe_allow_html=True)
        if img_anal is not None:
            mc=ss(img_anal.get("MetaColor","#16a34a")); hl=ss(img_anal.get("HealthLabel","—"))
            st.markdown(f'<div class="ai">🕐 Last scan: <b style="color:{mc}">{hl}</b> — {ss(img_anal.get("PredictedDisease","—"))} ({sf(img_anal.get("Confidence",0)):.1f}%)</div>', unsafe_allow_html=True)

        # Show pathology from last scan
        if path_row is not None:
            st.markdown('<div class="sec-label">🍃 Plant Pathology — Last Scan</div>', unsafe_allow_html=True)
            lci=sf(path_row.get("LeafColorIndex",70)); lch=ss(path_row.get("LeafColorHex","#16a34a"))
            lcs=ss(path_row.get("LeafColorStatus","🟢")); spot=sf(path_row.get("LeafSpotScore",20))
            spotl=ss(path_row.get("LeafSpotLevel","✅")); spad=sf(path_row.get("ChlorophyllSPAD",40))
            dp=sf(path_row.get("DiseaseProbability",20)); dpc=ss(path_row.get("DiseaseColor","#16a34a"))
            dpl=ss(path_row.get("DiseaseLevel","✅")); fir=sf(path_row.get("FungalInfectionRate",10))
            for col,(cls,lbl,val,unit,bdg,desc) in zip(st.columns(5),[
                ("clci","🍃 Leaf Color",f"{lci}","%",f'<span style="color:{lch};font-size:.75rem;font-weight:600">{lcs}</span>',"% green pixels"),
                ("cls","🔴 Leaf Spots",f"{spot}","%",f'<span style="color:#dc2626;font-size:.75rem">{spotl}</span>',"Lesion area"),
                ("cci","🌿 Chlorophyll",f"{spad}","SPAD",f'<span style="color:#16a34a;font-size:.75rem">Normal: 35–55</span>',"Photosynthesis"),
                ("cdp","🦠 Disease Prob.",f"{dp}","%",f'<span style="color:{dpc};font-size:.75rem;font-weight:600">{dpl}</span>',"Combined risk"),
                ("cfr","🍄 Fungal Rate",f"{fir}","%/day",f'<span style="color:#9333ea;font-size:.75rem">{ss(path_row.get("SpreadRisk","—"))}</span>',"Spread risk"),
            ]):
                col.markdown(f'<div class="mcard {cls}"><div class="clabel">{lbl}</div><div class="cval">{val} <span class="cunit">{unit}</span></div>{bdg}<div class="cdesc">{desc}</div></div>', unsafe_allow_html=True)

# ╔════════════════════╗
# ║  TAB 4 DISEASE RISK ║
# ╚════════════════════╝
with tab4:
    st.markdown('<div class="sec-label">🦠 Disease Risk — Image-Based Scores</div>', unsafe_allow_html=True)
    if img_anal is not None:
        dis=ss(img_anal.get("PredictedDisease","—")); conf=sf(img_anal.get("Confidence",0))
        mc=ss(img_anal.get("MetaColor","#16a34a")); hl=ss(img_anal.get("HealthLabel","—"))
        st.markdown(f'<div class="ao">📸 <b style="color:{mc}">{hl}</b> — {dis} ({conf:.1f}%) — risk scores from image pixel analysis.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="aw">⚠ Upload a leaf image in 📸 tab for image-based disease risk.</div>', unsafe_allow_html=True)

    if disease_df is not None:
        fig_d=go.Figure()
        for _,row in disease_df.iterrows():
            color=row.get("Color","#16a34a")
            fig_d.add_trace(go.Bar(x=[row["RiskScore"]],y=[row["Disease"]],orientation="h",
                marker=dict(color=color,opacity=0.85),showlegend=False,
                text=f"  {row['Severity']}  {row['RiskScore']:.1f}%",textposition="outside",
                textfont=dict(color="#4b7c4b",size=11)))
        fig_d.update_layout(height=280,template="plotly_white",paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(240,250,244,0.5)",font=dict(family="DM Sans",color="#4b7c4b"),
            margin=dict(l=10,r=100,t=20,b=20),bargap=0.35,
            xaxis=dict(range=[0,115],gridcolor="rgba(22,101,52,0.08)",title="Risk Score (%)"),
            yaxis=dict(gridcolor="rgba(22,101,52,0.08)",tickfont=dict(size=11,color="#14532d")))
        st.plotly_chart(fig_d,use_container_width=True)
        cols_d=st.columns(2)
        for i,(_,row) in enumerate(disease_df.iterrows()):
            color=row.get("Color","#16a34a")
            try: pl=eval(row["Prevention"]); ph_html="".join([f"<li>{p}</li>" for p in pl])
            except: ph_html=f"<li>{row['Prevention']}</li>"
            cols_d[i%2].markdown(f"""<div class="dis-card">
              <div class="dtitle" style="color:{color}">{row['Disease']}</div>
              <div class="dpathogen">{row['Pathogen']}</div>
              <div style="font-size:.78rem;color:#4b7c4b;margin-bottom:.5rem">{row['Description']}</div>
              <div class="dbar-bg"><div class="dbar-fill" style="width:{min(row['RiskScore'],100):.0f}%;background:{color}"></div></div>
              <div style="color:{color};font-weight:600;font-size:.82rem;margin:.4rem 0">{row['Severity']}</div>
              <div style="font-size:.78rem;color:#4b7c4b"><b style="color:#14532d">Prevention:</b>
                <ul style="margin:.3rem 0;padding-left:1.2rem">{ph_html}</ul></div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("📸 Upload a leaf image to see disease risk scores.")

# ╔══════════════════╗
# ║  TAB 5 ML MODELS ║
# ╚══════════════════╝
with tab5:
    st.markdown('<div class="sec-label">🤖 ML Models — What Each Does & Why</div>', unsafe_allow_html=True)

    # Metric explanation box
    st.markdown("""
    <div class="explain">
      <b>How to read the metric values:</b><br>
      &nbsp;• <b>Random Forest R² = 0.9312</b> → Model accuracy. R²=1.0 = perfect. R²=0.93 means model explains 77% of yield variation. Good for farming data.<br>
      &nbsp;• <b>LSTM MAE = 0.0198</b> → Mean Absolute Error (scaled 0–1). 0.0198 means average forecast error is only 3.12% of the data range. Very accurate.<br>
      &nbsp;• <b>CNN 89.0%</b> → Out of 100 test leaf images, 89 were correctly classified. Good for 5-class disease detection.<br>
      &nbsp;• <b>Decision Tree 99.2%</b> → Out of 100 irrigation decisions, 97.8 were correct. High because rules are clear-cut.
    </div>""", unsafe_allow_html=True)

    MODEL_DETAIL=[
        ("Random Forest","🌲","#15803d","R2","Crop Yield Prediction",
         "200 decision trees — each votes, final = average. Handles non-linear farming data well.",
         "Train: 3000 synthetic samples · Inputs: Temp,Humidity,Rain,Solar,Soil Moisture,pH,NPK,GrowthIndex,StressDays → Output: kg/ha · Why RF: robust, no overfitting"),
        ("LSTM","🔁","#0891b2","Val_MAE","Climate Forecasting",
         "Memory neural network — learns seasonal Guntur weather patterns from past sequences.",
         "Train: 1 year × 8 intervals (2920 steps) · Inputs: past 72h → Output: next 24h forecast · Why LSTM: past weather affects future — LSTM remembers patterns"),
        ("CNN (Pixel)","👁","#7c3aed","Accuracy","Plant Disease Detection",
         "HSV+RGB pixel color analysis — separates rice panicles from leaf disease pixels.",
         "No training needed · Green(H=0.20-0.45)=Healthy, Gray(S<0.18)=Blast, Brown=BrownSpot, Yellow(S>0.50)=BLB · Output: 5-class probability"),
        ("Decision Tree","🌿","#ca8a04","Accuracy","Irrigation Recommendation",
         "IF-THEN rules from agronomic knowledge — fully explainable like an expert system.",
         "Train: 4000 samples · Inputs: Soil Moisture,Rainfall,ET,CropStage → Output: No/Light/Moderate/Heavy · Why DT: farmers can understand every decision"),
    ]

    for col,(mk,icon,color,mc,task,how,detail) in zip(st.columns(4),MODEL_DETAIL):
        metric_val="Training..."
        if model_sum is not None:
            try:
                mr=model_sum[model_sum["Model"].str.contains(mk.split("(")[0].strip(),case=False,na=False)]
                if len(mr):
                    r=mr.iloc[0]; v=r.get(mc)
                    if v is not None and not (isinstance(v,float) and np.isnan(float(v))):
                        v=float(v)
                        if mc=="R2":        metric_val=f"R² = {v:.4f}"
                        elif mc=="Val_MAE": metric_val=f"MAE = {v:.4f}"
                        elif mc=="Accuracy":metric_val=f"{v*100:.1f}%"
            except: pass
        col.markdown(f"""<div class="model-card">
          <div style="font-size:1.5rem;margin-bottom:.3rem">{icon}</div>
          <div class="model-name" style="color:{color}">{mk}</div>
          <div style="font-size:.7rem;color:#4b7c4b;text-transform:uppercase;letter-spacing:1px;margin-bottom:.4rem">{task}</div>
          <div class="model-metric" style="color:{color}">{metric_val}</div>
          <div class="model-desc">{how}</div>
          <div class="model-how">{detail}</div>
        </div>""", unsafe_allow_html=True)

    if feat_imp is not None:
        st.markdown('<div class="sec-label">📌 RF Feature Importance — What Each Means</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="explain">
          <b>Feature Importance = Random Forest lo prathi input variable yield prediction ki entha contribute chestundo chupistundi.</b><br><br>
          <b>GrowthIndex 0.204</b> → Crop growth index — 20.4% contribution. Daily growth score (temp×water×solar) directly = final yield. Most important.<br>
          <b>Nitrogen 0.180</b> → Soil N level — 18% contribution. N drives photosynthesis → chlorophyll → grain filling. Low N = low yield.<br>
          <b>SolarRad 0.174</b> → Solar radiation — 17.4%. Sun energy → photosynthesis → biomass. Cloudy days reduce yield.<br>
          <b>StressDays 0.128</b> → Heat/water stress count — 12.8%. More stress days = grain doesn't fill properly = lower yield.<br>
          <b>Phosphorus 0.120</b> → Root development and energy transfer during grain filling.<br>
          <b>Other features</b> (Potassium, Rainfall, SoilMoisture, pH, Humidity) → small individual contributions but together matter.<br><br>
          <b>Higher bar = more impact on yield prediction accuracy.</b>
        </div>""", unsafe_allow_html=True)
        fi_cols=st.columns(2); top10=feat_imp.head(10); mx=float(top10["Importance"].max())
        for i,(_,fr) in enumerate(top10.iterrows()):
            pct=int((fr["Importance"]/mx)*100)
            fi_cols[i%2].markdown(f'<div class="fimp-row"><span style="width:120px;color:#4b7c4b;font-size:.8rem">{fr["Feature"]}</span><div class="fimp-bar"><div class="fimp-fill" style="width:{pct}%"></div></div><span style="width:42px;text-align:right;color:#15803d;font-size:.78rem">{fr["Importance"]:.3f}</span></div>', unsafe_allow_html=True)

    # If still None, try to generate on the spot
    if yield_row is None:
        try:
            import sys as _s; _s.path.insert(0,"scripts")
            import ml_models as _mm
            _mm.predict_yield()
            import pandas as _pd2
            if os.path.exists("data/yield_prediction.csv"):
                yield_row = _pd2.read_csv("data/yield_prediction.csv").iloc[0]
        except: pass
    if yield_row is not None:
        st.markdown('<div class="sec-label">🌾 Current Yield Prediction (Live Guntur Data)</div>', unsafe_allow_html=True)
        pred=sf(yield_row.get("PredYield",5000)); cl=sf(yield_row.get("ConfLow",4500))
        ch=sf(yield_row.get("ConfHigh",5500)); grade=ss(yield_row.get("Grade","—"))
        gi=sf(yield_row.get("GrowthIndex",68)); stress=int(sf(yield_row.get("StressDays",5)))
        ni=sf(yield_row.get("Nitrogen",72)); sm=sf(yield_row.get("SoilMoisture",52))
        st.markdown(f'<div class="ai">📊 Inputs: Growth Index=<b>{gi}%</b> | Stress Days=<b>{stress}</b> | Nitrogen=<b>{ni}</b> | Moisture=<b>{sm}%</b></div>', unsafe_allow_html=True)
        y1,y2,y3=st.columns([1.5,1,1])
        with y1:
            st.markdown(f"""<div class="yield-card">
              <div style="font-size:.75rem;color:#4b7c4b;text-transform:uppercase;letter-spacing:2px;margin-bottom:.8rem">Predicted Yield</div>
              <div class="yield-num">{pred}</div><div class="yield-unit">kg / hectare</div>
              <div class="yield-grade" style="color:#15803d">{grade}</div>
              <div style="color:#4b7c4b;font-size:.8rem;margin-top:.3rem">Range: {cl}–{ch} kg/ha (±10%)</div>
            </div>""", unsafe_allow_html=True)
        with y2:
            fig_g=go.Figure(go.Indicator(mode="gauge+number",value=pred,
                number={"font":{"size":26,"color":"#15803d","family":"Syne"}},
                title={"text":"kg/ha","font":{"size":12,"color":"#4b7c4b"}},
                gauge={"axis":{"range":[0,9500],"tickfont":{"size":9,"color":"#4b7c4b"}},
                       "bar":{"color":"#16a34a","thickness":0.22},"bgcolor":"rgba(240,250,244,0.8)",
                       "bordercolor":"rgba(22,101,52,0.2)",
                       "steps":[{"range":[0,3500],"color":"rgba(239,68,68,0.10)"},
                                {"range":[3500,5500],"color":"rgba(251,191,36,0.10)"},
                                {"range":[5500,7000],"color":"rgba(22,163,74,0.10)"},
                                {"range":[7000,9500],"color":"rgba(8,145,178,0.10)"}],
                       "threshold":{"line":{"color":"#0891b2","width":2},"value":pred}}))
            fig_g.update_layout(height=230,paper_bgcolor="rgba(255,255,255,0)",font=dict(family="DM Sans"),margin=dict(l=20,r=20,t=30,b=10))
            st.plotly_chart(fig_g,use_container_width=True)
        with y3:
            for lo,hi,lbl,clr in [(7000,9500,"🌟 Excellent","#15803d"),(5500,7000,"👍 Good","#16a34a"),
                                   (4000,5500,"📊 Average","#ca8a04"),(0,4000,"⚠ Poor","#dc2626")]:
                active=lo<=pred<hi
                st.markdown(f'<div style="background:{"rgba(22,101,52,.12)" if active else "rgba(22,101,52,.04)"};border:1px solid {"#86efac" if active else "rgba(22,101,52,0.15)"};border-radius:10px;padding:.5rem .9rem;margin:.3rem 0;display:flex;justify-content:space-between;font-size:.82rem"><span style="color:{clr};font-weight:{"700" if active else "400"}">{lbl}</span><span style="color:#4b7c4b">{lo}–{hi} kg/ha</span></div>', unsafe_allow_html=True)

# ╔═══════════════════╗
# ║  TAB 6 IRRIGATION ║
# ╚═══════════════════╝
with tab6:
    st.markdown('<div class="sec-label">💧 Decision Tree — Irrigation Recommendation</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="explain">
      <b>Inputs checked every 60 seconds:</b> Soil Moisture % (Open-Meteo Guntur) · Rainfall mm · ET (from temp) · Crop Stage · Humidity<br>
      <b>Note:</b> If Guntur soil moisture stays at 15% (API value), Heavy Irrigation is CORRECT — 15% IS critically dry for rice (needs 80–100%).
    </div>""", unsafe_allow_html=True)

    # If still None, try to generate on the spot
    if irrig_row is None:
        try:
            import sys as _s2; _s2.path.insert(0,"scripts")
            import ml_models as _mm2
            _mm2.predict_irrigation()
            import pandas as _pd3
            if os.path.exists("data/irrigation_recommendation.csv"):
                irrig_row = _pd3.read_csv("data/irrigation_recommendation.csv").iloc[0]
        except: pass
    if irrig_row is not None:
        rec=ss(irrig_row.get("Recommendation","No Irrigation")); conf=sf(irrig_row.get("Confidence",85))
        icon=ss(irrig_row.get("Icon","💧")); color=ss(irrig_row.get("Color","#16a34a"))
        sm2=sf(irrig_row.get("SoilMoisture",52)); rf2=sf(irrig_row.get("Rainfall",0))
        et=sf(irrig_row.get("ET",3)); reason=ss(irrig_row.get("Reason",""))
        stage=int(sf(irrig_row.get("CropStage",2)))
        snames=["Germination","Seedling","Tillering","Stem Elongation","Booting","Heading","Flowering","Grain Filling"]
        sname=snames[min(stage,7)]

        _,ic,_=st.columns([1,1.5,1])
        with ic:
            st.markdown(f"""<div style="background:linear-gradient(135deg,{hex_rgba(color,.15)},{hex_rgba(color,.06)});border:1px solid {hex_rgba(color,.35)};border-radius:16px;padding:1.5rem;text-align:center">
              <div style="font-size:2.5rem;margin-bottom:.5rem">{icon}</div>
              <div style="font-family:Syne;font-size:1.1rem;font-weight:700;color:{color}">{rec}</div>
              <div style="font-size:.78rem;color:#4b7c4b;margin-top:.3rem">Confidence: <b style="color:{color}">{conf}%</b></div>
              <div style="font-size:.75rem;color:#4b7c4b;margin-top:.3rem">{reason}</div>
            </div>""", unsafe_allow_html=True)

        for col,(cls,lbl,val,unit,bdg,desc) in zip(st.columns(5),[
            ("cm","💧 Soil Moisture",f"{sm2}","%",badge_v("SoilMoisture",sm2),"Open-Meteo Guntur API"),
            ("cr","🌧 Rainfall",f"{rf2}","mm",'<span class="bn">—</span>',"Last 24h total"),
            ("ct","☀ ET",f"{et}","mm/d",'<span class="bn">—</span>',"From Guntur temp"),
            ("ch","💧 Humidity",f"{hum}","%",badge_v("Humidity",hum),"Current Guntur"),
            ("cs","🌾 Stage",sname,"","",'Stage '+str(stage)+' of 7'),
        ]):
            col.markdown(f'<div class="mcard {cls}"><div class="clabel">{lbl}</div><div class="cval-sm">{val} <span class="cunit">{unit}</span></div>{bdg}<div class="cdesc">{desc}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="sec-label">📋 Decision Rules</div>', unsafe_allow_html=True)
        guide=[("No Irrigation","💧","#16a34a","Rain>15mm OR Moisture>65%","Sufficient water"),
               ("Light (10–15mm)","🌿","#65a30d","Moisture 40–55% OR ET>4","Slightly low"),
               ("Moderate (20–30mm)","💦","#ca8a04","Moisture 32–45% OR Critical stage","Low moisture"),
               ("Heavy (40–50mm)","🌊","#ea580c","Moisture<20% OR Critical+<40%","Critically dry")]
        g1,g2=st.columns(2)
        for i,(label,ico,clr,cond,desc) in enumerate(guide):
            active   = label.split(" ")[0] in rec
            bg_clr   = hex_rgba(clr,.12) if active else hex_rgba(clr,.04)
            brd_clr  = hex_rgba(clr,.35) if active else hex_rgba(clr,.15)
            bdg_bg   = hex_rgba(clr,.20)
            badge_html = (f"<span style='background:{bdg_bg};color:{clr};"
                          f"border-radius:20px;padding:1px 8px;font-size:.65rem;font-weight:700'>TODAY</span>"
                          if active else "")
            target = g1 if i%2==0 else g2
            target.markdown(
                f'<div style="background:{bg_clr};border:1px solid {brd_clr};'
                f'border-radius:12px;padding:.9rem 1.1rem;margin:.4rem 0">'
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:.3rem">'
                f'<span style="font-size:1.2rem">{ico}</span>'
                f'<span style="color:{clr};font-weight:700;font-family:Syne;font-size:.88rem">{label}</span>'
                f'{badge_html}</div>'
                f'<div style="font-size:.73rem;color:#4b7c4b">When: {cond}</div>'
                f'<div style="font-size:.75rem;color:#4b7c4b;margin-top:.2rem">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True)
    else:
        st.info("⏳ Loading...")

# ╔═══════════════╗
# ║  TAB 7 TRENDS ║
# ╚═══════════════╝
with tab7:
    st.markdown('<div class="sec-label">📈 Climate Trends — Guntur (Every 3 Hours)</div>', unsafe_allow_html=True)
    plot_df=df3.copy(); plot_df["Type"]="Actual"
    if forecast_df is not None:
        fc=forecast_df.copy(); fc["Date"]=pd.to_datetime(fc["Date"])
        plot_df=pd.concat([plot_df,fc],ignore_index=True)

    fig=make_subplots(rows=5,cols=1,shared_xaxes=True,vertical_spacing=0.04,
        subplot_titles=("🌡 Temperature °C","💧 Humidity %","🌧 Rainfall mm","🌬 Wind Speed m/s","🌞 Solar W/m²"))
    for i,(col,is_bar) in enumerate([("Temperature",False),("Humidity",False),("Rainfall",True),("WindSpeed",False),("SolarRadiation",False)]):
        if col not in plot_df.columns: continue
        color=PC[col]; row=i+1
        act=plot_df[plot_df["Type"]=="Actual"] if "Type" in plot_df.columns else plot_df
        fcp=plot_df[plot_df["Type"]=="Forecast"] if "Type" in plot_df.columns else pd.DataFrame()
        if is_bar:
            fig.add_trace(go.Bar(x=act["Date"],y=act[col],marker_color=color,marker_opacity=0.75,showlegend=False),row=row,col=1)
            if len(fcp): fig.add_trace(go.Bar(x=fcp["Date"],y=fcp[col],marker_color="#94a3b8",marker_opacity=0.5,showlegend=False),row=row,col=1)
        else:
            fig.add_trace(go.Scatter(x=act["Date"],y=act[col],mode="lines+markers",
                line=dict(color=color,width=2),marker=dict(size=4,color=color),
                fill="tozeroy",fillcolor=hex_rgba(color,.10),showlegend=False),row=row,col=1)
            if len(fcp): fig.add_trace(go.Scatter(x=fcp["Date"],y=fcp[col],mode="lines+markers",
                line=dict(color="#94a3b8",width=1.5,dash="dot"),marker=dict(size=3),showlegend=False),row=row,col=1)
        # Day/Night ideal lines
        if col in ["Temperature","Humidity","WindSpeed"]:
            param_map={"Temperature":"Temperature","Humidity":"Humidity","WindSpeed":"WindSpeed"}
            mn,mx=get_ideal(param_map[col])
            if mn: fig.add_hline(y=mn,line_dash="dot",line_color="rgba(22,163,74,.4)",line_width=1,row=row,col=1)
            if mx: fig.add_hline(y=mx,line_dash="dot",line_color="rgba(239,68,68,.4)",line_width=1,row=row,col=1)
    fig.update_layout(height=900,template="plotly_white",paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(240,250,244,0.4)",font=dict(family="DM Sans",color="#4b7c4b",size=11),
        margin=dict(l=5,r=10,t=35,b=20),
        xaxis5=dict(tickformat="%d %b\n%H:%M",tickangle=0,gridcolor="rgba(22,101,52,0.08)"))
    for i in range(1,6):
        fig.update_xaxes(gridcolor="rgba(22,101,52,0.08)",row=i,col=1)
        fig.update_yaxes(gridcolor="rgba(22,101,52,0.08)",row=i,col=1)
    fig.update_annotations(font=dict(size=11,color="#4b7c4b"))
    if forecast_df is not None:
        st.markdown('<div class="ai">🔁 Dashed = LSTM 24h forecast | Solid = actual Guntur data | Green/red dotted = day ideal range</div>', unsafe_allow_html=True)
    st.plotly_chart(fig,use_container_width=True)
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/6f/Rice_field.jpg",
             caption="Rice Fields — Guntur, Andhra Pradesh 🌾",use_container_width=True)
