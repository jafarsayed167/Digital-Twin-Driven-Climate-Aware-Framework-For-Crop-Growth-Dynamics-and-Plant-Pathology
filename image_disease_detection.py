"""
IMAGE-BASED PLANT DISEASE DETECTION + SOIL PREDICTION
=======================================================
After disease detection, soil parameters are predicted based on:
- Leaf color (chlorophyll → nitrogen estimate)
- Spot score (lesion severity → pH stress estimate)
- Vegetation indices (NDVI → overall soil health)
- Disease type (each disease correlates with soil deficiencies)
"""

import numpy as np
import pandas as pd
import os, joblib, warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

os.makedirs("data/models", exist_ok=True)

CLASSES = ["Healthy","Rice Blast","Brown Spot","Sheath Blight","Bacterial Leaf Blight"]

DISEASE_META = {
    "Healthy": {
        "pathogen":"None",
        "description":"Leaf tissue appears healthy. Green color, no lesions.",
        "severity_range":(0,10),"yield_loss":(0,0),"color":"#16a34a",
        "health_label":"✅ Healthy",
        "soil_stress": {"N":"Normal","P":"Normal","K":"Normal","pH":"Normal","Moisture":"Normal"},
        "prevention":["Continue monitoring every 3–4 days","Maintain balanced NPK fertilization",
                      "Ensure proper drainage","Preventive neem spray @ 5mL/L every 15 days"],
    },
    "Rice Blast": {
        "pathogen":"Magnaporthe oryzae (Fungal)",
        "description":"Diamond-shaped gray/white lesions with dark borders. Most destructive.",
        "severity_range":(45,95),"yield_loss":(15,70),"color":"#dc2626",
        "health_label":"🔴 Unhealthy — Rice Blast",
        "soil_stress": {"N":"Excess (promotes blast)","P":"Low","K":"Low","pH":"Acidic risk","Moisture":"High (waterlogged)"},
        "prevention":["Apply Tricyclazole 75 WP @ 0.6g/L immediately",
                      "Spray Propiconazole 25 EC @ 1mL/L as curative",
                      "Use blast-resistant varieties: IR64, Swarna, MTU1010",
                      "Avoid excess nitrogen — split urea doses","Drain field"],
    },
    "Brown Spot": {
        "pathogen":"Bipolaris oryzae (Fungal)",
        "description":"Oval brown lesions with yellow halo. Linked to potassium/silicon deficiency.",
        "severity_range":(25,70),"yield_loss":(5,45),"color":"#ea580c",
        "health_label":"🟠 Unhealthy — Brown Spot",
        "soil_stress": {"N":"Low-Moderate","P":"Low","K":"Deficient","pH":"Slightly acidic","Moisture":"Variable"},
        "prevention":["Apply Mancozeb 75 WP @ 2g/L","Apply Propiconazole 25 EC @ 1mL/L",
                      "Apply Potassium (MOP) @ 25 kg/acre immediately",
                      "Seed treatment: Thiram 75 WS @ 2g/kg","Maintain pH 6.0–7.0"],
    },
    "Sheath Blight": {
        "pathogen":"Rhizoctonia solani (Fungal)",
        "description":"Pale water-soaked lesions on sheath. Dense canopy + high humidity.",
        "severity_range":(30,80),"yield_loss":(10,50),"color":"#ca8a04",
        "health_label":"🟡 Unhealthy — Sheath Blight",
        "soil_stress": {"N":"Excess","P":"Normal","K":"Low","pH":"Near neutral","Moisture":"Waterlogged"},
        "prevention":["Apply Hexaconazole 5 EC @ 2mL/L","Apply Validamycin 3 SL @ 2.5mL/L",
                      "Reduce plant density — row spacing 20cm","Drain field","Avoid excess N"],
    },
    "Bacterial Leaf Blight": {
        "pathogen":"Xanthomonas oryzae pv. oryzae (Bacterial)",
        "description":"Yellow-white wilting from leaf margins. Spreads through water.",
        "severity_range":(20,75),"yield_loss":(6,40),"color":"#b91c1c",
        "health_label":"🟡 Unhealthy — Bacterial Leaf Blight",
        "soil_stress": {"N":"Variable","P":"Low","K":"Deficient","pH":"Alkaline risk","Moisture":"Flood-irrigated"},
        "prevention":["Apply Copper Oxychloride 50 WP @ 3g/L",
                      "Spray Streptomycin + Copper 0.2%","Use resistant varieties",
                      "Avoid flood irrigation after rain","Remove infected debris"],
    },
}


def compute_hsv(rgb):
    maxc=np.max(rgb,axis=2); minc=np.min(rgb,axis=2); V=maxc
    S=np.where(maxc>1e-6,(maxc-minc)/maxc,0.)
    delta=maxc-minc+1e-8
    r,g,b=rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
    H=np.where(maxc==r,((g-b)/delta)%6,np.where(maxc==g,(b-r)/delta+2,(r-g)/delta+4))/6.
    return H%1.,S,V


def analyze_pixels(image_path):
    from PIL import Image
    img=Image.open(image_path).convert("RGB"); img=img.resize((256,256),Image.LANCZOS)
    rgb=np.array(img,dtype=np.float32)/255.
    r,g,b=rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]; total=256*256
    H,S,V=compute_hsv(rgb)

    leaf_green   =(H>=0.20)&(H<=0.45)&(S>0.20)&(V>0.18)
    panicle_mask =(H>=0.10)&(H<=0.22)&(S>0.15)&(S<0.60)&(V>0.45)
    blb_yellow   =(H>=0.05)&(H<=0.18)&(S>0.50)&(V<0.80)
    brown_lesion =((H<=0.07)|(H>=0.93))&(S>0.30)&(V>0.15)&(V<0.70)
    blast_gray   =(S<0.18)&(V>0.50)&~leaf_green&~panicle_mask
    necrosis     =(V<0.15)
    sheath_pale  =(S<0.20)&(V>0.65)&~leaf_green&~blast_gray

    gp=float(leaf_green.sum()/total); pp=float(panicle_mask.sum()/total)
    byp=float(blb_yellow.sum()/total); brp=float(brown_lesion.sum()/total)
    grp=float(blast_gray.sum()/total); nec=float(necrosis.sum()/total)
    spp=float(sheath_pale.sum()/total)
    lesion_pct=float(np.clip(byp+brp+grp+nec,0,1))
    mean_r=float(r.mean()); mean_g=float(g.mean()); mean_b=float(b.mean())
    std_r=float(r.std()); std_g=float(g.std()); std_b=float(b.std())
    greenness=float(mean_g/(mean_r+mean_g+mean_b+1e-6))
    exg=float(2*mean_g-mean_r-mean_b)
    ndvi_proxy=float((mean_g-mean_r)/(mean_g+mean_r+1e-6))
    texture=float((std_r+std_g+std_b)/3)

    return {"green_pct":round(gp,4),"panicle_pct":round(pp,4),"blb_yellow":round(byp,4),
            "brown_pct":round(brp,4),"gray_pct":round(grp,4),"necrosis_pct":round(nec,4),
            "sheath_pct":round(spp,4),"lesion_pct":round(lesion_pct,4),
            "greenness":round(greenness,4),"exg":round(exg,4),"ndvi_proxy":round(ndvi_proxy,4),
            "texture":round(texture,4),"mean_r":round(mean_r,4),"mean_g":round(mean_g,4),"mean_b":round(mean_b,4)}


def classify_disease(feats):
    gp=feats["green_pct"]; byp=feats["blb_yellow"]; brp=feats["brown_pct"]
    grp=feats["gray_pct"]; nec=feats["necrosis_pct"]; spp=feats["sheath_pct"]
    lp=feats["lesion_pct"]; txt=feats["texture"]; pp=feats["panicle_pct"]
    scores={}

    if gp>0.40 and lp<0.05: healthy=90+gp*10-lp*100
    elif gp>0.25 and lp<0.10: healthy=72+gp*15-lp*80
    elif gp>0.15 and lp<0.18: healthy=50+gp*12-lp*60
    else: healthy=max(5,30+gp*8-lp*50)
    if pp>0.05: healthy=min(healthy+pp*15,95)
    scores["Healthy"]=float(np.clip(healthy,0,95))

    blast=grp*220+nec*90+txt*70-gp*20
    if grp>0.03 and nec>0.01 and txt>0.06: blast+=30
    if grp>0.06: blast+=20
    scores["Rice Blast"]=float(np.clip(blast,0,95))

    brown=brp*230+nec*50-gp*10
    if brp>0.04 and gp>0.15: brown+=25
    if brp>0.08: brown+=20
    scores["Brown Spot"]=float(np.clip(brown,0,95))

    sheath=spp*180+grp*50+txt*50-gp*15
    if spp>0.04 and txt>0.06: sheath+=20
    scores["Sheath Blight"]=float(np.clip(sheath,0,95))

    blb=byp*250+spp*40-gp*20
    if byp>0.06: blb+=30
    if byp>0.12: blb+=20
    scores["Bacterial Leaf Blight"]=float(np.clip(blb,0,95))

    total_s=sum(scores.values()) or 1
    probs={k:round(v/total_s*100,1) for k,v in scores.items()}
    predicted=max(probs,key=probs.get)
    return predicted,probs[predicted],probs


def predict_soil_from_image(feats, predicted, pathology):
    """
    Predict soil parameter status from leaf image.
    Based on: leaf color (N indicator), lesion type, vegetation indices.
    """
    gp    = feats["green_pct"]
    lp    = feats["lesion_pct"]
    ndvi  = feats["ndvi_proxy"]
    exg   = feats["exg"]
    brp   = feats["brown_pct"]
    byp   = feats["blb_yellow"]
    spad  = pathology["ChlorophyllSPAD"]
    meta  = DISEASE_META[predicted]

    # Nitrogen — estimated from chlorophyll (SPAD proxy)
    if spad >= 45:   N_est, N_status = round(np.random.uniform(90,120),1),  "🟢 Sufficient (90–120 kg/ha)"
    elif spad >= 35: N_est, N_status = round(np.random.uniform(70,90),1),   "🟡 Moderate (70–90 kg/ha)"
    elif spad >= 25: N_est, N_status = round(np.random.uniform(45,70),1),   "🟠 Low (45–70 kg/ha) — Apply urea"
    else:            N_est, N_status = round(np.random.uniform(20,45),1),   "🔴 Deficient (<45 kg/ha) — Urgent N needed"

    # Phosphorus — brown spot indicates P deficiency
    if predicted == "Brown Spot" or brp > 0.05:
        P_est, P_status = round(np.random.uniform(20,40),1), "🟠 Low (20–40 kg/ha) — Apply DAP"
    elif gp > 0.45:
        P_est, P_status = round(np.random.uniform(45,65),1), "🟢 Good (45–65 kg/ha)"
    else:
        P_est, P_status = round(np.random.uniform(35,50),1), "🟡 Moderate (35–50 kg/ha)"

    # Potassium — BLB and sheath blight indicate K issues
    if predicted in ["Bacterial Leaf Blight","Sheath Blight","Brown Spot"] or byp > 0.06:
        K_est, K_status = round(np.random.uniform(25,50),1), "🟠 Deficient (25–50 kg/ha) — Apply MOP"
    elif gp > 0.45:
        K_est, K_status = round(np.random.uniform(60,85),1), "🟢 Good (60–85 kg/ha)"
    else:
        K_est, K_status = round(np.random.uniform(45,65),1), "🟡 Moderate (45–65 kg/ha)"

    # pH — estimated from disease type and leaf color
    if predicted == "Rice Blast":
        pH_est, pH_status = round(np.random.uniform(4.8,5.8),1), "🟠 Acidic risk (4.8–5.8) — Add lime"
    elif predicted == "Bacterial Leaf Blight":
        pH_est, pH_status = round(np.random.uniform(7.0,7.8),1), "🟡 Slightly alkaline (7.0–7.8)"
    elif gp > 0.45:
        pH_est, pH_status = round(np.random.uniform(6.0,7.0),1), "🟢 Optimal (6.0–7.0)"
    else:
        pH_est, pH_status = round(np.random.uniform(5.5,6.5),1), "🟡 Moderate (5.5–6.5)"

    # Soil moisture — from disease type
    if predicted in ["Rice Blast","Sheath Blight"]:
        M_est, M_status = round(np.random.uniform(80,100),1), "🟠 High/Waterlogged (>80%) — Improve drainage"
    elif predicted == "Brown Spot":
        M_est, M_status = round(np.random.uniform(40,60),1),  "🟡 Variable (40–60%) — Monitor"
    elif lp < 0.05 and gp > 0.40:
        M_est, M_status = round(np.random.uniform(70,90),1),  "🟢 Good (70–90%)"
    else:
        M_est, M_status = round(np.random.uniform(30,55),1),  "🟠 Low (<55%) — Irrigate"

    return {
        "N_est":N_est,"N_status":N_status,
        "P_est":P_est,"P_status":P_status,
        "K_est":K_est,"K_status":K_status,
        "pH_est":pH_est,"pH_status":pH_status,
        "M_est":M_est,"M_status":M_status,
        "disease_soil_link": meta["soil_stress"],
        "basis": f"Estimated from: SPAD={pathology['ChlorophyllSPAD']}, NDVI={round(ndvi*100,1)}%, Disease={predicted}"
    }


def compute_pathology(feats, predicted):
    lp=feats["lesion_pct"]; gp=feats["green_pct"]
    exg=feats["exg"]; brp=feats["brown_pct"]; nec=feats["necrosis_pct"]; pp=feats["panicle_pct"]

    lci=round(float(np.clip(gp*160,0,100)),1)
    if lci>=70:   lci_s,lci_h="🟢 Healthy Green","#16a34a"
    elif lci>=50: lci_s,lci_h="🟡 Slightly Pale","#ca8a04"
    elif lci>=30: lci_s,lci_h="🟠 Moderate Damage","#ea580c"
    else:         lci_s,lci_h="🔴 Severe Damage","#dc2626"

    spot=round(float(np.clip(lp*300,0,100)),1)
    if spot<8:    spot_l="✅ None"
    elif spot<25: spot_l="🟡 Low"
    elif spot<50: spot_l="🟠 Moderate"
    elif spot<75: spot_l="🔴 High"
    else:         spot_l="🔴 Severe"

    spad=round(float(np.clip(exg*90+gp*35+12,5,80)),1)
    if spad>=45:   ci_s="🟢 Excellent"
    elif spad>=35: ci_s="🟢 Normal"
    elif spad>=25: ci_s="🟡 Moderate Deficiency"
    elif spad>=15: ci_s="🟠 Low"
    else:          ci_s="🔴 Critical Chlorosis"

    ndvi=round(feats["ndvi_proxy"]*100,1)
    meta=DISEASE_META[predicted]; smn,smx=meta["severity_range"]
    sev=round(float(np.clip(lp*120,0,12)),1) if predicted=="Healthy" else round(float(np.clip(smn+lp*(smx-smn)*6,smn,smx)),1)
    dp_val=round(float(np.clip(sev*0.65+spot*0.35*(1-gp*0.5),0,100)),1)
    if predicted=="Healthy": dp_val=min(dp_val,12)
    if dp_val<15:   dp_l,dp_c="✅ Very Low","#16a34a"
    elif dp_val<35: dp_l,dp_c="🟡 Low","#65a30d"
    elif dp_val<55: dp_l,dp_c="🟠 Moderate","#ca8a04"
    elif dp_val<75: dp_l,dp_c="🔴 High","#ea580c"
    else:           dp_l,dp_c="🔴 Critical","#dc2626"

    fir=round(float(np.clip((brp+nec)*350,0,100)),1)
    spore=round(float(np.clip(lp*500,0,100)),1)
    if fir<10:   spread="✅ Negligible"
    elif fir<30: spread="🟡 Low"
    elif fir<55: spread="🟠 Moderate"
    elif fir<75: spread="🔴 Rapid"
    else:        spread="🔴 Epidemic"

    yln,ylx=meta["yield_loss"]
    yl=round(float(np.clip(yln+(sev/100)*(ylx-yln),0,100)),1)

    return {"HealthLabel":meta["health_label"],
            "LeafColorIndex":lci,"LeafColorStatus":lci_s,"LeafColorHex":lci_h,
            "LeafSpotScore":spot,"LeafSpotLevel":spot_l,
            "ChlorophyllSPAD":spad,"ChlorophyllStatus":ci_s,"NDVI":ndvi,
            "DiseaseSeverity":sev,"DiseaseProbability":dp_val,"DiseaseLevel":dp_l,"DiseaseColor":dp_c,
            "FungalInfectionRate":fir,"SporeViability":spore,"SpreadRisk":spread,"YieldLossPct":yl,"PaniclePct":round(pp*100,1)}


def build_risk_df(probs,feats,pathology):
    lp=feats["lesion_pct"]; rows=[]
    for cls in CLASSES:
        meta=DISEASE_META[cls]; prob=probs.get(cls,0)
        risk=round(prob*(1+lp*3) if cls!="Healthy" else prob*(1-lp*4),1)
        risk=float(np.clip(risk,0,100))
        smn,smx=meta["severity_range"]; sv=round(smn+(prob/100)*(smx-smn),1)
        svl="🔴 Severe" if sv>70 else ("🟠 High" if sv>50 else ("🟡 Moderate" if sv>30 else ("🟢 Low" if sv>10 else "✅ None")))
        rows.append({"Disease":cls,"Pathogen":meta["pathogen"],"RiskScore":risk,"Severity":svl,
                     "Color":meta["color"],"Prevention":str(meta["prevention"]),"Description":meta["description"]})
    df=pd.DataFrame(rows).sort_values("RiskScore",ascending=False)
    df.to_csv("data/disease_risk.csv",index=False); return df


def disease_forecast(pathology,climate_csv="data/climate_data.csv"):
    try:
        df=pd.read_csv(climate_csv); df["Date"]=pd.to_datetime(df["Date"])
        recent=df.tail(56)
        avg_t=float(recent["Temperature"].mean()); avg_h=float(recent["Humidity"].mean())
        avg_r=float(recent["Rainfall"].mean())
    except: avg_t,avg_h,avg_r=30.,75.,3.
    base=pathology["DiseaseSeverity"]; rows=[]
    for day in range(1,8):
        t=avg_t+np.random.normal(0,.8); h=avg_h+np.random.normal(0,2.5)
        r=max(0,avg_r+np.random.normal(0,1.5))
        cf=1.08 if (h>85 and 20<=t<=32) else (0.96 if h<60 or t>38 else 1.02)
        nt=round(min(base*(cf**day),100),1); wt=round(max(nt*(0.68**(day/3.5)),1.5),1)
        rl="🔴 Critical" if nt>75 else ("🟠 High" if nt>55 else ("🟡 Moderate" if nt>35 else "🟢 Low"))
        rows.append({"Day":day,"Date":(datetime.now()+timedelta(days=day)).strftime("%Y-%m-%d"),
                     "Temperature":round(t,1),"Humidity":round(h,1),"Rainfall":round(r,1),
                     "SeverityNoTreat":nt,"SeverityTreated":wt,"RiskLabel":rl})
    fc=pd.DataFrame(rows); fc.to_csv("data/disease_forecast.csv",index=False); return fc


def analyze_leaf_image(image_path, climate_csv="data/climate_data.csv"):
    print(f"\n🔬 Analyzing: {os.path.basename(image_path)}")
    feats=analyze_pixels(image_path)
    predicted,confidence,probs=classify_disease(feats)
    pathology=compute_pathology(feats,predicted)
    soil_pred=predict_soil_from_image(feats,predicted,pathology)
    meta=DISEASE_META[predicted]

    print(f"   Green:{feats['green_pct']*100:.1f}% Panicle:{feats['panicle_pct']*100:.1f}% Lesion:{feats['lesion_pct']*100:.1f}%")
    print(f"   → {pathology['HealthLabel']} | {predicted} ({confidence:.1f}%)")

    result={
        "PredictedDisease":predicted,"Confidence":confidence,"Probabilities":probs,
        "Method":"HSV+RGB Pixel Analysis (Rice-Aware)",
        "Pathogen":meta["pathogen"],"Description":meta["description"],
        "MetaColor":meta["color"],"Prevention":meta["prevention"],
        "HealthLabel":pathology["HealthLabel"],
        "NDVI_proxy":pathology["NDVI"],"Greenness":round(feats["green_pct"]*100,1),
        "Yellowness":round(feats["blb_yellow"]*100,1),"Brownness":round(feats["brown_pct"]*100,1),
        "PaniclePct":pathology["PaniclePct"],"SoilPrediction":soil_pred,
        **pathology,
    }

    build_risk_df(probs,feats,pathology)

    pd.DataFrame([{"Timestamp":datetime.now().strftime("%Y-%m-%d %H:%M"),
                   "LeafColorIndex":pathology["LeafColorIndex"],"LeafColorStatus":pathology["LeafColorStatus"],
                   "LeafColorHex":pathology["LeafColorHex"],"LeafSpotScore":pathology["LeafSpotScore"],
                   "LeafSpotLevel":pathology["LeafSpotLevel"],"LeafSpotType":f"HSV:{predicted}",
                   "ChlorophyllSPAD":pathology["ChlorophyllSPAD"],"ChlorophyllStatus":pathology["ChlorophyllStatus"],
                   "DiseaseProbability":pathology["DiseaseProbability"],"DiseaseLevel":pathology["DiseaseLevel"],
                   "DiseaseColor":pathology["DiseaseColor"],"DiseaseAction":meta["prevention"][0],
                   "FungalInfectionRate":pathology["FungalInfectionRate"],
                   "SporeViability":pathology["SporeViability"],"SpreadRisk":pathology["SpreadRisk"]}
                  ]).to_csv("data/plant_pathology.csv",index=False)

    row={k:v for k,v in result.items() if not isinstance(v,(dict,list))}
    row["Probabilities"]=str(probs); row["Prevention"]=" | ".join(meta["prevention"])
    pd.DataFrame([row]).to_csv("data/image_analysis.csv",index=False)

    fc_df=disease_forecast(pathology,climate_csv)
    return result,fc_df


if __name__ == "__main__":
    print("✅ Rice-aware image disease + soil prediction ready")

