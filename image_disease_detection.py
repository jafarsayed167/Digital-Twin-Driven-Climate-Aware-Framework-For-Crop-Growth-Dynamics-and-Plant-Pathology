"""
IMAGE DISEASE DETECTION + SOIL PREDICTION
==========================================
Rice-aware: Panicle (grain head) pixels excluded from disease analysis.
After disease detect -> soil N,P,K,pH,Moisture predicted from leaf indicators.
"""
import numpy as np, pandas as pd, os
from datetime import datetime, timedelta

os.makedirs("data/models", exist_ok=True)

CLASSES = ["Healthy","Rice Blast","Brown Spot","Sheath Blight","Bacterial Leaf Blight"]

DISEASE_META = {
    "Healthy":{"pathogen":"None","description":"Leaf tissue healthy. Green, no lesions.",
        "severity_range":(0,10),"yield_loss":(0,0),"color":"#16a34a","health_label":"Healthy",
        "soil_stress":{"N":"Normal","P":"Normal","K":"Normal","pH":"Normal","Moisture":"Normal"},
        "prevention":["Monitor every 3-4 days","Maintain balanced NPK","Ensure drainage","Preventive neem spray 5mL/L"]},
    "Rice Blast":{"pathogen":"Magnaporthe oryzae (Fungal)","description":"Diamond-shaped gray/white lesions. Most destructive.",
        "severity_range":(45,95),"yield_loss":(15,70),"color":"#dc2626","health_label":"Unhealthy - Rice Blast",
        "soil_stress":{"N":"Excess (promotes blast)","P":"Low","K":"Low","pH":"Acidic","Moisture":"Waterlogged"},
        "prevention":["Tricyclazole 75 WP @ 0.6g/L immediately","Propiconazole 25 EC @ 1mL/L curative",
                      "Use blast-resistant varieties IR64/Swarna/MTU1010","Split urea doses","Drain field"]},
    "Brown Spot":{"pathogen":"Bipolaris oryzae (Fungal)","description":"Oval brown lesions with yellow halo. K deficiency linked.",
        "severity_range":(25,70),"yield_loss":(5,45),"color":"#ea580c","health_label":"Unhealthy - Brown Spot",
        "soil_stress":{"N":"Low-Moderate","P":"Low","K":"Deficient","pH":"Slightly acidic","Moisture":"Variable"},
        "prevention":["Mancozeb 75 WP @ 2g/L","Propiconazole 25 EC @ 1mL/L","Potassium MOP @ 25 kg/acre","Thiram seed treatment","pH 6.0-7.0"]},
    "Sheath Blight":{"pathogen":"Rhizoctonia solani (Fungal)","description":"Pale water-soaked lesions on sheath. Dense canopy.",
        "severity_range":(30,80),"yield_loss":(10,50),"color":"#ca8a04","health_label":"Unhealthy - Sheath Blight",
        "soil_stress":{"N":"Excess","P":"Normal","K":"Low","pH":"Near neutral","Moisture":"Waterlogged"},
        "prevention":["Hexaconazole 5 EC @ 2mL/L","Validamycin 3 SL @ 2.5mL/L","Row spacing 20cm","Drain field","Reduce N"]},
    "Bacterial Leaf Blight":{"pathogen":"Xanthomonas oryzae pv. oryzae","description":"Yellow-white wilting from leaf margins.",
        "severity_range":(20,75),"yield_loss":(6,40),"color":"#b91c1c","health_label":"Unhealthy - Bacterial Leaf Blight",
        "soil_stress":{"N":"Variable","P":"Low","K":"Deficient","pH":"Alkaline risk","Moisture":"Flood-irrigated"},
        "prevention":["Copper Oxychloride 50 WP @ 3g/L","Streptomycin + Copper 0.2%","Resistant varieties","Avoid flood after rain","Remove infected debris"]},
}

def compute_hsv(rgb):
    maxc=np.max(rgb,axis=2); minc=np.min(rgb,axis=2); V=maxc
    S=np.where(maxc>1e-6,(maxc-minc)/maxc,0.0)
    delta=maxc-minc+1e-8; r,g,b=rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
    H=np.where(maxc==r,((g-b)/delta)%6,np.where(maxc==g,(b-r)/delta+2,(r-g)/delta+4))/6.0
    return H%1.0, S, V

def analyze_pixels(image_path):
    from PIL import Image
    img = Image.open(image_path).convert("RGB").resize((256,256))
    rgb = np.array(img, dtype=np.float32)/255.0
    r,g,b = rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]; total=256*256
    H,S,V = compute_hsv(rgb)

    leaf_green   = (H>=0.20)&(H<=0.45)&(S>0.20)&(V>0.18)
    panicle_mask = (H>=0.10)&(H<=0.22)&(S>0.15)&(S<0.60)&(V>0.45)
    blb_yellow   = (H>=0.05)&(H<=0.18)&(S>0.50)&(V<0.80)
    brown_lesion = ((H<=0.07)|(H>=0.93))&(S>0.30)&(V>0.15)&(V<0.70)
    blast_gray   = (S<0.18)&(V>0.50)&~leaf_green&~panicle_mask
    necrosis     = (V<0.15)
    sheath_pale  = (S<0.20)&(V>0.65)&~leaf_green&~blast_gray

    gp=float(leaf_green.sum()/total); pp=float(panicle_mask.sum()/total)
    byp=float(blb_yellow.sum()/total); brp=float(brown_lesion.sum()/total)
    grp=float(blast_gray.sum()/total); nec=float(necrosis.sum()/total)
    spp=float(sheath_pale.sum()/total)
    lp=float(np.clip(byp+brp+grp+nec,0,1))
    mean_r=float(r.mean()); mean_g=float(g.mean()); mean_b=float(b.mean())
    exg=float(2*mean_g-mean_r-mean_b)
    ndvi=float(np.clip((mean_g-mean_r)/(mean_g+mean_r+1e-6), -1.0, 1.0))
    texture=float((r.std()+g.std()+b.std())/3)

    return {"green_pct":round(gp,4),"panicle_pct":round(pp,4),"blb_yellow":round(byp,4),
            "brown_pct":round(brp,4),"gray_pct":round(grp,4),"necrosis_pct":round(nec,4),
            "sheath_pct":round(spp,4),"lesion_pct":round(lp,4),
            "exg":round(exg,4),"ndvi_proxy":round(ndvi,4),"texture":round(texture,4),
            "mean_r":round(mean_r,4),"mean_g":round(mean_g,4),"mean_b":round(mean_b,4)}

def classify_disease(feats):
    gp=feats["green_pct"]; byp=feats["blb_yellow"]; brp=feats["brown_pct"]
    grp=feats["gray_pct"]; nec=feats["necrosis_pct"]; spp=feats["sheath_pct"]
    lp=feats["lesion_pct"]; txt=feats["texture"]; pp=feats["panicle_pct"]
    sc={}

    if gp>0.40 and lp<0.05:   h=90+gp*10-lp*100
    elif gp>0.25 and lp<0.10: h=72+gp*15-lp*80
    elif gp>0.15 and lp<0.18: h=50+gp*12-lp*60
    else:                      h=max(5,30+gp*8-lp*50)
    if pp>0.05: h=min(h+pp*15, 95)
    sc["Healthy"]=float(np.clip(h,0,95))

    blast=grp*220+nec*90+txt*70-gp*20
    if grp>0.03 and nec>0.01 and txt>0.06: blast+=30
    if grp>0.06: blast+=20
    sc["Rice Blast"]=float(np.clip(blast,0,95))

    brown=brp*230+nec*50-gp*10
    if brp>0.04 and gp>0.15: brown+=25
    if brp>0.08: brown+=20
    sc["Brown Spot"]=float(np.clip(brown,0,95))

    sheath=spp*180+grp*50+txt*50-gp*15
    if spp>0.04 and txt>0.06: sheath+=20
    sc["Sheath Blight"]=float(np.clip(sheath,0,95))

    blb=byp*250+spp*40-gp*20
    if byp>0.06: blb+=30
    if byp>0.12: blb+=20
    sc["Bacterial Leaf Blight"]=float(np.clip(blb,0,95))

    total_s=sum(sc.values()) or 1
    probs={k:round(v/total_s*100,1) for k,v in sc.items()}
    pred=max(probs,key=probs.get)
    return pred, probs[pred], probs

def compute_pathology(feats, predicted):
    lp=feats["lesion_pct"]; gp=feats["green_pct"]
    exg=feats["exg"]; brp=feats["brown_pct"]; nec=feats["necrosis_pct"]; pp=feats["panicle_pct"]

    lci=round(float(np.clip(gp*160,0,100)),1)
    if lci>=70:   lci_s,lci_h="Healthy Green","#16a34a"
    elif lci>=50: lci_s,lci_h="Slightly Pale","#ca8a04"
    elif lci>=30: lci_s,lci_h="Moderate Damage","#ea580c"
    else:         lci_s,lci_h="Severe Damage","#dc2626"

    spot=round(float(np.clip(lp*300,0,100)),1)
    spot_l=("None" if spot<8 else "Low" if spot<25 else "Moderate" if spot<50 else "High" if spot<75 else "Severe")

    spad=round(float(np.clip(exg*90+gp*35+12,5,80)),1)
    ci_s=("Excellent" if spad>=45 else "Normal" if spad>=35 else "Moderate Deficiency" if spad>=25 else "Low" if spad>=15 else "Critical")

    # NDVI: strictly -1.0 to +1.0 range
    # Formula: (Green - Red) / (Green + Red)
    # +1 = pure vegetation, 0 = bare soil, -1 = water/dead
    ndvi = round(float(np.clip(feats["ndvi_proxy"], -1.0, 1.0)), 3)
    meta=DISEASE_META[predicted]; smn,smx=meta["severity_range"]
    sev=(round(float(np.clip(lp*120,0,12)),1) if predicted=="Healthy"
         else round(float(np.clip(smn+lp*(smx-smn)*6,smn,smx)),1))

    dp=round(float(np.clip(sev*0.65+spot*0.35*(1-gp*0.5),0,100)),1)
    if predicted=="Healthy": dp=min(dp,12)
    dp_l,dp_c=("Very Low","#16a34a") if dp<15 else ("Low","#65a30d") if dp<35 else ("Moderate","#ca8a04") if dp<55 else ("High","#ea580c") if dp<75 else ("Critical","#dc2626")

    fir=round(float(np.clip((brp+nec)*350,0,100)),1)
    spread=("Negligible" if fir<10 else "Low" if fir<30 else "Moderate" if fir<55 else "Rapid" if fir<75 else "Epidemic")
    yln,ylx=meta["yield_loss"]
    yl=round(float(np.clip(yln+(sev/100)*(ylx-yln),0,100)),1)

    return {"HealthLabel":meta["health_label"],"LeafColorIndex":lci,"LeafColorStatus":lci_s,
            "LeafColorHex":lci_h,"LeafSpotScore":spot,"LeafSpotLevel":spot_l,
            "ChlorophyllSPAD":spad,"ChlorophyllStatus":ci_s,"NDVI":ndvi,
            "DiseaseSeverity":sev,"DiseaseProbability":dp,"DiseaseLevel":dp_l,"DiseaseColor":dp_c,
            "FungalInfectionRate":fir,"SpreadRisk":spread,"YieldLossPct":yl,"PaniclePct":round(pp*100,1)}

def predict_soil_from_image(feats, predicted, pathology):
    gp=feats["green_pct"]; lp=feats["lesion_pct"]
    byp=feats["blb_yellow"]; brp=feats["brown_pct"]
    spad=pathology["ChlorophyllSPAD"]; ndvi=pathology["NDVI"]

    if spad>=45:   N_est,N_s=round(np.random.uniform(90,120),1),"Sufficient (90-120 kg/ha)"
    elif spad>=35: N_est,N_s=round(np.random.uniform(70,90),1),"Moderate (70-90 kg/ha)"
    elif spad>=25: N_est,N_s=round(np.random.uniform(45,70),1),"Low (45-70 kg/ha) - Apply urea"
    else:          N_est,N_s=round(np.random.uniform(20,45),1),"Deficient - Urgent N needed"

    if predicted=="Brown Spot" or brp>0.05: P_est,P_s=round(np.random.uniform(20,40),1),"Low - Apply DAP"
    elif gp>0.45:                           P_est,P_s=round(np.random.uniform(45,65),1),"Good (45-65 kg/ha)"
    else:                                   P_est,P_s=round(np.random.uniform(35,50),1),"Moderate"

    if predicted in ["Bacterial Leaf Blight","Sheath Blight","Brown Spot"] or byp>0.06:
        K_est,K_s=round(np.random.uniform(25,50),1),"Deficient - Apply MOP"
    elif gp>0.45: K_est,K_s=round(np.random.uniform(60,85),1),"Good (60-85 kg/ha)"
    else:         K_est,K_s=round(np.random.uniform(45,65),1),"Moderate"

    if predicted=="Rice Blast":             pH_est,pH_s=round(np.random.uniform(4.8,5.8),1),"Acidic - Add lime"
    elif predicted=="Bacterial Leaf Blight":pH_est,pH_s=round(np.random.uniform(7.0,7.8),1),"Slightly alkaline"
    elif gp>0.45:                           pH_est,pH_s=round(np.random.uniform(6.0,7.0),1),"Optimal"
    else:                                   pH_est,pH_s=round(np.random.uniform(5.5,6.5),1),"Moderate"

    if predicted in ["Rice Blast","Sheath Blight"]: M_est,M_s=round(np.random.uniform(80,100),1),"High/Waterlogged - Drain"
    elif predicted=="Brown Spot":                   M_est,M_s=round(np.random.uniform(40,60),1),"Variable - Monitor"
    elif lp<0.05 and gp>0.40:                      M_est,M_s=round(np.random.uniform(70,90),1),"Good"
    else:                                           M_est,M_s=round(np.random.uniform(30,55),1),"Low - Irrigate"

    meta=DISEASE_META[predicted]
    return {"N_est":N_est,"N_status":N_s,"P_est":P_est,"P_status":P_s,
            "K_est":K_est,"K_status":K_s,"pH_est":pH_est,"pH_status":pH_s,
            "M_est":M_est,"M_status":M_s,"disease_soil_link":meta["soil_stress"],
            "basis":f"From: SPAD={spad}, NDVI={ndvi} (scale -1 to +1), Disease={predicted}"}

def build_risk_df(probs, feats, pathology):
    lp=feats["lesion_pct"]; rows=[]
    for cls in CLASSES:
        meta=DISEASE_META[cls]; prob=probs.get(cls,0)
        risk=round(prob*(1+lp*3) if cls!="Healthy" else prob*(1-lp*4),1)
        risk=float(np.clip(risk,0,100))
        smn,smx=meta["severity_range"]; sv=round(smn+(prob/100)*(smx-smn),1)
        svl=("Severe" if sv>70 else "High" if sv>50 else "Moderate" if sv>30 else "Low" if sv>10 else "None")
        rows.append({"Disease":cls,"Pathogen":meta["pathogen"],"RiskScore":risk,"Severity":svl,
                     "Color":meta["color"],"Prevention":str(meta["prevention"]),"Description":meta["description"]})
    df=pd.DataFrame(rows).sort_values("RiskScore",ascending=False)
    df.to_csv("data/disease_risk.csv",index=False); return df

def disease_forecast(pathology, climate_csv="data/climate_data.csv"):
    try:
        df=pd.read_csv(climate_csv); df["Date"]=pd.to_datetime(df["Date"])
        avg_t=float(df.tail(56)["Temperature"].mean())
        avg_h=float(df.tail(56)["Humidity"].mean())
    except: avg_t,avg_h=30.,75.
    base=pathology["DiseaseSeverity"]; rows=[]
    for day in range(1,8):
        t=avg_t+np.random.normal(0,.8); h=avg_h+np.random.normal(0,2.5)
        cf=1.08 if (h>85 and 20<=t<=32) else (0.96 if h<60 or t>38 else 1.02)
        nt=round(min(base*(cf**day),100),1); wt=round(max(nt*(0.68**(day/3.5)),1.5),1)
        rows.append({"Day":day,"Date":(datetime.now()+timedelta(days=day)).strftime("%Y-%m-%d"),
                     "Temperature":round(t,1),"Humidity":round(h,1),"Rainfall":round(max(0,np.random.normal(2,2)),1),
                     "SeverityNoTreat":nt,"SeverityTreated":wt})
    fc=pd.DataFrame(rows); fc.to_csv("data/disease_forecast.csv",index=False); return fc

def analyze_leaf_image(image_path, climate_csv="data/climate_data.csv"):
    feats=analyze_pixels(image_path)
    predicted,confidence,probs=classify_disease(feats)
    pathology=compute_pathology(feats,predicted)
    soil_pred=predict_soil_from_image(feats,predicted,pathology)
    meta=DISEASE_META[predicted]

    result={
        "PredictedDisease":predicted,"Confidence":confidence,"Probabilities":probs,
        "Method":"HSV+RGB Pixel (Rice-Aware)","Pathogen":meta["pathogen"],
        "Description":meta["description"],"MetaColor":meta["color"],
        "Prevention":meta["prevention"],"HealthLabel":pathology["HealthLabel"],
        "NDVI_proxy":pathology["NDVI"],"Greenness":round(feats["green_pct"]*100,1),
        "Yellowness":round(feats["blb_yellow"]*100,1),"Brownness":round(feats["brown_pct"]*100,1),
        "PaniclePct":pathology["PaniclePct"],"SoilPrediction":soil_pred,**pathology,
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
        "SporeViability":round(feats["lesion_pct"]*500,1),"SpreadRisk":pathology["SpreadRisk"],
    }]).to_csv("data/plant_pathology.csv",index=False)

    row={k:v for k,v in result.items() if not isinstance(v,(dict,list))}
    row["Probabilities"]=str(probs); row["Prevention"]=" | ".join(meta["prevention"])
    pd.DataFrame([row]).to_csv("data/image_analysis.csv",index=False)

    fc_df=disease_forecast(pathology,climate_csv)
    print(f"Image: {pathology['HealthLabel']} | {predicted} ({confidence:.1f}%)")
    return result, fc_df
