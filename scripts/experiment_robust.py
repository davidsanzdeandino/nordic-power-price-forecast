"""
experiment_robust.py — Ronda 2 (aislado). Tras descartar VST/asinh para GBM,
prueba técnicas robustas pensadas para gradient boosting:
  A. baseline (L2)                 — referencia (config actual)
  G. quantile-0.5 (mediana pura)
  H. huber (regresión robusta)
  I. winsorized-L2 (cap target train a [p1,p99])
  J. ensemble A+B  (L2 ⊕ L1, media)
  K. ensemble A+G+H (L2 ⊕ mediana ⊕ huber, media — diversidad de objetivos)
Métrica clave: MAE global y MAE en spikes (>200) y en normal (20-100).
"""
import sys, warnings, json
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from scripts.experiment_vst import load, metrics, PARAMS

def fit(Xtr,ytr,Xva,yva,objective,**extra):
    p=dict(PARAMS,objective=objective,**extra)
    if objective=="quantile": p["alpha"]=0.5
    p["metric"]="mae"
    m=lgb.LGBMRegressor(**p); m.fit(Xtr,ytr,eval_set=[(Xva,yva)],callbacks=[lgb.early_stopping(50,verbose=False)])
    return m

def run(zone):
    X,y,tr,va,te=load(zone); Xtr,ytr=X[tr],y[tr]; Xva,yva=X[va],y[va]; Xte,yte=X[te],y[te]
    print(f"\n==== {zone} ====")
    res={}; P={}
    P["A"]=fit(Xtr,ytr,Xva,yva,"regression").predict(Xte); res["A_baseline_L2"]=metrics(yte.values,P["A"])
    P["G"]=fit(Xtr,ytr,Xva,yva,"quantile").predict(Xte); res["G_quantile50"]=metrics(yte.values,P["G"])
    P["H"]=fit(Xtr,ytr,Xva,yva,"huber",alpha=5.0).predict(Xte); res["H_huber"]=metrics(yte.values,P["H"])
    lo,hi=np.percentile(ytr,[1,99]); ytr_w=np.clip(ytr,lo,hi)
    P["I"]=fit(Xtr,ytr_w,Xva,np.clip(yva,lo,hi),"regression").predict(Xte); res["I_winsor_L2"]=metrics(yte.values,P["I"])
    res["J_ens_A_B"]=metrics(yte.values,np.mean([P["A"],fit(Xtr,ytr,Xva,yva,"mae").predict(Xte)],axis=0))
    res["K_ens_A_G_H"]=metrics(yte.values,np.mean([P["A"],P["G"],P["H"]],axis=0))
    return res

if __name__=="__main__":
    allr={}
    for z in ["DK1","DK2"]:
        allr[z]=run(z); base=allr[z]["A_baseline_L2"]["MAE"]
        print(f"\n{'config':16s} {'MAE':>7} {'RMSE':>7} {'DirAcc':>7} {'MAE_spike':>10} {'MAE_norm':>9}")
        for k,v in allr[z].items():
            print(f"{k:16s} {v['MAE']:7.2f} {v['RMSE']:7.2f} {v['DirAcc']:7.1f} {v['MAE_spike']:10.1f} {v['MAE_normal']:9.2f}  ({(base-v['MAE'])/base*100:+.1f}%)")
    Path(ROOT/"data"/"metadata"/"experiment_robust.json").write_text(json.dumps(allr,indent=2,default=float),encoding="utf-8")
    print("\nGuardado.")
