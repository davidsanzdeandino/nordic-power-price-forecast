"""
experiment_walkforward.py — Ronda 3: valida en WALK-FORWARD (no un solo split)
si el ensemble de objetivos diversos (L2+mediana+Huber) es una mejora robusta
sobre el baseline L2, mes a mes de 2024. Evita confundir ruido con señal.
"""
import sys, warnings, json
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from scripts.experiment_vst import load, PARAMS

def fit(Xtr,ytr,objective,**ex):
    p=dict(PARAMS,objective=objective,n_estimators=600,**ex); p["metric"]="mae"
    if objective=="quantile": p["alpha"]=0.5
    m=lgb.LGBMRegressor(**p); m.fit(Xtr,ytr); return m

def mae(a,b): return float(np.mean(np.abs(a-b)))
def diracc(a,b): return float(np.mean(np.sign(np.diff(a))==np.sign(np.diff(b)))*100)

def run(zone):
    X,y,_,_,_=load(zone)
    months=pd.date_range(pd.Timestamp("2024-01-01",tz="UTC"),end=X.index.max(),freq="MS")
    rows=[]
    for ms in months:
        me=ms+pd.offsets.MonthEnd(1); me=me.replace(hour=23)
        tr=X.index<ms; pr=(X.index>=ms)&(X.index<=me)
        if tr.sum()<3000 or pr.sum()==0: continue
        Xtr,ytr=X[tr],y[tr].values; Xpr,ypr=X[pr],y[pr].values
        a=fit(Xtr,ytr,"regression").predict(Xpr)
        g=fit(Xtr,ytr,"quantile").predict(Xpr)
        h=fit(Xtr,ytr,"huber",alpha=5.0).predict(Xpr)
        k=np.mean([a,g,h],axis=0)
        rows.append(dict(month=ms.strftime("%Y-%m"),
                         MAE_A=mae(ypr,a),MAE_K=mae(ypr,k),
                         DA_A=diracc(ypr,a),DA_K=diracc(ypr,k)))
    df=pd.DataFrame(rows)
    print(f"\n==== {zone} (walk-forward 2024) ====")
    print(df.to_string(index=False,float_format=lambda x:f"{x:.2f}"))
    print(f"  MEDIA  MAE: A={df.MAE_A.mean():.2f}  K={df.MAE_K.mean():.2f}  "
          f"({(df.MAE_A.mean()-df.MAE_K.mean())/df.MAE_A.mean()*100:+.1f}%) | "
          f"DirAcc: A={df.DA_A.mean():.1f}  K={df.DA_K.mean():.1f} ({df.DA_K.mean()-df.DA_A.mean():+.1f}pp)")
    print(f"  K mejor MAE en {int((df.MAE_K<df.MAE_A).sum())}/{len(df)} meses | "
          f"K mejor DirAcc en {int((df.DA_K>df.DA_A).sum())}/{len(df)} meses")
    return df

if __name__=="__main__":
    for z in ["DK1","DK2"]:
        run(z)
    print("\nOK")
