"""
experiment_vst.py — Experimento aislado (NO toca el pipeline de producción).

Compara, en nuestro propio dataset/split temporal, varias configuraciones de
objetivo + transformación del target para LightGBM, para decidir con datos
(no por fe) la opción más robusta que reduzca MAE — especialmente en spikes.

Configs comparadas:
  A. baseline   : objective=regression (L2), target = precio
  B. L1         : objective=mae         (L1), target = precio
  C. asinh+L2   : objective=regression (L2), target = asinh((p-med)/mad)
  D. asinh+L1   : objective=mae         (L1), target = asinh((p-med)/mad)
  E. asinh+L1 por-hora (24 modelos)
  F. asinh+L1 ensemble de ventanas de calibración (365d, 730d, full) promediado

asinh es monótona → invertir con sinh preserva la mediana (correcto para MAE).
median/MAD se calculan SOLO con el train (sin leakage).
"""
import sys, warnings, json
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FEATURES_PATH = ROOT / "data" / "master" / "dataset_features.parquet"
TRAIN_END = "2024-06-30 23:00"; VAL_START="2024-07-01"; VAL_END="2024-09-30 23:00"; TEST_START="2024-10-01"
SAFE_IS = ("is_weekend", "is_holiday")

PARAMS = dict(n_estimators=2000, learning_rate=0.03, num_leaves=95, max_depth=9,
              min_child_samples=40, subsample=0.8, colsample_bytree=0.75,
              reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1)

def load(zone="DK1"):
    df = pd.read_parquet(FEATURES_PATH).sort_index()
    tgt = f"price_{zone}"
    fcols = [c for c in df.columns if c.startswith("feat_")] + \
            [c for c in df.columns if c.startswith("is_") and c.startswith(SAFE_IS)]
    d = df[[tgt]+fcols].dropna(subset=[tgt])
    d = d.dropna(thresh=int(len(fcols)*0.5)+1)
    for c in fcols:
        if d[c].isna().any(): d[c] = d[c].fillna(d[c].median())
    X, y = d[fcols], d[tgt]
    tr = X.index <= pd.Timestamp(TRAIN_END, tz="UTC")
    va = (X.index>=pd.Timestamp(VAL_START,tz="UTC")) & (X.index<=pd.Timestamp(VAL_END,tz="UTC"))
    te = X.index >= pd.Timestamp(TEST_START, tz="UTC")
    return X, y, tr, va, te

def metrics(y_true, y_pred):
    e = y_true - y_pred
    mae = np.mean(np.abs(e)); rmse = np.sqrt(np.mean(e**2))
    dir_acc = np.mean(np.sign(np.diff(y_true))==np.sign(np.diff(y_pred)))*100
    spike = y_true > 200
    mae_spike = np.mean(np.abs(e[spike])) if spike.any() else np.nan
    mae_norm = np.mean(np.abs(e[(y_true>=20)&(y_true<=100)]))
    return dict(MAE=mae, RMSE=rmse, DirAcc=dir_acc, MAE_spike=mae_spike,
                MAE_normal=mae_norm, n_spike=int(spike.sum()))

def fit_lgbm(Xtr,ytr,Xva,yva,objective):
    p = dict(PARAMS, objective=objective, metric="mae" if objective!="quantile" else "quantile")
    m = lgb.LGBMRegressor(**p)
    m.fit(Xtr,ytr,eval_set=[(Xva,yva)],callbacks=[lgb.early_stopping(50,verbose=False)])
    return m

def run(zone="DK1"):
    X,y,tr,va,te = load(zone)
    Xtr,ytr=X[tr],y[tr]; Xva,yva=X[va],y[va]; Xte,yte=X[te],y[te]
    med=float(np.median(ytr)); mad=float(np.median(np.abs(ytr-med)))*1.4826
    asinh=lambda p:np.arcsinh((p-med)/mad); sinh=lambda z:med+mad*np.sinh(z)
    print(f"\n==== {zone} | train={tr.sum()} val={va.sum()} test={te.sum()} | med={med:.1f} mad={mad:.1f} ====")
    res={}
    # A baseline L2
    m=fit_lgbm(Xtr,ytr,Xva,yva,"regression"); res["A_baseline_L2"]=metrics(yte.values,m.predict(Xte))
    # B L1
    m=fit_lgbm(Xtr,ytr,Xva,yva,"mae"); res["B_L1"]=metrics(yte.values,m.predict(Xte))
    # C asinh+L2
    m=fit_lgbm(Xtr,asinh(ytr.values),Xva,asinh(yva.values),"regression"); res["C_asinh_L2"]=metrics(yte.values,sinh(m.predict(Xte)))
    # D asinh+L1
    m=fit_lgbm(Xtr,asinh(ytr.values),Xva,asinh(yva.values),"mae"); res["D_asinh_L1"]=metrics(yte.values,sinh(m.predict(Xte)))
    # E asinh+L1 per-hour (24 modelos)
    hours = X.index.tz_convert("Europe/Copenhagen").hour
    pred_e=np.full(te.sum(),np.nan); hte=hours[te]
    for h in range(24):
        mtr=tr&(hours==h); mva=va&(hours==h); mte=te&(hours==h)
        if mtr.sum()<200 or mte.sum()==0: continue
        mm=fit_lgbm(X[mtr],asinh(y[mtr].values),X[mva],asinh(y[mva].values),"mae")
        pred_e[hte==h]=sinh(mm.predict(X[mte]))
    res["E_asinh_L1_perhour"]=metrics(yte.values,pred_e)
    # F asinh+L1 ensemble de ventanas (full, 730d, 365d) promediado
    preds=[]
    for days in [None,730,365]:
        if days is None: mtr=tr
        else:
            cut=pd.Timestamp(TRAIN_END,tz="UTC")-pd.Timedelta(days=days); mtr=tr&(X.index>=cut)
        mm=fit_lgbm(X[mtr],asinh(y[mtr].values),Xva,asinh(yva.values),"mae")
        preds.append(sinh(mm.predict(Xte)))
    res["F_asinh_L1_winens"]=metrics(yte.values,np.mean(preds,axis=0))
    return res

if __name__=="__main__":
    allres={}
    for z in ["DK1","DK2"]:
        allres[z]=run(z)
        print(f"\n{'config':22s} {'MAE':>7} {'RMSE':>7} {'DirAcc':>7} {'MAE_spike':>10} {'MAE_norm':>9}")
        base=allres[z]["A_baseline_L2"]["MAE"]
        for k,v in allres[z].items():
            imp=(base-v['MAE'])/base*100
            print(f"{k:22s} {v['MAE']:7.2f} {v['RMSE']:7.2f} {v['DirAcc']:7.1f} {v['MAE_spike']:10.1f} {v['MAE_normal']:9.2f}  ({imp:+.1f}% vs A)")
    Path(ROOT/"data"/"metadata"/"experiment_vst.json").write_text(json.dumps(allres,indent=2,default=float),encoding="utf-8")
    print("\nGuardado: data/metadata/experiment_vst.json")
