"""
experiment_phase3.py — Experimentos aislados (NO tocan producción):
  A. Feature pruning: ¿un subconjunto top-K por importancia iguala/mejora al set
     completo de 166 features? (walk-forward 2024, LightGBM)
  B. DNN diverso: ¿un MLP (sklearn) aporta diversidad real? Se evalúa MLP solo,
     y el promedio LightGBM⊕MLP, con test de Diebold-Mariano vs LightGBM.

Decisión por evidencia: aplicar en producción solo si mejora de forma robusta.
"""
import warnings, sys, json
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from scripts.experiment_vst import load, PARAMS, metrics

def dm_test(ea, eb, loss="mae"):
    from scipy.stats import norm
    a=np.abs(ea) if loss=="mae" else ea**2; b=np.abs(eb) if loss=="mae" else eb**2
    d=(a-b); d=d[~np.isnan(d)]; n=len(d); v=np.var(d,ddof=0)
    if n<30 or v<=0: return np.nan,np.nan
    dm=np.mean(d)/np.sqrt(v/n); return float(dm), float(2*(1-norm.cdf(abs(dm))))

def fit_lgbm(Xtr,ytr,Xva,yva,obj="regression",n=2000):
    p=dict(PARAMS,objective=obj,metric="mae",n_estimators=n)
    m=lgb.LGBMRegressor(**p); m.fit(Xtr,ytr,eval_set=[(Xva,yva)],callbacks=[lgb.early_stopping(50,verbose=False)]); return m

# ── A. Feature pruning (walk-forward) ──
def pruning(zone="DK1"):
    X,y,tr,va,te=load(zone)
    base=fit_lgbm(X[tr],y[tr],X[va],y[va])
    imp=pd.Series(base.feature_importances_,index=X.columns).sort_values(ascending=False)
    print(f"\n=== A. PRUNING {zone} ===")
    months=pd.date_range(pd.Timestamp("2024-01-01",tz="UTC"),end=X.index.max(),freq="MS")
    res={}
    for K in [40,70,100,140,len(X.columns)]:
        feats=list(imp.head(K).index); maes=[]
        for ms in months:
            me=(ms+pd.offsets.MonthEnd(1)).replace(hour=23)
            m_tr=X.index<ms; m_pr=(X.index>=ms)&(X.index<=me)
            if m_tr.sum()<3000 or m_pr.sum()==0: continue
            mm=lgb.LGBMRegressor(**dict(PARAMS,objective="regression",metric="mae",n_estimators=500))
            mm.fit(X[m_tr][feats],y[m_tr])
            maes.append(np.mean(np.abs(y[m_pr].values-mm.predict(X[m_pr][feats]))))
        res[K]=float(np.mean(maes))
        print(f"  top-{K:>3} features: walk-forward MAE = {res[K]:.3f}")
    full=res[len(X.columns)]; best=min(res,key=res.get)
    print(f"  → mejor K={best} (MAE {res[best]:.3f}) vs full {full:.3f} | "
          f"{'PRUNING AYUDA' if res[best]<full-0.05 else 'full ~ mejor (no podar)'}")
    return res

# ── B. DNN diverso (single split + DM) ──
def dnn(zone="DK1"):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    X,y,tr,va,te=load(zone)
    print(f"\n=== B. DNN diverso {zone} ===")
    sc=StandardScaler().fit(X[tr])
    Xtr,Xva,Xte=sc.transform(X[tr]),sc.transform(X[va]),sc.transform(X[te])
    ytr=y[tr].values; yte=y[te].values
    lgbm=fit_lgbm(X[tr],y[tr],X[va],y[va]); p_lgbm=lgbm.predict(X[te])
    mlp=MLPRegressor(hidden_layer_sizes=(128,64),activation="relu",alpha=1e-3,
                     batch_size=256,learning_rate_init=1e-3,max_iter=300,
                     early_stopping=True,n_iter_no_change=15,random_state=42)
    mlp.fit(Xtr,ytr); p_mlp=mlp.predict(Xte)
    p_avg=0.5*p_lgbm+0.5*p_mlp
    for name,p in [("LightGBM",p_lgbm),("MLP",p_mlp),("LGBM+MLP avg",p_avg)]:
        m=metrics(yte,p); print(f"  {name:14s} MAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f} DirAcc={m['DirAcc']:.1f}%")
    corr=np.corrcoef(p_lgbm,p_mlp)[0,1]
    dm,pv=dm_test(yte-p_avg, yte-p_lgbm)
    print(f"  corr(LGBM,MLP)={corr:.3f} | DM(avg vs LGBM)={dm:+.2f} p={pv:.3f} "
          f"→ {'avg mejor SIGNIF' if (dm<0 and pv<0.05) else 'LGBM solo >= avg'}")
    return dict(corr=float(corr),dm=dm,p=pv)

if __name__=="__main__":
    out={}
    for z in ["DK1","DK2"]:
        out[z]={"pruning":pruning(z),"dnn":dnn(z)}
    Path(ROOT/"data"/"metadata"/"experiment_phase3.json").write_text(json.dumps(out,indent=2,default=float),encoding="utf-8")
    print("\nGuardado experiment_phase3.json")
