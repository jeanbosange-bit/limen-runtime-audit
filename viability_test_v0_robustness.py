import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

rng=np.random.default_rng(20261007)
OUT=Path("/home/jeanbatuli/limen-runtime-audit/reports/viability-test-v0")
d18=pd.read_csv("/home/jeanbatuli/sra_v14_18_output_dynamic_linkage/output_dynamic_results.csv")
d14r=pd.read_csv("/home/jeanbatuli/sra_v14_14_temporal_control_outputs/temporal_token_results.csv")
d14c=pd.read_csv("/home/jeanbatuli/sra_v14_14_temporal_control_outputs/temporal_token_curves.csv")
dyn=["ratio_norm","dynamic_mean","dynamic_max","transition_entropy","markov_stability","oscillation_rate","recovery_rate","top5_prob_mean","logit_entropy_mean"]

def auc_oof(X,y,g):
    cv=GroupKFold(n_splits=min(6,len(np.unique(g))))
    p=np.full(len(y),np.nan)
    for tr,te in cv.split(X,y,g):
        if len(np.unique(y[tr]))<2: continue
        m=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("lr",LogisticRegression(max_iter=500,class_weight="balanced"))])
        m.fit(X.iloc[tr],y[tr]); p[te]=m.predict_proba(X.iloc[te])[:,1]
    ok=np.isfinite(p)
    if ok.sum()==0 or len(np.unique(y[ok]))<2:return np.nan
    return roc_auc_score(y[ok],p[ok])

def perm_test(df,features,target,group,nperm=10):
    X=df[features].replace([np.inf,-np.inf],np.nan); y=df[target].astype(int).to_numpy(); g=df[group].astype(str).to_numpy()
    obs=auc_oof(X,y,g); vals=[]
    for _ in range(nperm):
        vals.append(auc_oof(X,rng.permutation(y),g))
    vals=np.asarray(vals,float)
    return {"observed":float(obs),"permutation_mean":float(np.nanmean(vals)),"p_one_sided":float((1+np.sum(vals>=obs))/(1+np.sum(np.isfinite(vals)))),"nperm":int(np.sum(np.isfinite(vals)))}
d18["high_repetition"]=(d18["repetition_rate"]>=d18["repetition_rate"].quantile(.75)).astype(int)
rob={}
rob["A_high_repetition_full"]=perm_test(d18,dyn,"high_repetition","prompt")
rob["A_high_repetition_entropy"]=perm_test(d18,["logit_entropy_mean"],"high_repetition","prompt")
keys=["model","label","condition","prompt"]
def prefix(g,frac):
    g=g.sort_values("step")
    n=max(2,int(np.ceil(len(g)*frac)))
    x=g.iloc[:n]
    ent=x["entropy"].to_numpy(float)
    kap=x["kappa"].to_numpy(float)
    rat=x["ratio_partial"].to_numpy(float)
    def sl(a):
        return float(np.polyfit(np.arange(len(a)),a,1)[0]) if len(a)>1 else 0.0
    de=np.diff(ent)
    osc=float(np.mean(np.sign(de[1:])*np.sign(de[:-1])<0)) if len(de)>1 else 0.0
    return pd.Series({"entropy_mean_prefix":float(ent.mean()),"entropy_std_prefix":float(ent.std()),"entropy_slope_prefix":sl(ent),"positive_entropy_variation":float(np.maximum(de,0).sum()) if len(de) else 0.0,"kappa_mean_prefix":float(kap.mean()),"kappa_std_prefix":float(kap.std()),"kappa_slope_prefix":sl(kap),"top5_mean_prefix":float(x["top5"].mean()),"margin_mean_prefix":float(x["margin"].mean()),"ratio_last_prefix":float(rat[-1]),"ratio_slope_prefix":sl(rat),"oscillation_prefix":osc})
cols=["entropy_mean_prefix","entropy_std_prefix","entropy_slope_prefix","positive_entropy_variation","kappa_mean_prefix","kappa_std_prefix","kappa_slope_prefix","top5_mean_prefix","margin_mean_prefix","ratio_last_prefix","ratio_slope_prefix","oscillation_prefix"]
for frac in [.25,.5,.75]:
    pref=d14c.groupby(keys,dropna=False).apply(lambda g:prefix(g,frac),include_groups=False).reset_index()
    m=pref.merge(d14r,on=keys,validate="one_to_one")
    m["high_final_ratio"]=(m["ratio_final"]>=m["ratio_final"].median()).astype(int)
    rob["B_ratio_"+str(frac)+"_full"]=perm_test(m,cols,"high_final_ratio","prompt")
    rob["B_ratio_"+str(frac)+"_entropy"]=perm_test(m,["entropy_mean_prefix"],"high_final_ratio","prompt")
(OUT/"robustness.json").write_text(json.dumps(rob,indent=2))
print(json.dumps(rob,indent=2))
