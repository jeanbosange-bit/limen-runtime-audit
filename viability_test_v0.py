import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, brier_score_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT=Path("/home/jeanbatuli/limen-runtime-audit/reports/viability-test-v0")
OUT.mkdir(parents=True, exist_ok=True)
D18=Path("/home/jeanbatuli/sra_v14_18_output_dynamic_linkage/output_dynamic_results.csv")
D14R=Path("/home/jeanbatuli/sra_v14_14_temporal_control_outputs/temporal_token_results.csv")
D14C=Path("/home/jeanbatuli/sra_v14_14_temporal_control_outputs/temporal_token_curves.csv")
d18=pd.read_csv(D18)
d14r=pd.read_csv(D14R)
d14c=pd.read_csv(D14C)

def oof_binary(df, features, target, groups, n_splits=6):
    y=df[target].astype(int).to_numpy()
    if len(np.unique(y))<2:
        return None
    g=df[groups].astype(str).to_numpy()
    X=df[features].replace([np.inf,-np.inf],np.nan)
    splits=min(n_splits, len(np.unique(g)))
    cv=GroupKFold(n_splits=splits)
    prob=np.full(len(df), np.nan)
    for tr,te in cv.split(X,y,g):
        if len(np.unique(y[tr]))<2:
            continue
        pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("lr",LogisticRegression(max_iter=500,class_weight="balanced"))])
        pipe.fit(X.iloc[tr],y[tr])
        prob[te]=pipe.predict_proba(X.iloc[te])[:,1]
    ok=np.isfinite(prob)
    if ok.sum()==0 or len(np.unique(y[ok]))<2:
        return None
    pred=(prob[ok]>=0.5).astype(int)
    return {"n":int(ok.sum()),"positives":int(y[ok].sum()),"auroc":float(roc_auc_score(y[ok],prob[ok])),"balanced_accuracy":float(balanced_accuracy_score(y[ok],pred)),"brier":float(brier_score_loss(y[ok],prob[ok]))}
dyn=["ratio_norm","dynamic_mean","dynamic_max","transition_entropy","markov_stability","oscillation_rate","recovery_rate","top5_prob_mean","logit_entropy_mean"]
d18["any_marker"]=((d18["uncertainty_markers"]>0)|(d18["refusal_markers"]>0)|(d18["contradiction_markers"]>0)).astype(int)
rep_thr=float(d18["repetition_rate"].quantile(.75))
d18["high_repetition"]=(d18["repetition_rate"]>=rep_thr).astype(int)
for c in ["uncertainty_markers","refusal_markers","contradiction_markers"]:
    d18[c+"_bin"]=(d18[c]>0).astype(int)

part_a={"repetition_threshold_q75":rep_thr,"class_counts":{},"full_dynamic":{},"entropy_only":{},"continuous_spearman":{}}
targets=["any_marker","high_repetition","uncertainty_markers_bin","refusal_markers_bin","contradiction_markers_bin"]
for t in targets:
    part_a["class_counts"][t]=d18[t].value_counts().to_dict()
    part_a["full_dynamic"][t]=oof_binary(d18,dyn,t,"prompt")
    part_a["entropy_only"][t]=oof_binary(d18,["logit_entropy_mean"],t,"prompt")
for o in ["repetition_rate","lexical_diversity","compression_ratio","output_length","mean_sentence_length"]:
    vals={}
    for f in dyn:
        x=d18[[f,o]].replace([np.inf,-np.inf],np.nan).dropna()
        if len(x)>2:
            rho,pv=spearmanr(x[f],x[o])
            vals[f]={"rho":float(rho),"p":float(pv)}
    part_a["continuous_spearman"][o]=vals

keys=["model","label","condition","prompt"]
def prefix_features(g, frac):
    g=g.sort_values("step")
    n=max(2,int(np.ceil(len(g)*frac)))
    x=g.iloc[:n]
    ent=x["entropy"].to_numpy(float); kap=x["kappa"].to_numpy(float); rat=x["ratio_partial"].to_numpy(float)
    def slope(a):
        return float(np.polyfit(np.arange(len(a)),a,1)[0]) if len(a)>1 else 0.0
    de=np.diff(ent)
    osc=float(np.mean(np.sign(de[1:])*np.sign(de[:-1])<0)) if len(de)>1 else 0.0
    pos=float(np.maximum(de,0).sum()) if len(de) else 0.0
    return pd.Series({"entropy_mean_prefix":float(np.mean(ent)),"entropy_std_prefix":float(np.std(ent)),"entropy_slope_prefix":slope(ent),"positive_entropy_variation":pos,"kappa_mean_prefix":float(np.mean(kap)),"kappa_std_prefix":float(np.std(kap)),"kappa_slope_prefix":slope(kap),"top5_mean_prefix":float(x["top5"].mean()),"margin_mean_prefix":float(x["margin"].mean()),"ratio_last_prefix":float(rat[-1]),"ratio_slope_prefix":slope(rat),"oscillation_prefix":osc})
part_b={}
prefix_cols=["entropy_mean_prefix","entropy_std_prefix","entropy_slope_prefix","positive_entropy_variation","kappa_mean_prefix","kappa_std_prefix","kappa_slope_prefix","top5_mean_prefix","margin_mean_prefix","ratio_last_prefix","ratio_slope_prefix","oscillation_prefix"]
for frac in [.25,.50,.75]:
    pref=d14c.groupby(keys,dropna=False).apply(lambda g: prefix_features(g,frac),include_groups=False).reset_index()
    m=pref.merge(d14r,on=keys,how="inner",validate="one_to_one")
    m["high_final_ratio"]=(m["ratio_final"]>=m["ratio_final"].median()).astype(int)
    m["high_final_entropy"]=(m["entropy_mean"]>=m["entropy_mean"].median()).astype(int)
    part_b[str(frac)]={
        "n":int(len(m)),
        "high_final_ratio_full_prefix":oof_binary(m,prefix_cols,"high_final_ratio","prompt"),
        "high_final_ratio_entropy_only":oof_binary(m,["entropy_mean_prefix"],"high_final_ratio","prompt"),
        "high_final_entropy_full_prefix":oof_binary(m,prefix_cols,"high_final_entropy","prompt"),
        "high_final_entropy_entropy_only":oof_binary(m,["entropy_mean_prefix"],"high_final_entropy","prompt")
    }

base=d18[d18["condition"]=="BASELINE"].set_index(["model","prompt"])
paired=[]
for _,r in d18[d18["condition"]!="BASELINE"].iterrows():
    k=(r["model"],r["prompt"])
    if k not in base.index:
        continue
    b=base.loc[k]
    row={"model":r["model"],"prompt":r["prompt"],"condition":r["condition"]}
    for c in dyn+["repetition_rate","lexical_diversity","uncertainty_markers","refusal_markers","contradiction_markers","compression_ratio","output_length"]:
        row["delta_"+c]=float(r[c]-b[c])
    paired.append(row)
pair=pd.DataFrame(paired)
part_c={"n_pairs":int(len(pair)),"by_condition":{},"delta_correlations":{}}
for cond,g in pair.groupby("condition"):
    rec={}
    for c in ["delta_ratio_norm","delta_transition_entropy","delta_oscillation_rate","delta_recovery_rate","delta_logit_entropy_mean","delta_repetition_rate","delta_lexical_diversity","delta_compression_ratio","delta_output_length"]:
        v=g[c].dropna()
        rec[c]={"mean":float(v.mean()),"median":float(v.median())}
        if len(v)>2 and np.any(v!=0):
            try:
                rec[c]["wilcoxon_p"]=float(wilcoxon(v).pvalue)
            except Exception:
                pass
    part_c["by_condition"][cond]=rec
for a in ["delta_ratio_norm","delta_transition_entropy","delta_oscillation_rate","delta_recovery_rate","delta_logit_entropy_mean"]:
    part_c["delta_correlations"][a]={}
    for o in ["delta_repetition_rate","delta_lexical_diversity","delta_compression_ratio","delta_output_length","delta_uncertainty_markers","delta_refusal_markers","delta_contradiction_markers"]:
        x=pair[[a,o]].replace([np.inf,-np.inf],np.nan).dropna()
        if len(x)>2 and x[a].nunique()>1 and x[o].nunique()>1:
            rho,pv=spearmanr(x[a],x[o])
            part_c["delta_correlations"][a][o]={"rho":float(rho),"p":float(pv)}

result={"design":{"part_a":"same-run full dynamics -> output properties, prompt-grouped OOF","part_b":"same-run prefix dynamics -> final dynamic endpoint, prompt-grouped OOF","part_c":"within model+prompt intervention minus baseline deltas","no_claim":"No early-output-failure or causal claim from V0"},"part_a":part_a,"part_b":part_b,"part_c":part_c}
(OUT/"results.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
