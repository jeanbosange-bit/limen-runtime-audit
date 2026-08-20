"""
phase9_logit_lens.py

MISSION CLAUDE PHASE 9 -- representation decodability vs INTERMEDIATE_PROJECTED
readout concentration. Read-only against
~/limen-runtime-audit/reports/deep-feature-extraction-2026-08-19/. No model loading,
no new generation, no source-data modification. Descriptive only.
"""
import warnings
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score

warnings.filterwarnings("ignore")
np.random.seed(20260819)

SRC = "/home/jeanbatuli/limen-runtime-audit/reports/deep-feature-extraction-2026-08-19"
OUT = "/home/jeanbatuli/V20_probabilistic_dynamics/reports/decodability-v0-2026-08-19"

LOGIT_LENS_MODELS = [
    "EleutherAI/pythia-160m", "HuggingFaceTB/SmolLM-360M-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "deepseek-ai/deepseek-coder-1.3b-instruct", "facebook/opt-125m", "gpt2",
    "ibm-granite/granite-3.0-1b-a400m-instruct",
]
# already computed in the V0 pilot (ORIGINAL representation), reused not recomputed
ALREADY_DONE = ["Qwen/Qwen2.5-0.5B-Instruct", "gpt2", "EleutherAI/pythia-160m"]
TO_COMPUTE = [m for m in LOGIT_LENS_MODELS if m not in ALREADY_DONE]

DEPTH_BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
DEPTH_LABELS = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
PROJ_COLS = [f"proj_{i}" for i in range(128)]

# ============================================================
# PHASE 1 -- verify logit-lens subset (INTERMEDIATE_PROJECTED)
# ============================================================
print("=== PHASE 1: logit-lens coverage ===")
readout = pq.read_table(f"{SRC}/readout_features.parquet").to_pandas()
ip = readout[(readout.readout_type == "INTERMEDIATE_PROJECTED") & (readout.model_id.isin(LOGIT_LENS_MODELS))].copy()
ip["normalized_depth"] = ip["layer_index"] / ip["num_layers"]
ip["depth_bin"] = pd.cut(ip["normalized_depth"], bins=DEPTH_BIN_EDGES, labels=DEPTH_LABELS, include_lowest=True, right=False)

coverage_rows = []
for m in LOGIT_LENS_MODELS:
    sub = ip[ip.model_id == m]
    fam_counts = sub.prompt_family.value_counts()
    coverage_rows.append(dict(
        model_id=m,
        total_rows=len(sub),
        n_layers_present=sub.layer_index.nunique(),
        layer_min=int(sub.layer_index.min()) if len(sub) else None,
        layer_max=int(sub.layer_index.max()) if len(sub) else None,
        num_layers=int(sub.num_layers.iloc[0]) if len(sub) else None,
        depth_min=round(float(sub.normalized_depth.min()), 3) if len(sub) else None,
        depth_max=round(float(sub.normalized_depth.max()), 3) if len(sub) else None,
        n_runs=sub.run_id.nunique(),
        n_seeds=sub.seed.nunique(),
        n_prompt_families=sub.prompt_family.nunique(),
        min_rows_per_family=int(fam_counts.min()) if len(fam_counts) else 0,
        max_rows_per_family=int(fam_counts.max()) if len(fam_counts) else 0,
        depth_bins_present=sorted(sub.depth_bin.dropna().unique().tolist(), key=lambda x: DEPTH_LABELS.index(x)),
    ))
    print(f"  {m:45s} rows={len(sub):6d} layers={sub.layer_index.nunique():3d} families={sub.prompt_family.nunique()}")

cov_df = pd.DataFrame(coverage_rows)
cov_df.to_csv(f"{OUT}/logit_lens_model_coverage.csv", index=False)
print(f"Wrote logit_lens_model_coverage.csv ({len(cov_df)} models)")

# ============================================================
# Decodability: reuse pilot results for 3 models, compute for 5 more
# ============================================================
print("\n=== Loading existing pilot decodability (ORIGINAL) for 3 models ===")
prior = pd.read_csv(f"{OUT}/pilot_metrics.csv")
prior_orig = prior[(prior.representation == "ORIGINAL") & (prior.model_id.isin(ALREADY_DONE)) & (prior.status == "OK")].copy()
prior_orig = prior_orig[["model_id", "representation", "depth_bin", "n_rows", "n_classes_present",
                          "fold_balanced_acc_mean", "fold_balanced_acc_std", "fold_macro_f1_mean", "chance_level"]]
print(f"  reused {len(prior_orig)} rows from pilot_metrics.csv")

print(f"\n=== PHASE 3/4: computing decodability for {len(TO_COMPUTE)} additional models ===")
df = pq.read_table(f"{SRC}/h_random_projection.parquet").to_pandas()
df = df[df.model_id.isin(TO_COMPUTE)].copy()
df["normalized_depth"] = df["layer_index"] / df["num_layers"]
df["depth_bin"] = pd.cut(df["normalized_depth"], bins=DEPTH_BIN_EDGES, labels=DEPTH_LABELS, include_lowest=True, right=False)
classes = sorted(df["prompt_family"].unique())
n_classes = len(classes)
chance = 1.0 / n_classes
seeds = sorted(df["seed"].unique())
print(f"  rows={len(df)}, classes={n_classes}, chance={chance:.4f}, seeds={seeds}")

new_metric_rows = []
for model_id in TO_COMPUTE:
    for depth_label in DEPTH_LABELS:
        sub = df[(df.model_id == model_id) & (df.depth_bin == depth_label)]
        if len(sub) < n_classes * 6:
            new_metric_rows.append(dict(model_id=model_id, representation="ORIGINAL", depth_bin=depth_label,
                                         n_rows=len(sub), n_classes_present=sub.prompt_family.nunique(),
                                         fold_balanced_acc_mean=None, fold_balanced_acc_std=None,
                                         fold_macro_f1_mean=None, chance_level=chance, status="SKIPPED_INSUFFICIENT_ROWS"))
            continue
        fold_accs, fold_f1s = [], []
        for test_seed in seeds:
            train = sub[sub.seed != test_seed]
            test = sub[sub.seed == test_seed]
            if train.prompt_family.nunique() < n_classes or test.prompt_family.nunique() < n_classes:
                continue
            Xtr, Xte = train[PROJ_COLS].values, test[PROJ_COLS].values
            ytr, yte = train["prompt_family"].values, test["prompt_family"].values
            scaler = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=2000, multi_class="multinomial")
            clf.fit(scaler.transform(Xtr), ytr)
            pred = clf.predict(scaler.transform(Xte))
            fold_accs.append(balanced_accuracy_score(yte, pred))
            fold_f1s.append(f1_score(yte, pred, average="macro"))
        if not fold_accs:
            new_metric_rows.append(dict(model_id=model_id, representation="ORIGINAL", depth_bin=depth_label,
                                         n_rows=len(sub), n_classes_present=sub.prompt_family.nunique(),
                                         fold_balanced_acc_mean=None, fold_balanced_acc_std=None,
                                         fold_macro_f1_mean=None, chance_level=chance, status="SKIPPED_MISSING_CLASS_IN_FOLD"))
            continue
        new_metric_rows.append(dict(model_id=model_id, representation="ORIGINAL", depth_bin=depth_label,
                                     n_rows=len(sub), n_classes_present=sub.prompt_family.nunique(),
                                     fold_balanced_acc_mean=float(np.mean(fold_accs)),
                                     fold_balanced_acc_std=float(np.std(fold_accs)),
                                     fold_macro_f1_mean=float(np.mean(fold_f1s)),
                                     chance_level=chance, status="OK"))
        print(f"  {model_id:45s} {depth_label:8s} n={len(sub):6d} bal_acc={np.mean(fold_accs):.3f}+-{np.std(fold_accs):.3f}")

new_decode_df = pd.DataFrame(new_metric_rows)
decode_all = pd.concat([prior_orig, new_decode_df[["model_id","representation","depth_bin","n_rows","n_classes_present",
                                                     "fold_balanced_acc_mean","fold_balanced_acc_std","fold_macro_f1_mean","chance_level"]]],
                        ignore_index=True)
decode_all.to_csv(f"{OUT}/phase9_decodability_all8.csv", index=False)
print(f"\nWrote phase9_decodability_all8.csv ({len(decode_all)} rows)")

# ============================================================
# PHASE 2/4 -- readout concentration aggregates per (model, depth_bin)
# ============================================================
print("\n=== Aggregating INTERMEDIATE_PROJECTED readout per (model, depth_bin) ===")
ip["top1_top2_gap_intermediate"] = ip["top1_prob"] - ip["top2_prob"]
agg = ip.groupby(["model_id", "depth_bin"], observed=True).agg(
    n_rows=("entropy", "size"),
    entropy_mean=("entropy", "mean"),
    entropy_std=("entropy", "std"),
    top1_prob_mean=("top1_prob", "mean"),
    top2_prob_mean=("top2_prob", "mean"),
    top1_top2_gap_mean=("top1_top2_gap_intermediate", "mean"),
    commitment_score_mean=("commitment_score", "mean"),
    branching_factor_mean=("branching_factor", "mean"),
    concentration_mean=("concentration", "mean"),
    collapse_score_mean=("collapse_score", "mean"),
).reset_index()
agg.to_csv(f"{OUT}/phase9_readout_agg.csv", index=False)
print(f"Wrote phase9_readout_agg.csv ({len(agg)} rows)")

# ============================================================
# Merge decodability + readout -> aligned_depth_summary / decodability_vs_readout
# ============================================================
merged = pd.merge(decode_all, agg, on=["model_id", "depth_bin"], how="outer", suffixes=("_decode", "_readout"))
merged["decodability_margin"] = merged["fold_balanced_acc_mean"] - merged["chance_level"]
merged.to_csv(f"{OUT}/decodability_vs_readout.csv", index=False)
merged.to_csv(f"{OUT}/aligned_depth_summary.csv", index=False)
print(f"Wrote decodability_vs_readout.csv / aligned_depth_summary.csv ({len(merged)} rows)")

print("\nDONE PHASE 1-4/9/10 data prep.")
