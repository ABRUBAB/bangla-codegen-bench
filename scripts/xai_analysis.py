#!/usr/bin/env python3
"""Explainability analyses on the *outcome* data (RQ5), complementing the white-box probes
(scripts/xai_attribution.py):

  1. GEE logistic regression (task-clustered) : pass ~ variant [+ token-inflation + task difficulty] per model
  2. Failure-risk model + SHAP : gradient-boosted classifier predicting a failed sample from prompt/model features;
     grouped (by task) cross-validation; global and per-feature SHAP values
  3. Variant-level mediation check : does the token-inflation ratio explain the variant effect?

Outputs data/results/xai_gee.csv, xai_shap_values.npz/csv, xai_model_cv.json
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
warnings.filterwarnings("ignore")

PARAMS_B = {"qwen2.5-coder-1.5b": 1.54, "qwen2.5-coder-3b": 3.09, "qwen2.5-coder-7b": 7.62,
            "deepseek-coder-6.7b": 6.74, "qwen3-8b": 8.19, "tigercoder-1b": 1.0, "tigercoder-9b": 12.2,
            "gemini-3.8-flash": np.nan}
FAMILY = {"qwen2.5-coder-1.5b": "qwen", "qwen2.5-coder-3b": "qwen", "qwen2.5-coder-7b": "qwen",
          "deepseek-coder-6.7b": "deepseek", "qwen3-8b": "qwen", "tigercoder-1b": "tiger",
          "tigercoder-9b": "tiger", "gemini-3.8-flash": "gemini"}


def features(ex: pd.DataFrame, tok: pd.DataFrame) -> pd.DataFrame:
    t = tok.copy()
    en = t[t.variant == "EN"].set_index("task_id")
    for name in ("qwen2.5", "deepseek-coder", "gpt-4o (o200k)"):
        t[f"ratio_{name}"] = t.apply(lambda r: r[f"desc__{name}"] / en.loc[r["task_id"], f"desc__{name}"]
                                     if r["task_id"] in en.index and not np.isnan(r.get(f"desc__{name}", np.nan)) else np.nan, axis=1)
    t["tok_per_word_qwen"] = t["desc__qwen2.5"] / t["desc_words"]
    t["tok_per_char_qwen"] = t["desc__qwen2.5"] / t["desc_chars"]
    t["bn_share"] = t["variant"].map(lambda v: {"BN": 1, "BN_SPELL": 1, "BN_TYPO": 1, "BN_MIX": 0.55}.get(v, 0.0))
    cols = ["task_id", "variant", "desc__qwen2.5", "desc_chars", "desc_words", "ratio_qwen2.5", "ratio_deepseek-coder",
            "ratio_gpt-4o (o200k)", "tok_per_word_qwen", "tok_per_char_qwen", "bn_share"]
    d = ex.merge(t[cols], on=["task_id", "variant"], how="left")
    # task difficulty = EN pass rate of the *other* models (leave-one-model-out)
    en_pass = ex[(ex.variant == "EN") & (ex.condition == "DIRECT")].groupby(["task_id", "model_id"])["pass"].mean().unstack()
    diff = {}
    for m in en_pass.columns:
        diff[m] = en_pass.drop(columns=[m]).mean(axis=1)
    d["difficulty"] = d.apply(lambda r: diff.get(r["model_id"], pd.Series(dtype=float)).get(r["task_id"], np.nan), axis=1)
    d["log_params"] = d["model_id"].map(lambda m: np.log(PARAMS_B.get(m, np.nan)) if not np.isnan(PARAMS_B.get(m, np.nan)) else np.nan)
    d["is_security"] = (d["kind"] == "security").astype(int)
    return d


def run_gee(d: pd.DataFrame, out: Path):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    rows = []
    for m, g in d[(d.condition == "DIRECT") & (d.variant != "ID_ONLY") & (~d.variant.str.endswith("_HINT"))].groupby("model_id"):
        g = g.dropna(subset=["ratio_qwen2.5", "difficulty"]).copy()
        g["fail"] = 1 - g["pass"]
        g["log_ratio"] = np.log(g["ratio_qwen2.5"])
        for label, formula in [("variant", "pass ~ C(variant, Treatment('EN'))"),
                               ("variant+difficulty", "pass ~ C(variant, Treatment('EN')) + difficulty"),
                               ("tokens", "pass ~ log_ratio + difficulty"),
                               ("variant+tokens", "pass ~ C(variant, Treatment('EN')) + log_ratio + difficulty")]:
            try:
                res = smf.gee(formula, groups="task_id", data=g, family=sm.families.Binomial(),
                              cov_struct=sm.cov_struct.Exchangeable()).fit()
            except Exception as exc:
                continue
            for name in res.params.index:
                rows.append({"model_id": m, "spec": label, "term": name, "coef": res.params[name],
                             "se": res.bse[name], "p": res.pvalues[name], "OR": float(np.exp(res.params[name])),
                             "n": int(len(g))})
    pd.DataFrame(rows).to_csv(out / "xai_gee.csv", index=False)


def run_shap(d: pd.DataFrame, out: Path):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    import shap
    d = d[(d.condition == "DIRECT") & (d.status != "harness_error")].copy()
    d["fail"] = (~d["functional_pass"]).astype(int)
    for v in ["EN", "EN_TYPO", "BN", "BN_SPELL", "BN_MIX", "BN_TYPO", "BANGLISH", "ID_ONLY"]:
        d[f"var_{v}"] = (d["variant"] == v).astype(int)
    for f in sorted(d["model_id"].map(FAMILY).dropna().unique()):
        d[f"fam_{f}"] = (d["model_id"].map(FAMILY) == f).astype(int)
    d["log_params"] = d["log_params"].fillna(d["log_params"].median())
    base_feats = ["is_security", "difficulty", "log_params"] + [c for c in d.columns if c.startswith("fam_")]
    var_feats = [c for c in d.columns if c.startswith("var_")]
    tok_feats = ["ratio_qwen2.5", "ratio_deepseek-coder", "ratio_gpt-4o (o200k)", "tok_per_word_qwen", "tok_per_char_qwen",
                 "desc__qwen2.5", "desc_chars", "bn_share"]
    sets = {"task+model": base_feats, "+variant": base_feats + var_feats,
            "+variant+tokenization": base_feats + var_feats + tok_feats}
    d = d.dropna(subset=["difficulty"])
    X_all = d[sets["+variant+tokenization"]].astype(float)
    y, groups = d["fail"].values, d["task_id"].values
    cv = {}
    gkf = GroupKFold(n_splits=5)
    for name, cols in sets.items():
        aucs = []
        for tr, te in gkf.split(d, y, groups):
            clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250, random_state=0)
            clf.fit(d.iloc[tr][cols].astype(float), y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(d.iloc[te][cols].astype(float))[:, 1]))
        cv[name] = {"auroc_mean": float(np.mean(aucs)), "auroc_sd": float(np.std(aucs))}
    (out / "xai_model_cv.json").write_text(json.dumps(cv, indent=2))
    # final model on all data for SHAP
    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250, random_state=0).fit(X_all, y)
    sub = X_all.sample(min(3000, len(X_all)), random_state=0)
    try:
        expl = shap.TreeExplainer(clf)
        sv = expl.shap_values(sub)
        if isinstance(sv, list):
            sv = sv[1]
    except Exception:
        bg = X_all.sample(min(200, len(X_all)), random_state=1)
        expl = shap.Explainer(lambda z: clf.predict_proba(pd.DataFrame(z, columns=X_all.columns))[:, 1], bg)
        sv = expl(sub.iloc[:800]).values
        sub = sub.iloc[:800]
    np.savez_compressed(out / "xai_shap_values.npz", shap=np.asarray(sv), X=sub.values, cols=np.array(sub.columns))
    imp = pd.DataFrame({"feature": sub.columns, "mean_abs_shap": np.abs(sv).mean(0)}).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(out / "xai_shap_importance.csv", index=False)
    return cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default="data/results/exec_merged.csv")
    ap.add_argument("--tokens", default="data/results/token_counts.csv")
    ap.add_argument("--out", default="data/results")
    a = ap.parse_args()
    out = Path(a.out)
    ex = pd.read_csv(a.merged)
    ex["functional_pass"] = ex["functional_pass"].astype(bool)
    tok = pd.read_csv(a.tokens)
    d = features(ex, tok)
    d.to_csv(out / "xai_features.csv", index=False)
    run_gee(d, out)
    cv = run_shap(d, out)
    print("CV AUROC:", json.dumps(cv))


if __name__ == "__main__":
    main()
