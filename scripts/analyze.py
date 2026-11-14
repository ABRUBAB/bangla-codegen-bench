#!/usr/bin/env python3
"""Main statistical analysis (RQ1-RQ5). Reads execution/static/token data, writes CSV tables +
results.json (every number quoted in the paper comes from here via paper/numbers.tex).

usage: python scripts/analyze.py --exec data/results/exec.jsonl --bandit data/results/bandit.jsonl \
          --gens data/generations --out data/results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bcb import stats as S  # noqa: E402
from bcb.tasks import load_tasks, TASK_DIR  # noqa: E402
from run_execution import load_gens  # noqa: E402

VARIANTS = ["EN", "EN_TYPO", "BN", "BN_SPELL", "BN_MIX", "BN_TYPO", "BANGLISH"]
BN_FAMILY = ["BN", "BN_SPELL", "BN_MIX", "BN_TYPO", "BANGLISH"]
# Bandit test ids that are relevant to each CWE (used to decide whether a *relevant* static flag exists)
CWE_BANDIT = {
    "CWE-89": {"B608"}, "CWE-78": {"B602", "B603", "B604", "B605", "B606", "B607"},
    "CWE-22": set(), "CWE-502": {"B506", "B301", "B302", "B403", "B614"},
    "CWE-79": {"B701", "B703", "B308", "B704"}, "CWE-94": {"B307", "B102", "B101"},
    "CWE-1333": set(), "CWE-916": {"B303", "B324"}, "CWE-327": {"B303", "B304", "B305", "B324"},
    "CWE-328": {"B303", "B324"}, "CWE-338": {"B311"}, "CWE-601": set(), "CWE-918": {"B310"},
    "CWE-377": {"B306", "B108"}, "CWE-732": {"B103"},
}


def load(args):
    ex = pd.DataFrame([json.loads(l) for l in Path(args.exec).read_text(encoding="utf-8").splitlines() if l.strip()])
    ex["pass"] = ex["functional_pass"].astype(float)
    gens = load_gens(Path(args.gens), None)
    meta = pd.DataFrame([{k: g.get(k) for k in ("gen_id", "prompt_tokens", "output_tokens", "thought_tokens",
                          "translation_tokens_in", "translation_tokens_out", "finish", "extraction_status",
                          "signature_reinjected")} for g in gens.values()])
    ex = ex.merge(meta, on="gen_id", how="left")
    ex["thought_tokens"] = ex["thought_tokens"].fillna(0)
    ex["total_tokens"] = (ex["prompt_tokens"].fillna(0) + ex["output_tokens"].fillna(0) + ex["thought_tokens"] +
                          ex["translation_tokens_in"].fillna(0) + ex["translation_tokens_out"].fillna(0))
    if args.bandit and Path(args.bandit).exists():
        bd = {}
        for l in Path(args.bandit).read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l); bd[r["gen_id"]] = r
        tasks = {t["task_id"]: t for t in load_tasks([p for p in TASK_DIR.glob("*.jsonl") if not p.name.startswith("_")])}

        def flags(row):
            r = bd.get(row["gen_id"], {"issues": []})
            iss = r["issues"]
            t = tasks[row["task_id"]]
            rel = set().union(*[CWE_BANDIT.get(c, set()) for c in t.get("cwe_ids", [])]) if t.get("cwe_ids") else set()
            cw = set(t.get("cwe_ids", []))
            any_f = any(i["severity"] in ("MEDIUM", "HIGH") for i in iss)
            rel_f = any((i["test_id"] in rel) or (f"CWE-{i['cwe']}" in cw) for i in iss)
            return pd.Series({"bandit_any": any_f, "bandit_relevant": rel_f})
        ex = pd.concat([ex, ex.apply(flags, axis=1)], axis=1)
    tasks_df = pd.DataFrame([{"task_id": t["task_id"], "kind": t["kind"], "category": t["category"],
                              "cwe": (t.get("cwe_ids") or [""])[0]} for t in
                             load_tasks([p for p in TASK_DIR.glob("*.jsonl") if not p.name.startswith("_")])])
    ex = ex.merge(tasks_df[["task_id", "category"]], on="task_id", how="left")
    return ex


def cell_table(ex, value="pass", subset=None, n_boot=2000):
    """Per (model, variant, condition): mean with task-cluster bootstrap CI."""
    d = ex if subset is None else ex[subset]
    rows = []
    for (m, v, c), g in d.groupby(["model_id", "variant", "condition"]):
        pt = g.groupby("task_id")[value].mean().values
        mean, lo, hi = S.cluster_bootstrap_mean(pt, n_boot)
        rows.append({"model_id": m, "variant": v, "condition": c, "n_tasks": len(pt), "n_samples": len(g),
                     "mean": mean, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def contrasts(ex, value="pass", subset=None, base="EN", cond="DIRECT", variants=None, n_boot=2000):
    """Variant-vs-EN paired contrasts per model with bootstrap CI, Wilcoxon, exact McNemar, Holm."""
    d = (ex if subset is None else ex[subset])
    d = d[d["condition"] == cond]
    rows = []
    for m, gm in d.groupby("model_id"):
        piv = gm.groupby(["task_id", "variant"])[value].mean().unstack("variant")
        if base not in piv:
            continue
        fam = []
        for v in (variants or [c for c in piv.columns if c != base]):
            if v not in piv:
                continue
            a, b = piv[v].values, piv[base].values
            diff, lo, hi = S.cluster_bootstrap_diff(a, b, n_boot)
            _, pw = S.wilcoxon_paired(a, b)
            n10, n01, pm = S.mcnemar_exact((a >= 0.5).astype(float), (b >= 0.5).astype(float))
            fam.append({"model_id": m, "variant": v, "base": base, "diff": diff, "lo": lo, "hi": hi,
                        "p_wilcoxon": pw, "n_base_only": n10 and 0, "p_mcnemar": pm,
                        "n_variant_better": n01, "n_base_better": n10,
                        "odds_ratio": S.odds_ratio_paired(n01, n10)})
        if fam:
            adj = S.holm([f["p_wilcoxon"] for f in fam])
            for f, ap in zip(fam, adj):
                f["p_holm"] = ap
            rows += fam
    return pd.DataFrame(rows)


def mitigation(ex, n_boot=2000):
    """TRANSLATE_FIRST vs DIRECT for BN/BN_MIX/BANGLISH: dPass, token ratio (total workflow tokens)."""
    rows = []
    for m, gm in ex.groupby("model_id"):
        for v in ["BN", "BN_MIX", "BANGLISH"]:
            d = gm[(gm["variant"] == v) & (gm["condition"] == "DIRECT")]
            t = gm[(gm["variant"] == v) & (gm["condition"] == "TRANSLATE_FIRST")]
            if d.empty or t.empty:
                continue
            pd_ = d.groupby("task_id")["pass"].mean(); pt_ = t.groupby("task_id")["pass"].mean()
            idx = pd_.index.intersection(pt_.index)
            diff, lo, hi = S.cluster_bootstrap_diff(pt_[idx].values, pd_[idx].values, n_boot)
            _, p = S.wilcoxon_paired(pt_[idx].values, pd_[idx].values)
            td = d.groupby("task_id")["total_tokens"].mean(); tt = t.groupby("task_id")["total_tokens"].mean()
            ratio = float((tt[idx] / td[idx]).median())
            en = gm[(gm["variant"] == "EN") & (gm["condition"] == "DIRECT")].groupby("task_id")["pass"].mean()
            gap_closed = np.nan
            gap = (en[idx] - pd_[idx]).mean()
            if gap > 1e-9:
                gap_closed = float(diff / gap)
            rows.append({"model_id": m, "variant": v, "pass_direct": float(pd_[idx].mean()),
                         "pass_translate": float(pt_[idx].mean()), "delta": diff, "lo": lo, "hi": hi, "p": p,
                         "gap_vs_en": float(gap), "gap_closed_frac": gap_closed,
                         "tokens_direct": float(td[idx].mean()), "tokens_translate": float(tt[idx].mean()),
                         "token_ratio_median": ratio})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = S.holm(out["p"].tolist())
    return out


def security(ex, n_boot=2000):
    """Security outcomes on security tasks.
    SC  = functional pass & exploit pass (secure and correct)
    VC  = functional pass & exploit fail (vulnerable but working)
    vulnerability rate V = VC / (SC+VC)  (conditional on working code)
    """
    d = ex[(ex["kind"] == "security") & (ex["status"] != "harness_error")].copy()
    d["SC"] = ((d["functional_pass"]) & (d["exploit_pass"] == True)).astype(float)  # noqa: E712
    d["VC"] = ((d["functional_pass"]) & (d["exploit_pass"] == False)).astype(float)  # noqa: E712
    d["FF"] = (~d["functional_pass"]).astype(float)
    rows = []
    for (m, v, c), g in d.groupby(["model_id", "variant", "condition"]):
        pt = g.groupby("task_id").agg(SC=("SC", "mean"), VC=("VC", "mean"), FF=("FF", "mean"))
        sc, sc_lo, sc_hi = S.cluster_bootstrap_mean(pt["SC"].values, n_boot)
        # conditional vulnerability rate: ratio-of-sums bootstrap over tasks
        rng = np.random.default_rng(42)
        vc, sc_ = pt["VC"].values, pt["SC"].values
        idx = rng.integers(0, len(pt), size=(n_boot, len(pt)))
        num, den = vc[idx].sum(1), (vc[idx] + sc_[idx]).sum(1)
        ok = den > 0
        vr = vc.sum() / max(1e-9, (vc + sc_).sum())
        lo, hi = (np.quantile(num[ok] / den[ok], [0.025, 0.975]) if ok.any() else (np.nan, np.nan))
        rows.append({"model_id": m, "variant": v, "condition": c, "n_tasks": len(pt),
                     "SC": sc, "SC_lo": sc_lo, "SC_hi": sc_hi, "VC_mean": float(pt["VC"].mean()),
                     "FF_mean": float(pt["FF"].mean()), "vuln_rate": float(vr), "vuln_lo": float(lo), "vuln_hi": float(hi)})
    return pd.DataFrame(rows), d


def static_vs_dynamic(d):
    """On working security samples: Bandit (relevant / any) vs exploit outcome."""
    if "bandit_relevant" not in d:
        return pd.DataFrame()
    w = d[d["functional_pass"] & d["exploit_pass"].notna()].copy()
    w["vulnerable"] = (w["exploit_pass"] == False)  # noqa: E712
    rows = []
    for name in ("bandit_relevant", "bandit_any"):
        tp = int((w[name] & w["vulnerable"]).sum()); fp = int((w[name] & ~w["vulnerable"]).sum())
        fn = int((~w[name] & w["vulnerable"]).sum()); tn = int((~w[name] & ~w["vulnerable"]).sum())
        n = tp + fp + fn + tn
        rows.append({"static_rule": name, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "n": n,
                     "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
                     "kappa": S.cohens_kappa(w[name].astype(int), w["vulnerable"].astype(int)) if n else np.nan,
                     "spdf_rate": fn / max(1, tp + fn)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec", default="data/results/exec.jsonl")
    ap.add_argument("--bandit", default="data/results/bandit.jsonl")
    ap.add_argument("--gens", default="data/generations")
    ap.add_argument("--out", default="data/results")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ex = load(a)
    ex.to_csv(out / "exec_merged.csv", index=False, encoding="utf-8")
    res = {}

    func = ex["kind"] == "functional"
    cells = cell_table(ex, "pass", func, a.boot); cells.to_csv(out / "tab_pass_functional.csv", index=False)
    cells_s = cell_table(ex, "pass", ~func, a.boot); cells_s.to_csv(out / "tab_pass_security_tasks.csv", index=False)
    cells_all = cell_table(ex, "pass", None, a.boot); cells_all.to_csv(out / "tab_pass_all.csv", index=False)
    con = contrasts(ex, "pass", None, n_boot=a.boot); con.to_csv(out / "tab_contrasts_all.csv", index=False)
    contrasts(ex, "pass", func, n_boot=a.boot).to_csv(out / "tab_contrasts_functional.csv", index=False)
    contrasts(ex, "pass", ~func, n_boot=a.boot).to_csv(out / "tab_contrasts_security_tasks.csv", index=False)
    mit = mitigation(ex, a.boot); mit.to_csv(out / "tab_mitigation.csv", index=False)
    sec, secd = security(ex, a.boot); sec.to_csv(out / "tab_security.csv", index=False)
    sd = static_vs_dynamic(secd); sd.to_csv(out / "tab_static_vs_dynamic.csv", index=False)
    # per-CWE vulnerability by model/variant
    if not secd.empty:
        cw = secd.groupby(["model_id", "variant", "category"]).agg(
            VC=("VC", "sum"), SC=("SC", "sum"), FF=("FF", "sum"), n=("VC", "size")).reset_index()
        cw["vuln_rate"] = cw["VC"] / (cw["VC"] + cw["SC"]).replace(0, np.nan)
        cw.to_csv(out / "tab_cwe.csv", index=False)
    # failure types
    ft = ex[~ex["functional_pass"]].groupby(["model_id", "variant", "func_failure_type"]).size().rename("n").reset_index()
    ft.to_csv(out / "tab_failure_types.csv", index=False)
    # tokens per generation
    tk = ex.groupby(["model_id", "variant", "condition"])[["prompt_tokens", "output_tokens", "total_tokens"]].mean().reset_index()
    tk.to_csv(out / "tab_gen_tokens.csv", index=False)
    res["n_exec"] = int(len(ex)); res["models"] = sorted(ex["model_id"].unique().tolist())
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print("analysis tables written to", out)


if __name__ == "__main__":
    main()
