#!/usr/bin/env python3
"""Does the Bangla penalty depend on model size / recency? (RQ1 add-on)

Per model: functional Pass@1 in English, the paired formal-Bangla minus English gap (pp, task-cluster bootstrap CI), the odds ratio
of a per-model logistic GEE (task-clustered), the effect of the security reminder in English and in formal Bangla.
Across models: Spearman correlations of the gap with log10(parameters), of the log odds ratio with log10(parameters), of the gap with
English accuracy (ceiling effect) and with the Bangla token premium of the model's own tokenizer; a pooled GEE with a
formal-Bangla x log10(parameters) interaction. Writes paper/scale_numbers.tex (\\SG{key}) and figures/fig_scale.{pdf,png}.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bcb.stats import cluster_bootstrap_diff  # noqa: E402
from figstyle import MODEL_LABEL, NPG, save, setup  # noqa: E402

TAG = {"qwen2.5-coder-1.5b": "qA", "qwen2.5-coder-3b": "qB", "qwen2.5-coder-7b": "qC", "deepseek-coder-6.7b": "dsA",
       "qwen3-8b": "qwT", "tigercoder-1b": "tgA", "qwen2.5-coder-7b-q4": "qCn", "qwen2.5-coder-14b": "qD",
       "qwen3.8-27b": "qwN", "gemma-4-31b": "gmA", "ministral-3-14b": "msA", "gemma-4-e4b": "gmB"}
TIER1 = {"qwen2.5-coder-1.5b", "qwen2.5-coder-3b", "qwen2.5-coder-7b", "deepseek-coder-6.7b", "qwen3-8b", "tigercoder-1b"}
TOKENIZER = {"qwen2.5-coder-1.5b": "qwen2.5", "qwen2.5-coder-3b": "qwen2.5", "qwen2.5-coder-7b": "qwen2.5", "qwen2.5-coder-7b-q4": "qwen2.5",
             "qwen2.5-coder-14b": "qwen2.5", "deepseek-coder-6.7b": "deepseek-coder", "qwen3-8b": "qwen3", "tigercoder-1b": "tigercoder",
             "qwen3.8-27b": "qwen3.8", "gemma-4-31b": "gemma4", "gemma-4-e4b": "gemma4", "ministral-3-14b": "ministral3"}
SHORT = {"qwen2.5-coder-1.5b": "Qwen2.5-Coder 1.5B", "qwen2.5-coder-3b": "Qwen2.5-Coder 3B", "qwen2.5-coder-7b": "Qwen2.5-Coder 7B",
         "qwen2.5-coder-7b-q4": "Qwen2.5-Coder 7B (Q4)", "qwen2.5-coder-14b": "Qwen2.5-Coder 14B (Q4)", "deepseek-coder-6.7b": "DeepSeek-Coder 6.7B",
         "qwen3-8b": "Qwen3 8B", "tigercoder-1b": "TigerCoder 1B", "qwen3.8-27b": "Qwen3.8 27B (Q4)", "gemma-4-31b": "Gemma-4 31B (Q4)",
         "gemma-4-e4b": "Gemma-4 E4B (Q4)", "ministral-3-14b": "Ministral-3 14B (Q4)"}
LANGS = ["EN", "BN", "BN_MIX", "BANGLISH", "ES"]
KEYS: dict[str, str] = {}


def fnum(x: float, d: int = 1) -> str:
    s = f"{abs(x):.{d}f}"
    return ("$-$" + s) if x < 0 and float(s) != 0 else s


def fp(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def put(k: str, v: str) -> None:
    KEYS[k] = v


def main() -> None:
    ex = pd.read_csv(ROOT / "data/results/exec_merged.csv")
    ex = ex[(ex.condition == "DIRECT") & (ex.status != "harness_error")]
    mods = json.load(open(ROOT / "data/models.json", encoding="utf-8"))
    params = {k: mods[k]["params"] / 1e9 for k in TAG}
    tok = pd.read_csv(ROOT / "data/results/token_counts.csv")
    en = tok[tok.variant == "EN"].set_index("task_id")
    bn = tok[tok.variant == "BN"].set_index("task_id")
    premium = {t: float(np.median(bn[f"desc__{t}"] / en[f"desc__{t}"])) for t in set(TOKENIZER.values())}

    f = ex[(ex.kind == "functional") & ex.variant.isin(LANGS)].copy()
    f["y"] = f.functional_pass.astype(float)
    rows = []
    for m, g in f.groupby("model_id"):
        if m not in TAG:
            continue
        pt = g.groupby(["task_id", "variant"]).y.mean().unstack()
        r = {"model_id": m, "params": params[m], "tier": 1 if m in TIER1 else 2, "EN": 100 * pt["EN"].mean(), "premium": premium[TOKENIZER[m]]}
        for v in LANGS[1:]:
            d, lo, hi = cluster_bootstrap_diff(pt[v].values, pt["EN"].values)
            r[v], r[v + "_lo"], r[v + "_hi"] = 100 * d, 100 * lo, 100 * hi
        gg = g[g.variant.isin(["EN", "BN"])]
        res = smf.gee("y ~ C(variant, Treatment('EN'))", groups="task_id", data=gg, family=sm.families.Binomial(),
                      cov_struct=sm.cov_struct.Exchangeable()).fit()
        k = [i for i in res.params.index if "variant" in i][0]
        ci = res.conf_int().loc[k]
        r["OR"], r["OR_lo"], r["OR_hi"] = float(np.exp(res.params[k])), float(np.exp(ci[0])), float(np.exp(ci[1]))
        rows.append(r)
    d = pd.DataFrame(rows).sort_values("params").reset_index(drop=True)

    # security reminder: vulnerable share among working security samples, English vs formal Bangla
    s = ex[(ex.kind == "security") & ex.functional_pass & ex.exploit_pass.notna()].copy()
    s["v"] = (s.exploit_pass == False).astype(float)  # noqa: E712
    vt = s.pivot_table(index="model_id", columns="variant", values="v", aggfunc="mean") * 100
    d["hEN"] = [vt.loc[m, "EN_HINT"] - vt.loc[m, "EN"] for m in d.model_id]
    d["hBN"] = [vt.loc[m, "BN_HINT"] - vt.loc[m, "BN"] for m in d.model_id]
    d["VEN"] = [vt.loc[m, "EN"] for m in d.model_id]
    d["VBN"] = [vt.loc[m, "BN"] for m in d.model_id]
    d.to_csv(ROOT / "data/results/tab_scale.csv", index=False)

    # security tasks: paired formal-Bangla minus English gap in functional pass and in secure-and-correct share (pp)
    sec = ex[(ex.kind == "security") & ex.variant.isin(["EN", "BN"])].copy()
    sec["func"] = sec.functional_pass.astype(float)
    sec["sc"] = (sec.functional_pass & (sec.exploit_pass == True)).astype(float)  # noqa: E712
    for m, g in sec.groupby("model_id"):
        if m not in TAG:
            continue
        for col, nm in (("func", "sf"), ("sc", "sc")):
            pt = g.groupby(["task_id", "variant"])[col].mean().unstack()
            dd, lo, hi = cluster_bootstrap_diff(pt["BN"].values, pt["EN"].values)
            put(f"{TAG[m]}-{nm}Gap", fnum(100 * dd)); put(f"{TAG[m]}-{nm}Gap-lo", fnum(100 * lo)); put(f"{TAG[m]}-{nm}Gap-hi", fnum(100 * hi))
        put(f"{TAG[m]}-scEN", fnum(100 * g[g.variant == "EN"].sc.mean(), 0)); put(f"{TAG[m]}-scBN", fnum(100 * g[g.variant == "BN"].sc.mean(), 0))

    for _, r in d.iterrows():
        t = TAG[r.model_id]
        put(f"{t}-EN", fnum(r.EN)); put(f"{t}-params", f"{r.params:.1f}" if r.params < 10 else f"{r.params:.0f}")
        for v, w in (("BN", "BN"), ("BN_MIX", "BNMIX"), ("BANGLISH", "BL"), ("ES", "ES")):
            put(f"{t}-{w}", fnum(r[v])); put(f"{t}-{w}-lo", fnum(r[v + "_lo"])); put(f"{t}-{w}-hi", fnum(r[v + "_hi"]))
        put(f"{t}-OR", f"{r.OR:.2f}"); put(f"{t}-OR-lo", f"{r.OR_lo:.2f}"); put(f"{t}-OR-hi", f"{r.OR_hi:.2f}")
        put(f"{t}-hEN", fnum(r.hEN)); put(f"{t}-hBN", fnum(r.hBN)); put(f"{t}-VEN", fnum(r.VEN, 0)); put(f"{t}-VBN", fnum(r.VBN, 0))

    lp = np.log10(d.params)
    for name, x, y in (("rho", lp, d.BN), ("orrho", lp, np.log(d.OR)), ("enrho", d.EN, d.BN), ("tokrho", d.premium, d.BN),
                       ("tokorrho", d.premium, np.log(d.OR))):
        rho, p = stats.spearmanr(x, y)
        put(name, fnum(rho, 2)); put(name + "p", fp(p))
    put("nNeg", str(int((d.BN < 0).sum()))); put("nMod", str(len(d)))
    put("nNegCIexcl", str(int((d.BN_hi < 0).sum())))
    put("gapMin", fnum(d.BN.min())); put("gapMax", fnum(d.BN.max()))
    put("ORmedian", f"{d.OR.median():.2f}")

    gg = f[f.variant.isin(["EN", "BN"])].copy()
    gg["lp"] = np.log10(gg.model_id.map(params))
    gg["bn"] = (gg.variant == "BN").astype(float)
    res = smf.gee("y ~ bn * lp + C(model_id)", groups="task_id", data=gg, family=sm.families.Binomial(),
                  cov_struct=sm.cov_struct.Exchangeable()).fit()
    ci = res.conf_int()
    put("int-OR", f"{np.exp(res.params['bn:lp']):.2f}"); put("int-lo", f"{np.exp(ci.loc['bn:lp', 0]):.2f}")
    put("int-hi", f"{np.exp(ci.loc['bn:lp', 1]):.2f}"); put("int-p", fp(res.pvalues["bn:lp"]))

    # ---- translate-first (Qwen3-8B translator): share of the English gap that is closed, cells with a gap above 3 pp ----
    mq = pd.read_csv(ROOT / "data/results/tab_mitigation_q3.csv")
    cells = mq[(mq.variant != "ES") & (mq.gap_vs_en > 0.03)].copy()
    cells["t1"] = cells.model_id.isin(TIER1)
    put("gcN", str(len(cells)))
    put("gcMed", f"{100 * cells.gap_closed_frac.median():.0f}")
    put("gcMin", f"{100 * cells.gap_closed_frac.min():.0f}")
    put("gcMax", f"{100 * cells.gap_closed_frac.max():.0f}")
    for nm, sub in (("T1", cells[cells.t1]), ("T2", cells[~cells.t1])):
        put(f"gcN{nm}", str(len(sub)))
        put(f"gcMed{nm}", f"{100 * sub.gap_closed_frac.median():.0f}")
        put(f"gcMin{nm}", f"{100 * sub.gap_closed_frac.min():.0f}")
        put(f"gcMax{nm}", f"{100 * sub.gap_closed_frac.max():.0f}")

    # ---- figure: forest plots sorted by parameter count ----
    setup()
    n = len(d)
    y = np.arange(n)[::-1]
    col = [NPG["navy"] if t == 1 else NPG["red"] for t in d.tier]
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.4), sharey=True, gridspec_kw={"width_ratios": [1, 1, 1.05]})
    ax = axs[0]
    ax.errorbar(d.BN, y, xerr=[d.BN - d.BN_lo, d.BN_hi - d.BN], fmt="none", ecolor="#888", elinewidth=0.8, capsize=1.5)
    ax.scatter(d.BN, y, c=col, s=22, zorder=3, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="#444", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{SHORT[m]}  [{p:.1f}B]" if p < 10 else f"{SHORT[m]}  [{p:.0f}B]" for m, p in zip(d.model_id, d.params)], fontsize=6.6)
    ax.set_xlabel("Formal Bangla minus English,\nPass@1 (pp)")
    ax.text(0.0, 1.03, "(a) accuracy gap", transform=ax.transAxes, fontsize=8, fontweight="bold")
    ax = axs[1]
    ax.errorbar(d.OR, y, xerr=[d.OR - d.OR_lo, d.OR_hi - d.OR], fmt="none", ecolor="#888", elinewidth=0.8, capsize=1.5)
    ax.scatter(d.OR, y, c=col, s=22, zorder=3, edgecolor="white", linewidth=0.4)
    ax.axvline(1, color="#444", lw=0.6)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.3, 1, 3])
    ax.set_xticklabels(["0.1", "0.3", "1", "3"])
    ax.set_xlim(0.03, 6.5)
    ax.set_xlabel("Odds ratio of a correct solution,\nformal Bangla vs English")
    ax.text(0.0, 1.03, "(b) odds scale", transform=ax.transAxes, fontsize=8, fontweight="bold")
    ax = axs[2]
    for yi, (a, b) in zip(y, zip(d.hEN, d.hBN)):
        ax.plot([a, b], [yi, yi], color="#bbb", lw=1.0, zorder=1)
    ax.scatter(d.hEN, y, c=NPG["teal"], s=22, zorder=3, label="English reminder", edgecolor="white", linewidth=0.4)
    ax.scatter(d.hBN, y, c=NPG["purple"], s=22, zorder=3, marker="D", label="Bangla reminder", edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="#444", lw=0.6)
    ax.set_xlabel("Change in vulnerable share of\nworking code with the reminder (pp)")
    ax.text(0.0, 1.03, "(c) security reminder", transform=ax.transAxes, fontsize=8, fontweight="bold")
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], marker="o", ls="", color=NPG["navy"]), Line2D([], [], marker="o", ls="", color=NPG["red"]),
                Line2D([], [], marker="o", ls="", color=NPG["teal"]), Line2D([], [], marker="D", ls="", color=NPG["purple"])],
               ["tier 1 (fp16, Kaggle T4)", "tier 2 (Q4 GGUF, RTX 5090)", "English reminder", "Bangla reminder"], loc="lower center",
               ncol=4, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    save(fig, "fig_scale", ROOT / "figures")

    out = ["% auto-generated by scripts/scale_analysis.py - do not edit", "\\newcommand{\\SG}[1]{\\csname sg@#1\\endcsname}"]
    out += [f"\\expandafter\\def\\csname sg@{k}\\endcsname{{{v}}}" for k, v in KEYS.items()]
    (ROOT / "paper/scale_numbers.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(d[["model_id", "params", "EN", "BN", "BN_lo", "BN_hi", "OR", "OR_lo", "OR_hi", "hEN", "hBN"]].round(2).to_string())
    print({k: KEYS[k] for k in ("rho", "rhop", "orrho", "orrhop", "enrho", "enrhop", "tokrho", "tokrhop", "int-OR", "int-lo", "int-hi", "int-p", "nNeg", "nNegCIexcl")})


if __name__ == "__main__":
    main()
