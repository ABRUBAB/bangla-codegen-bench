"""Result figures (Pass@1, contrasts, security, CWE, static-vs-dynamic, mitigation, scaling, hint, errors)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from figstyle import (MODEL_LABEL, MODEL_ORDER, NPG, VARIANT_COLOR, VARIANT_LABEL, VARIANT_ORDER, save)  # noqa: E402

R = ROOT / "data/results"
F = ROOT / "figures"
PARAMS = {"qwen2.5-coder-1.5b": 1.54, "qwen2.5-coder-3b": 3.09, "qwen2.5-coder-7b": 7.62}
PAL = plt.cm.Paired(np.linspace(0, 1, 12))  # twelve distinct model colours


def _read(name):
    p = R / name
    if not p.exists():
        return None
    try:
        d = pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return None
    return d if len(d) else None


def _ordered(df, col="model_id"):
    return [m for m in MODEL_ORDER if m in set(df[col])]


def fig_pass():
    f, s = _read("tab_pass_functional.csv"), _read("tab_pass_security_tasks.csv")
    if f is None:
        return
    vs = VARIANT_ORDER + ["ID_ONLY"]
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)
    models = _ordered(f)
    for ax, df, lab in ((axs[0], f, f"(a) functional tasks (n={int(f.n_tasks.max())})"), (axs[1], s, f"(b) security tasks, functional tests (n={int(s.n_tasks.max()) if s is not None else 0})")):
        if df is None:
            continue
        d = df[df.condition == "DIRECT"]
        w = 0.8 / len(vs)
        for i, v in enumerate(vs):
            sub = d[d.variant == v].set_index("model_id").reindex(models)
            x = np.arange(len(models)) + (i - (len(vs) - 1) / 2) * w
            ax.bar(x, 100 * sub["mean"], w * 0.92, color=VARIANT_COLOR[v], label=VARIANT_LABEL[v], linewidth=0,
                   yerr=[100 * (sub["mean"] - sub["lo"]), 100 * (sub["hi"] - sub["mean"])],
                   error_kw={"elinewidth": 0.5, "capsize": 0.8, "capthick": 0.5, "ecolor": "#333"})
        ax.set_ylabel("Pass@1 (%)")
        ax.set_ylim(0, 105)
        ax.text(0.0, 1.02, lab, transform=ax.transAxes, fontsize=8, fontweight="bold")
    axs[1].set_xticks(range(len(models)))
    axs[1].set_xticklabels([MODEL_LABEL[m] for m in models], fontsize=7, rotation=30, ha="right")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=8, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=7, columnspacing=1.0, handlelength=1.0)
    fig.subplots_adjust(hspace=0.18)
    save(fig, "fig_pass", F)


def fig_forest():
    cf, cs = _read("tab_contrasts_functional.csv"), _read("tab_contrasts_security_tasks.csv")
    if cf is None:
        return
    vs = ["EN_TYPO", "BN", "BN_SPELL", "BN_MIX", "BN_TYPO", "BANGLISH", "ES", "ES_TYPO", "ID_ONLY"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    for ax, df, lab in ((axs[0], cf, "(a) functional tasks"), (axs[1], cs, "(b) security tasks")):
        if df is None:
            continue
        models = _ordered(df)
        for j, m in enumerate(models):
            d = df[df.model_id == m].set_index("variant").reindex(vs)
            y = np.arange(len(vs))[::-1] + (j - (len(models) - 1) / 2) * 0.09
            ax.errorbar(100 * d["diff"], y, xerr=[100 * (d["diff"] - d["lo"]), 100 * (d["hi"] - d["diff"])], fmt="o", ms=2.8,
                        lw=0.7, color=PAL[j], label=MODEL_LABEL[m], capsize=0)
        ax.axvline(0, color="k", lw=0.7)
        ax.set_xlabel(r"$\Delta$ Pass@1 vs EN (percentage points)")
        ax.text(0.0, 1.02, lab, transform=ax.transAxes, fontsize=8, fontweight="bold")
    axs[0].set_yticks(np.arange(len(vs))[::-1])
    axs[0].set_yticklabels([VARIANT_LABEL[v] for v in vs])
    axs[1].legend(loc="lower right", fontsize=6, handletextpad=0.2)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "fig_forest", F)


def fig_security():
    ex = _read("exec_merged.csv")
    if ex is None:
        return
    d = ex[(ex.kind == "security") & (ex.condition == "DIRECT")].copy()
    if d.empty:
        return
    d["SC"] = (d.functional_pass & (d.exploit_pass == True)).astype(float)  # noqa: E712
    d["VC"] = (d.functional_pass & (d.exploit_pass == False)).astype(float)  # noqa: E712
    d["FF"] = (~d.functional_pass).astype(float)
    vs = ["EN", "EN_TYPO", "BN", "BN_SPELL", "BN_MIX", "BN_TYPO", "BANGLISH", "ES", "ES_TYPO", "ID_ONLY"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1.0, 1.0]})
    ax = axs[0]
    agg = d.groupby(["variant", "task_id"])[["SC", "VC", "FF"]].mean().groupby("variant").mean().reindex(vs)
    bottom = np.zeros(len(vs))
    for col, c, lab in (("SC", NPG["teal"], "secure & correct"), ("VC", NPG["red"], "correct but vulnerable"), ("FF", "#CBD5E1", "incorrect")):
        ax.bar(range(len(vs)), 100 * agg[col], bottom=bottom, color=c, label=lab, width=0.75, linewidth=0)
        bottom += 100 * agg[col].values
    ax.set_xticks(range(len(vs)))
    ax.set_xticklabels([VARIANT_LABEL[v] for v in vs], rotation=35, ha="right")
    ax.set_ylabel("share of samples (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.0)
    ax.text(-0.15, 1.12, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=9)
    ax = axs[1]
    sec = _read("tab_security.csv")
    if sec is not None:
        s = sec[sec.condition == "DIRECT"]
        models = _ordered(s)
        cols = {"EN": NPG["navy"], "BN": NPG["red"], "BN_MIX": NPG["teal"], "BANGLISH": NPG["purple"]}
        for i, m in enumerate(models):
            q = s[s.model_id == m].set_index("variant")
            vals = [100 * q.loc[v, "vuln_rate"] for v in cols if v in q.index]
            ax.plot([min(vals), max(vals)], [i, i], color="#bbbbbb", lw=1.0, zorder=1)
            for v, c in cols.items():
                if v in q.index:
                    ax.scatter(100 * q.loc[v, "vuln_rate"], i, s=13, color=c, zorder=3, linewidth=0,
                               label={"EN": "EN", "BN": "BN", "BN_MIX": "BN-mix", "BANGLISH": "Banglish"}[v] if i == 0 else None)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([MODEL_LABEL[m] for m in models], fontsize=6.5)
        ax.invert_yaxis()
        ax.set_xlabel("vulnerability rate V (%)")
        ax.legend(fontsize=6.5, ncol=4, loc="lower center", bbox_to_anchor=(0.45, 1.0), handletextpad=0.1, columnspacing=0.8, frameon=False)
    ax.text(-0.62, 1.12, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=9)
    fig.subplots_adjust(wspace=0.75, left=0.08, right=0.98, bottom=0.2)
    save(fig, "fig_security", F)


def fig_cwe():
    c = _read("tab_cwe.csv")
    if c is None:
        return
    groups = {"EN": ["EN"], "BN family": ["BN", "BN_SPELL", "BN_TYPO"], "BN-mix": ["BN_MIX"], "Banglish": ["BANGLISH"]}
    cwes = sorted(c["category"].unique(), key=lambda x: int(x.split("-")[1]))
    M = np.full((len(cwes), len(groups)), np.nan)
    for j, (g, vs) in enumerate(groups.items()):
        sub = c[c.variant.isin(vs)].groupby("category")[["VC", "SC"]].sum()
        for i, cw in enumerate(cwes):
            if cw in sub.index and sub.loc[cw].sum() > 0:
                M[i, j] = sub.loc[cw, "VC"] / sub.loc[cw].sum()
    fig, ax = plt.subplots(figsize=(3.8, 4.6))
    im = ax.imshow(100 * M, cmap="RdYlGn_r", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(list(groups), rotation=30, ha="right")
    ax.set_yticks(range(len(cwes)))
    ax.set_yticklabels(cwes, fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{100 * M[i, j]:.0f}", ha="center", va="center", fontsize=7.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("vulnerability rate V (%)", fontsize=7)
    save(fig, "fig_cwe", F)


def fig_static():
    t = _read("tab_static_vs_dynamic.csv")
    if t is None or t.empty:
        return
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axs[0]
    lab = {"bandit_relevant": "Bandit, CWE-relevant", "bandit_any": "Bandit, any medium+"}
    names = ["flagged & vulnerable (TP)", "unflagged & vulnerable (SPDF)", "flagged & secure (FP)", "unflagged & secure (TN)"]
    for i, (_, r) in enumerate(t.iterrows()):
        bottom = 0
        for v, c, n in zip([r.TP, r.FN, r.FP, r.TN], [NPG["teal"], NPG["crimson"], NPG["coral"], "#CBD5E1"], names):
            ax.barh(i, 100 * v / r.n, left=bottom, color=c, label=n if i == 0 else None, height=0.55, linewidth=0)
            bottom += 100 * v / r.n
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([lab[x] for x in t.static_rule])
    ax.set_xlabel("share of working security samples (%)")
    ax.legend(fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.text(-0.3, 1.2, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=9)
    ax = axs[1]
    x = np.arange(len(t))
    w = 0.25
    ax.bar(x - w, 100 * t.precision, w, color=NPG["navy"], label="precision")
    ax.bar(x, 100 * t.recall, w, color=NPG["red"], label="recall")
    ax.bar(x + w, 100 * t.kappa, w, color=NPG["purple"], label=r"Cohen $\kappa\times100$")
    ax.set_xticks(x)
    ax.set_xticklabels(["CWE-relevant", "any medium+"])
    ax.set_ylim(0, 100)
    ax.legend(fontsize=6.5, loc="upper right")
    ax.text(-0.2, 1.2, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=9)
    fig.subplots_adjust(wspace=0.45)
    save(fig, "fig_static", F)


def fig_pareto():
    arms = [("tab_mitigation_clean.csv", "(a) own sanitised translation"), ("tab_mitigation_q3.csv", "(b) Qwen3-8B as translator")]
    data = [(_read(f), ttl) for f, ttl in arms]
    data = [(t, ttl) for t, ttl in data if t is not None and not t.empty]
    if not data:
        return
    fig, axs = plt.subplots(1, len(data), figsize=(3.6 * len(data), 3.6), sharey=True, squeeze=False)
    mk = {"BN": "o", "BN_MIX": "s", "BANGLISH": "^", "ES": "D"}
    models = _ordered(pd.concat([t for t, _ in data]))
    for ax, (t, ttl) in zip(axs[0], data):
        for j, m in enumerate(models):
            for v, mm in mk.items():
                r = t[(t.model_id == m) & (t.variant == v)]
                if r.empty:
                    continue
                ax.errorbar(r.token_ratio_median, 100 * r.delta, yerr=[100 * (r.delta - r.lo), 100 * (r.hi - r.delta)], fmt=mm, ms=4.5,
                            color=PAL[j], lw=0.6, capsize=0)
        ax.axhline(0, color="k", lw=0.7)
        ax.axvline(1, color="grey", lw=0.5, ls="--")
        ax.set_xlabel("workflow tokens: translate-first / direct (median)")
        ax.set_title(ttl, fontsize=8, loc="left")
    axs[0][0].set_ylabel(r"$\Delta$ Pass@1 vs direct (pp)")
    h1 = [Line2D([0], [0], marker=mm, color="k", lw=0, ms=4, label=VARIANT_LABEL[v]) for v, mm in mk.items()]
    h2 = [Line2D([0], [0], marker="o", color=PAL[j], lw=0, ms=4, label=MODEL_LABEL[m]) for j, m in enumerate(models)]
    fig.legend(handles=h1 + h2, fontsize=6.5, loc="lower center", ncol=8, handletextpad=0.1, columnspacing=0.9, frameon=False)
    fig.subplots_adjust(wspace=0.08, bottom=0.3)
    save(fig, "fig_pareto", F)


def fig_scaling():
    f = _read("tab_pass_functional.csv")
    if f is None:
        return
    d = f[(f.condition == "DIRECT") & f.model_id.isin(PARAMS)]
    if d.model_id.nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for v in ["EN", "BN", "BN_MIX", "BANGLISH", "ID_ONLY"]:
        q = d[d.variant == v].copy()
        q["p"] = q.model_id.map(PARAMS)
        q = q.sort_values("p")
        ax.errorbar(q.p, 100 * q["mean"], yerr=[100 * (q["mean"] - q.lo), 100 * (q.hi - q["mean"])], marker="o", ms=3, lw=1,
                    color=VARIANT_COLOR[v], label=VARIANT_LABEL[v], capsize=1.5)
    ax.set_xscale("log")
    ax.set_xticks([1.5, 3, 7.6])
    ax.set_xticklabels(["1.5B", "3B", "7B"])
    ax.set_xlabel("Qwen2.5-Coder parameters")
    ax.set_ylabel("Pass@1, functional (%)")
    ax.legend(fontsize=6.5)
    save(fig, "fig_scaling", F)


def fig_hint():
    ex = _read("exec_merged.csv")
    if ex is None:
        return
    d = ex[(ex.kind == "security") & (ex.condition == "DIRECT") & ex.functional_pass].copy()
    d["vuln"] = (d.exploit_pass == False).astype(float)  # noqa: E712
    langs = [("EN", "EN_HINT"), ("BN", "BN_HINT"), ("BN_MIX", "BN_MIX_HINT"), ("BANGLISH", "BANGLISH_HINT"), ("ES", "ES_HINT")]
    if not d.variant.isin([h for _, h in langs]).any():
        return
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    x = np.arange(len(langs))
    w = 0.35
    base = [100 * d[d.variant == b].vuln.mean() for b, _ in langs]
    hint = [100 * d[d.variant == h].vuln.mean() for _, h in langs]
    ax.bar(x - w / 2, base, w, color=NPG["red"], label="no hint")
    ax.bar(x + w / 2, hint, w, color=NPG["teal"], label="with hint")
    ax.set_xticks(x)
    ax.set_xticklabels(["EN", "BN", "BN-mix", "Banglish", "ES"])
    ax.set_ylabel("vulnerability rate V (%)")
    ax.set_ylim(0, 72)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.tight_layout()
    save(fig, "fig_hint", F)


def fig_errors():
    ex = _read("exec_merged.csv")
    if ex is None:
        return
    d = ex[(ex.condition == "DIRECT") & (~ex.functional_pass)]
    vs = ["EN", "EN_TYPO", "BN", "BN_SPELL", "BN_MIX", "BN_TYPO", "BANGLISH", "ES", "ES_TYPO", "ID_ONLY"]
    cats = ["wrong_answer", "signature_mismatch", "missing_symbol", "runtime_error", "syntax_error", "import_error", "timeout", "no_code", "other"]
    tab = d.groupby(["variant", "func_failure_type"]).size().unstack(fill_value=0).reindex(vs).fillna(0)
    totals = ex[ex.condition == "DIRECT"].groupby("variant").size().reindex(vs)
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    bottom = np.zeros(len(vs))
    pal = plt.cm.Set2(np.linspace(0, 1, len(cats)))
    for c, col in zip(cats, pal):
        if c not in tab:
            continue
        v = 100 * tab[c].values / totals.values
        ax.bar(range(len(vs)), v, bottom=bottom, color=col, label=c.replace("_", " "), width=0.75, linewidth=0)
        bottom += v
    ax.set_xticks(range(len(vs)))
    ax.set_xticklabels([VARIANT_LABEL[v] for v in vs], rotation=35, ha="right")
    ax.set_ylabel("failed samples (% of all)")
    ax.legend(fontsize=5.8, ncol=2, loc="upper right")
    save(fig, "fig_errors", F)


ALL = [fig_pass, fig_forest, fig_security, fig_cwe, fig_static, fig_pareto, fig_scaling, fig_hint, fig_errors]
