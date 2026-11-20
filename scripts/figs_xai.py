"""Explainability figures: SHAP summary, token-attribution profiles, segment shares + reference NLL, layer-wise alignment."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from figstyle import NPG, VARIANT_COLOR, VARIANT_LABEL, save  # noqa: E402

R = ROOT / "data/results"
X = ROOT / "data/xai"
SEG_COLOR = {"head": "#E5E7EB", "signature": "#BFDBFE", "description": "#FDE68A", "tail": "#E5E7EB", "template": "#FFFFFF"}
FEAT_LABEL = {"difficulty": "task difficulty", "log_params": "model size (log params)",
              "is_security": "security task", "ratio_qwen2.5": "token ratio (Qwen)", "ratio_deepseek-coder": "token ratio (DeepSeek)",
              "ratio_gpt-4o (o200k)": "token ratio (o200k)", "tok_per_word_qwen": "tokens per word", "tok_per_char_qwen": "tokens per character",
              "desc__qwen2.5": "description length (tokens)", "desc_chars": "description length (chars)", "bn_share": "share of Bangla script"}


def fig_shap():
    p = R / "xai_shap_values.npz"
    if not p.exists():
        return
    import shap
    z = np.load(p, allow_pickle=True)
    cols = [FEAT_LABEL.get(c, c.replace("var_ID_ONLY", "variant: signature only").replace("var_", "variant: ").replace("fam_", "family: ")) for c in z["cols"]]
    sv, Xv = z["shap"], z["X"]
    plt.figure(figsize=(3.8, 3.2))
    shap.summary_plot(sv, Xv, feature_names=cols, show=False, max_display=14, plot_size=None)
    fig = plt.gcf()
    fig.axes[0].tick_params(labelsize=6.5)
    fig.axes[0].set_xlabel("SHAP value (impact on failure risk)", fontsize=6.5)
    for ax_ in fig.axes[1:]:
        ax_.tick_params(labelsize=6)
        ax_.yaxis.label.set_size(6.5)
    # the library's automatic layout collapses the plot area for long feature names: place the axes by hand
    fig.axes[0].set_position([0.43, 0.13, 0.44, 0.85])
    if len(fig.axes) > 1:
        fig.axes[1].set_position([0.90, 0.13, 0.018, 0.85])
    save(fig, "fig_shap", ROOT / "figures")


def fig_attribution():
    import re
    files = sorted(X.glob("tokens_*.json")) if X.exists() else []
    if not files:
        return
    # choose model/task with all four variants available
    by = {}
    for f in files:
        m = re.match(r"tokens_(?P<model>.+?)_(?P<tid>[FS]\d{3})_(?P<var>.+)$", f.stem)
        if m:
            by.setdefault((m["model"], m["tid"]), {})[m["var"]] = f
    pick = None
    for k, v in by.items():
        if {"EN", "BN", "BN_MIX", "BANGLISH"} <= set(v):
            pick = k
            if k[1] == "F025":
                break
    if pick is None:
        return
    model, tid = pick
    vs = ["EN", "BN", "BN_MIX", "BANGLISH"]
    fig, axs = plt.subplots(len(vs), 1, figsize=(2.95, 3.2), sharex=False)
    for ax, v in zip(axs, vs):
        d = json.loads(by[pick][v].read_text(encoding="utf-8"))
        a = np.abs(np.array(d["attr"]))
        seg = d["segments"]
        a = a / a[[i for i, s in enumerate(seg) if s != "template"]].sum()
        for i, s in enumerate(seg):
            ax.axvspan(i - 0.5, i + 0.5, color=SEG_COLOR.get(s, "#fff"), lw=0)
        ax.bar(range(len(a)), a, color=VARIANT_COLOR[v], width=0.9, linewidth=0)
        ax.set_ylabel(VARIANT_LABEL[v], rotation=0, ha="right", va="center", fontsize=7)
        ax.set_yticks([])
        ax.set_xlim(-0.5, len(a) - 0.5)
        ax.set_xticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=SEG_COLOR[s]) for s in ("signature", "description", "head")]
    axs[0].legend(handles, ["signature", "description", "instruction"], ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=5.5, handlelength=0.9, columnspacing=0.6, handletextpad=0.4)
    axs[-1].set_xlabel(f"prompt token position (task {tid})", fontsize=6.5)
    fig.subplots_adjust(hspace=0.12)
    save(fig, "fig_attribution", ROOT / "figures")


def main_xai_model():
    """The largest model for which white-box probes exist (order of preference: 7B, 3B, 1.5B)."""
    for m in ("qwen2.5-coder-7b", "qwen2.5-coder-3b", "qwen2.5-coder-1.5b"):
        if (X / f"xai_{m}.jsonl").exists() and (X / f"xai_{m}_hidden.npz").exists():
            return m
    return None


def fig_segments(model=None):
    model = model or main_xai_model()
    p = X / f"xai_{model}.jsonl"
    if model is None or not p.exists():
        return
    d = pd.read_json(p, lines=True)
    vs = [v for v in ["EN", "EN_TYPO", "BN", "BN_SPELL", "BN_MIX", "BN_TYPO", "BANGLISH", "ES", "ID_ONLY"] if v in set(d.variant)]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axs[0]
    agg = d.groupby("variant")[["attr_head", "attr_signature", "attr_description", "attr_tail"]].mean().reindex(vs)
    bottom = np.zeros(len(vs))
    for col, c, lab in (("attr_signature", "#60A5FA", "signature"), ("attr_description", "#F59E0B", "description"),
                        ("attr_head", "#9CA3AF", "head instruction"), ("attr_tail", "#D1D5DB", "tail instruction")):
        ax.bar(range(len(vs)), 100 * agg[col], bottom=bottom, color=c, label=lab, width=0.75, linewidth=0)
        bottom += 100 * agg[col].values
    ax.set_xticks(range(len(vs)))
    ax.set_xticklabels([VARIANT_LABEL[v] for v in vs], rotation=35, ha="right")
    ax.set_ylabel("share of prompt saliency (%)")
    ax.legend(fontsize=6, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.text(-0.17, 1.1, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=9)
    ax = axs[1]
    en = d[d.variant == "EN"].set_index("task_id")["nll"]
    data, labs = [], []
    for v in vs:
        if v == "EN":
            continue
        x = d[d.variant == v].set_index("task_id")["nll"]
        data.append((x - en.reindex(x.index)).dropna().values)
        labs.append(VARIANT_LABEL[v])
    bp = ax.boxplot(data, vert=True, widths=0.6, patch_artist=True, showfliers=False, medianprops={"color": "k", "lw": 0.9})
    for b, v in zip(bp["boxes"], [v for v in vs if v != "EN"]):
        b.set_facecolor(VARIANT_COLOR[v]); b.set_alpha(0.8); b.set_linewidth(0.5)
    ax.axhline(0, color="grey", ls="--", lw=0.7)
    ax.set_xticks(range(1, len(labs) + 1))
    ax.set_xticklabels(labs, rotation=35, ha="right")
    ax.set_ylabel(r"$\Delta$NLL of reference vs EN (nats/token)")
    ax.text(-0.2, 1.1, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=9)
    fig.subplots_adjust(wspace=0.4)
    save(fig, "fig_segments", ROOT / "figures")


def fig_alignment(model=None):
    model = model or main_xai_model()
    p = X / f"xai_{model}_hidden.npz"
    if model is None or not p.exists():
        return
    z = np.load(p, allow_pickle=True)
    keys = [str(k) for k in z["keys"]]
    last = z["last"].astype(np.float32)  # N, L+1, D
    idx = {k: i for i, k in enumerate(keys)}
    tasks = sorted({k.split("|")[0] for k in keys})
    vs = [v for v in ["EN_TYPO", "BN", "BN_SPELL", "BN_MIX", "BN_TYPO", "BANGLISH", "ES", "ID_ONLY"] if any(k.endswith("|" + v) for k in keys)]
    nl = last.shape[1]

    def cos(a, b):
        return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)
    curves, per_task = {}, {}
    for v in vs:
        ts = [t for t in tasks if f"{t}|EN" in idx and f"{t}|{v}" in idx]
        c = np.stack([cos(last[idx[f"{t}|EN"]], last[idx[f"{t}|{v}"]]) for t in ts])  # T, L+1
        curves[v] = c.mean(0)
        per_task[v] = dict(zip(ts, c[:, int(0.75 * nl):].mean(1)))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    for v in vs:
        ax.plot(range(nl), curves[v], color=VARIANT_COLOR[v], lw=1.2, label=VARIANT_LABEL[v])
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine similarity to EN\n(last prompt token)")
    ax.legend(fontsize=6, ncol=2)
    ax.text(-0.17, 1.04, "(c)", transform=ax.transAxes, fontweight="bold", fontsize=9)
    ax = axs[1]
    ex = pd.read_csv(R / "exec_merged.csv") if (R / "exec_merged.csv").exists() else None
    if ex is not None and "BN" in per_task:
        e = ex[(ex.model_id == model) & (ex.condition == "DIRECT")]
        pv = e.groupby(["task_id", "variant"])["functional_pass"].mean().unstack()
        t = [k for k in per_task["BN"] if k in pv.index]
        drop = (pv.loc[t, "EN"] - pv.loc[t, "BN"]).values
        al = np.array([per_task["BN"][k] for k in t])
        rho, pval = sps.spearmanr(al, drop)
        ax.scatter(al, drop + np.random.default_rng(0).normal(0, 0.01, len(drop)), s=7, color=NPG["red"], alpha=0.6, linewidths=0)
        ax.set_xlabel("late-layer cosine, EN vs BN")
        ax.set_ylabel(r"per-task Pass@1 drop EN$-$BN")
        ax.set_ylim(-1.1, 1.45)
        ax.text(0.03, 0.95, f"Spearman $\\rho$={rho:.2f} (p={pval:.3f})", transform=ax.transAxes, fontsize=7, va="top")
        json.dump({"spearman_rho": float(rho), "p": float(pval), "n": int(len(t))}, open(R / "xai_alignment_corr.json", "w"))
    ax.text(-0.2, 1.04, "(d)", transform=ax.transAxes, fontweight="bold", fontsize=9)
    fig.subplots_adjust(wspace=0.35)
    save(fig, "fig_alignment", ROOT / "figures")
    pd.DataFrame(curves).to_csv(R / "xai_alignment_curves.csv", index_label="layer")


ALL = [fig_shap, fig_attribution, fig_segments, fig_alignment]
