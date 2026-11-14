#!/usr/bin/env python3
"""Generate all data-driven figures from data/results/*. Each function skips itself if its inputs are missing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from figstyle import (MODEL_LABEL, MODEL_ORDER, NPG, VARIANT_COLOR, VARIANT_LABEL, VARIANT_ORDER, save, setup)  # noqa

R = ROOT / "data/results"
F = ROOT / "figures"
TOK_LABEL = {"gpt-4 (cl100k)": "GPT-4 (cl100k)", "gpt-4o (o200k)": "GPT-4o (o200k)", "qwen2.5": "Qwen2.5 (151k)",
             "qwen3": "Qwen3 (151k)", "deepseek-coder": "DeepSeek-Coder (32k)", "deepseek-v2": "DeepSeek-V2 (100k)",
             "tigercoder": "TigerCoder (262k)", "llama3": "Llama-3.1 (128k)", "mistral": "Mistral-v0.3 (33k)",
             "qwen3.8": "Qwen3.8 (248k)", "gemma4": "Gemma-4 (262k)", "ministral3": "Ministral-3 (131k)", "glm4.7": "GLM-4.7 (155k)"}


def models_present(df):
    return [m for m in MODEL_ORDER if m in set(df["model_id"])]


# ----------------------------------------------------------------------------------------------- tokens
def fig_tokens():
    p = R / "token_counts.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    toks = [c[6:] for c in d.columns if c.startswith("desc__")]
    en = d[d.variant == "EN"].set_index("task_id")
    vs = ["EN_TYPO", "BN", "BN_SPELL", "BN_MIX", "BN_TYPO", "BANGLISH", "ES"]
    M = pd.DataFrame({TOK_LABEL.get(t, t): {v: float(np.median(d[d.variant == v].set_index("task_id")[f"desc__{t}"] /
                                           en[f"desc__{t}"])) for v in vs} for t in toks}).T
    M = M.sort_values("BN", ascending=False)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    im = ax.imshow(np.log2(M.values), cmap="YlOrRd", vmin=0, vmax=np.log2(10), aspect="auto")
    ax.set_xticks(range(len(vs))); ax.set_xticklabels([VARIANT_LABEL[v] for v in vs], rotation=35, ha="right")
    ax.set_yticks(range(len(M))); ax.set_yticklabels(M.index)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if np.log2(v) > 2.0 else "black")
    ax.text(-0.18, 1.02, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_ticks(np.log2([1, 2, 4, 8])); cb.set_ticklabels(["1x", "2x", "4x", "8x"]); cb.ax.tick_params(labelsize=7)
    cb.set_label("median token ratio vs EN", fontsize=7.5)
    # panel (b): per-task distribution for BN across tokenizers
    ax = axs[1]
    order = list(M.index)
    inv = {v: k for k, v in TOK_LABEL.items()}
    data = [(d[d.variant == "BN"].set_index("task_id")[f"desc__{inv.get(o, o)}"] / en[f"desc__{inv.get(o, o)}"]).values for o in order]
    bp = ax.boxplot(data[::-1], vert=False, widths=0.6, patch_artist=True, showfliers=False,
                    medianprops={"color": "black", "linewidth": 1.0})
    for b in bp["boxes"]:
        b.set_facecolor(NPG["red"]); b.set_alpha(0.75); b.set_linewidth(0.6)
    ax.set_yticks(range(1, len(order) + 1)); ax.set_yticklabels(order[::-1], fontsize=6.5)
    ax.axvline(1, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("BN / EN token ratio (per task)")
    ax.text(-0.35, 1.02, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=9)
    fig.subplots_adjust(wspace=0.95)
    save(fig, "fig_tokens", F)
    M.round(3).to_csv(R / "tab_token_ratio_median.csv")


if __name__ == "__main__":
    setup()
    import figs_results, figs_xai
    for fn in [fig_tokens] + figs_results.ALL + figs_xai.ALL:
        try:
            fn()
        except Exception as exc:  # keep going; report
            print("FAILED", fn.__name__, type(exc).__name__, exc)
