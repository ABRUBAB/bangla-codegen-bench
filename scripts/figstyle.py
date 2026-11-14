"""Shared plotting style for all paper figures (no in-canvas titles; titles live in LaTeX captions)."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

NPG = {"red": "#E64B35", "cyan": "#4DBBD5", "teal": "#00A087", "navy": "#3C5488", "coral": "#F39B7F",
       "slate": "#8491B4", "mint": "#91D1C2", "crimson": "#DC0000", "earth": "#7E6148", "purple": "#7E6BA8",
       "grey": "#6B7280"}
VARIANT_ORDER = ["EN", "EN_TYPO", "BN", "BN_SPELL", "BN_MIX", "BN_TYPO", "BANGLISH", "ES", "ES_TYPO"]
VARIANT_COLOR = {"EN": NPG["navy"], "EN_TYPO": NPG["slate"], "BN": NPG["red"], "BN_SPELL": NPG["coral"],
                 "BN_MIX": NPG["teal"], "BN_TYPO": NPG["earth"], "BANGLISH": NPG["purple"], "ES": NPG["cyan"], "ES_TYPO": NPG["mint"], "ID_ONLY": NPG["grey"]}
VARIANT_LABEL = {"EN": "EN", "EN_TYPO": "EN-typo", "BN": "BN", "BN_SPELL": "BN-spell", "BN_MIX": "BN-mix",
                 "BN_TYPO": "BN-typo", "BANGLISH": "Banglish", "ES": "ES", "ES_TYPO": "ES-typo", "ID_ONLY": "ID-only"}
MODEL_LABEL = {"qwen2.5-coder-1.5b": "Qwen2.5-Coder 1.5B", "qwen2.5-coder-3b": "Qwen2.5-Coder 3B",
               "qwen2.5-coder-7b": "Qwen2.5-Coder 7B", "deepseek-coder-6.7b": "DeepSeek-Coder 6.7B",
               "qwen3-8b": "Qwen3 8B", "tigercoder-1b": "TigerCoder 1B", "tigercoder-9b": "TigerCoder 9B",
               "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite"}
MODEL_ORDER = ["qwen2.5-coder-1.5b", "tigercoder-1b", "qwen2.5-coder-3b", "deepseek-coder-6.7b", "qwen2.5-coder-7b",
               "qwen3-8b", "tigercoder-9b", "gemini-3.1-flash-lite"]


def setup():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7, "figure.dpi": 150, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.grid": False, "legend.frameon": False,
    })


def save(fig, name, outdir="figures"):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("saved", name)
