#!/usr/bin/env python3
"""LaTeX tables for the interventional controls, the translation audit and the sanitised translate-first arms.
Reads data/results/{tab_causal,tab_translation_validity,tab_mitigation_clean,tab_mitigation_q3}.csv, writes paper/tab_*.tex."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from figstyle import MODEL_LABEL, MODEL_ORDER  # noqa: E402

R, P = ROOT / "data/results", ROOT / "paper"
BS = chr(92)
NL = chr(10)
VLAB = {"BN": "BN", "BN_MIX": "BN-mix", "BANGLISH": "Banglish", "ES": "ES"}


def tm(x, d=1):
    """signed number with a typographic minus"""
    return f"{x:+.{d}f}".replace("-", "$-$")


def ci(lo, hi):
    return f"[{tm(lo)}, {tm(hi)}]"


def stars(p):
    return "$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""


def rd(name):
    p = R / name
    return pd.read_csv(p) if p.exists() and p.stat().st_size > 5 else None


def models(df, col="model_id"):
    return [m for m in MODEL_ORDER if m in set(df[col])] + sorted(set(df[col]) - set(MODEL_ORDER))


def causal():
    t = rd("tab_causal.csv")
    if t is None:
        return
    rows = []
    for m in models(t):
        s = t[t.model_id == m].set_index("variant")
        if "EN_PAD" not in s.index:
            continue
        cells = [MODEL_LABEL.get(m, m), f"{s.loc['EN_PAD', 'en_pct']:.1f}"]
        for v in ("EN_PAD", "BN_DESC", "BN"):
            r = s.loc[v]
            cells.append(f"{tm(r.delta_pp)} {ci(r.lo, r.hi)}{stars(r.p)}")
        for v in ("HYB33", "HYB67", "BN_DESC_SUB"):
            if v in s.index:
                r = s.loc[v]
                cells.append(f"{tm(r.delta_pp)} {ci(r.lo, r.hi)}{stars(r.p)}")
            else:
                cells.append("--")
        rows.append(" & ".join(cells) + " " + BS + BS)
    head = ("\\begin{tabular}{@{}lrllllll@{}}" + NL + "\\toprule" + NL +
            "Model & EN & EN-pad & BN-desc & BN & Hybrid 1/3 & Hybrid 2/3 & BN-desc (hybrid tasks)" + BS + BS + NL + "\\midrule" + NL)
    (P / "tab_causal.tex").write_text((head + NL.join(rows) + NL + "\\bottomrule" + NL + "\\end{tabular}" + NL).replace("\\\\toprule", "\\toprule"),
                                      encoding="utf-8")


def translation():
    t = rd("tab_translation_validity.csv")
    if t is None:
        return
    rows = []
    for m in models(t, "model"):
        s = t[t.model == m].set_index("variant")
        cells = [MODEL_LABEL.get(m, m)]
        for v in VLAB:
            cells.append(f"{s.loc[v, 'valid']:.0f} / {s.loc[v, 'leaked']:.0f}" if v in s.index else "--")
        rows.append(" & ".join(cells) + " " + BS + BS)
    head = ("\\begin{tabular}{@{}lrrrr@{}}" + NL + "\\toprule" + NL + "Translating model & BN & BN-mix & Banglish & ES" + BS + BS + NL + "\\midrule" + NL)
    (P / "tab_translation.tex").write_text(head + NL.join(rows) + NL + "\\bottomrule" + NL + "\\end{tabular}" + NL, encoding="utf-8")


def mitigation():
    own, q3 = rd("tab_mitigation_clean.csv"), rd("tab_mitigation_q3.csv")
    if own is None and q3 is None:
        return
    base = pd.concat([x for x in (own, q3) if x is not None])
    rows = []
    for m in models(base):
        for v in VLAB:
            o = own[(own.model_id == m) & (own.variant == v)] if own is not None else pd.DataFrame()
            q = q3[(q3.model_id == m) & (q3.variant == v)] if q3 is not None else pd.DataFrame()
            ref = q if len(q) else o
            if not len(ref):
                continue
            cells = [MODEL_LABEL.get(m, m) if v == "BN" else "", VLAB[v], f"{100 * ref.pass_direct.iloc[0]:.1f}"]
            for x in (o, q):
                if len(x):
                    r = x.iloc[0]
                    cells.append(f"{100 * r.pass_translate:.1f} ({tm(100 * r.delta)}){stars(r.p_holm)}")
                else:
                    cells.append("--")
            cells.append(f"{q.token_ratio_median.iloc[0]:.2f}" if len(q) else "--")
            rows.append(" & ".join(cells) + " " + BS + BS)
        rows.append("\\midrule")
    if rows and rows[-1] == "\\midrule":
        rows.pop()
    head = ("\\begin{tabular}{@{}llrllr@{}}" + NL + "\\toprule" + NL +
            "Model & Variant & Direct & Own translation & Qwen3-8B translation & Tokens$\\times$" + BS + BS + NL + "\\midrule" + NL)
    (P / "tab_mit_clean.tex").write_text(head + NL.join(rows) + NL + "\\bottomrule" + NL + "\\end{tabular}" + NL, encoding="utf-8")


if __name__ == "__main__":
    causal()
    translation()
    mitigation()
    print("extra tables written")
