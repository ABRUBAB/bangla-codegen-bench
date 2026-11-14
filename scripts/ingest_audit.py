#!/usr/bin/env python3
"""Compute inter-annotator agreement from the two completed audit workbooks and write paper/audit_numbers.tex.

usage: python scripts/ingest_audit.py data/audit/audit_annotator1.xlsx data/audit/audit_annotator2.xlsx
If run, it defines \\auditdonetrue plus the statistics macros used by the paper; otherwise the paper keeps the
default \\auditdonefalse (see paper/audit_numbers.tex stub).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bcb.stats import cohens_kappa  # noqa: E402


def load(p):
    d = pd.read_excel(p, sheet_name="Audit")
    d["rendering"] = d["rendering"].str.split().str[0]
    return d.set_index(["task_id", "rendering"])


a, b = load(sys.argv[1]), load(sys.argv[2])
idx = a.index.intersection(b.index)
a, b = a.loc[idx], b.loc[idx]
ok = a["meaning_preserved"].notna() & b["meaning_preserved"].notna()
a, b = a[ok], b[ok]
n = len(a)
kappa3 = cohens_kappa(a["meaning_preserved"], b["meaning_preserved"])
bin_a = (a["meaning_preserved"] == "N").astype(int); bin_b = (b["meaning_preserved"] == "N").astype(int)
agree = float((a["meaning_preserved"] == b["meaning_preserved"]).mean())
nat_corr = float(np.corrcoef(a["naturalness"].astype(float), b["naturalness"].astype(float))[0, 1])
nat_mean = float(np.nanmean(np.r_[a["naturalness"].astype(float), b["naturalness"].astype(float)]))
major = float(((a["meaning_preserved"] == "N") | (b["meaning_preserved"] == "N")).mean())
minor = float(((a["meaning_preserved"] == "MINOR") | (b["meaning_preserved"] == "MINOR")).mean())
tex = [
    "\\auditdonetrue",
    f"\\newcommand{{\\auditN}}{{{n}}}",
    f"\\newcommand{{\\auditKappa}}{{{kappa3:.2f}}}",
    f"\\newcommand{{\\auditAgree}}{{{100 * agree:.1f}}}",
    f"\\newcommand{{\\auditNatMean}}{{{nat_mean:.2f}}}",
    f"\\newcommand{{\\auditNatCorr}}{{{nat_corr:.2f}}}",
    f"\\newcommand{{\\auditMajor}}{{{100 * major:.1f}}}",
    f"\\newcommand{{\\auditMinor}}{{{100 * minor:.1f}}}",
]
(ROOT / "paper/audit_numbers.tex").write_text("\n".join(tex) + "\n")
print("\n".join(tex))
for r in ("BN", "BN_MIX", "BANGLISH"):
    m = a.index.get_level_values("rendering") == r
    print(r, "major-issue rate:", float(((a[m]["meaning_preserved"] == "N") | (b[m]["meaning_preserved"] == "N")).mean()))
