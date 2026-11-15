#!/usr/bin/env python3
"""Print the value of \\N{key} macros from paper/numbers.tex (development helper).
usage: python scripts/show_macro.py pgpf-BN-OR df-qA-BN ...      (a trailing * matches a prefix)"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
text = (ROOT / "paper/numbers.tex").read_text(encoding="utf-8")
vals = dict(re.findall(r"csname n@(.*?)\\endcsname\{(.*?)\}", text))
for k in sys.argv[1:]:
    if k.endswith("*"):
        for kk, v in vals.items():
            if kk.startswith(k[:-1]):
                print(f"{kk} = {v}")
    else:
        print(f"{k} = {vals.get(k)}")
