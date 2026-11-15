#!/usr/bin/env python3
"""Sanitise the models' own 'translations to English' and measure how often they were not translations at all.

Finding that motivates this step: when a code model is asked to translate a programming task, it often *solves* the task inside the
translation (a code block with an implemented body) or answers instead of translating. A downstream model then copies that solution, so an
unsanitised translate-first arm measures solution leakage, not translation quality.

For every data/generations/<model>/translations.jsonl this script
  * classifies each translation:  valid (prose covers >= 50 % of the content words of the English reference description),
                                  nodesc (the model answered/commented instead of translating),
                                  untranslated (prose still mostly Bengali script);
  * flags `leaked` when a code block contains more than the signature (a body, not just a docstring/ellipsis);
  * writes translations_clean.jsonl in which the first code block is replaced by the canonical signature block and further blocks are
    dropped, so that only prose survives; token counts of the original translation call are kept.
Outputs: data/results/tab_translation_validity.csv, paper/translation_numbers.tex (macros \\TV{model}{variant}{field}) and
data/generations/<model>/translations_clean.jsonl."""
import glob
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bcb.tasks import TASK_DIR, load_tasks  # noqa: E402

NL = chr(10)
FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S | re.I)
UNTERM = re.compile(r"```(?:python|py)?[ \t]*\n(.*)$", re.S | re.I)
BENGALI = re.compile("[" + chr(0x0980) + "-" + chr(0x09FF) + "]")
LETTERS = re.compile("[A-Za-z" + chr(0x0980) + "-" + chr(0x09FF) + "]")
KEY = {"qwen2.5-coder-1.5b": "A", "qwen2.5-coder-3b": "B", "qwen2.5-coder-7b": "C", "deepseek-coder-6.7b": "D",
       "qwen3-8b": "E", "tigercoder-1b": "F", "gemini-3.1-flash-lite": "G", "qwen2.5-coder-14b": "H", "qwen2.5-coder-32b": "I"}
STOP = set("the a an of to in is are be for and or with that this it as by on at from if its not no only each than then "
           "they their into which will have has".split())


def has_body(block: str) -> bool:
    lines = [ln for ln in block.strip().split(NL) if ln.strip()]
    body = [ln for ln in lines[1:] if not ln.strip().startswith(('"""', "'''", "#", "..."))
            and ln.strip() not in ("pass", '"""', "'''")]
    return any(re.search(r"\breturn\b|\bfor\b|\bif\b|\bwhile\b|=|\bimport\b|\braise\b|\byield\b|\bwith\b", ln) for ln in body)


def clean(text: str, sig: str) -> str:
    """Replace the first code block by the canonical signature block, drop the others, keep the prose."""
    canon = "```python" + NL + sig.strip() + NL + "```"
    parts = []

    def mark(_):
        parts.append(1)
        return "@@BLOCK" + str(len(parts) - 1) + "@@"
    t = FENCE.sub(mark, text)
    t = UNTERM.sub(mark, t)  # an unterminated trailing fence counts as a block too
    if not parts:
        t = t.rstrip() + NL + NL + "@@BLOCK0@@"
        parts = [1]
    t = t.replace("@@BLOCK0@@", canon)
    for i in range(1, len(parts)):
        t = t.replace("@@BLOCK" + str(i) + "@@", "")
    return re.sub(r"\n{3,}", NL + NL, t).strip()


def words(t: str):
    return {w for w in re.findall(r"[a-z]{4,}", t.lower()) if w not in STOP}


def classify(text: str, sig_reinjected: bool, reference_en: str) -> str:
    """valid: the prose contains the task description; nodesc: the model answered instead of translating;
    untranslated: prose still mostly Bengali script."""
    t = text.rsplit("```python", 1)[0] if sig_reinjected else text
    prose = UNTERM.sub(" ", FENCE.sub(" ", t))
    letters = len(LETTERS.findall(prose)) or 1
    if len(BENGALI.findall(prose)) / letters >= 0.05:
        return "untranslated"
    ref = words(re.sub(r"```.*?```", " ", reference_en, flags=re.S))
    recall = len(ref & words(prose)) / max(1, len(ref))
    return "valid" if recall >= 0.5 else "nodesc"


def main():
    tasks = {t["task_id"]: t for t in load_tasks([p for p in TASK_DIR.glob("*.jsonl") if not p.name.startswith("_")])}
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data/generations/*/translations.jsonl"))):
        model = Path(f).parent.name
        out = []
        for line in Path(f).read_text(encoding="utf-8").split(NL):
            if not line.strip():
                continue
            r = json.loads(line)
            sig = tasks[r["task_id"]]["signature"]
            blocks = FENCE.findall(r["translation"]) + [m for m in UNTERM.findall(r["translation"]) if "```" not in m]
            leaked = any(has_body(b) for b in blocks)
            cls = classify(r["translation"], r["sig_reinjected"], tasks[r["task_id"]]["description"]["en"])
            rows.append({"model": model, "variant": r["variant"], "cls": cls, "leaked": leaked})
            out.append(dict(r, translation=clean(r["translation"], sig), leaked=leaked, cls=cls))
        Path(f).with_name("translations_clean.jsonl").write_text(
            NL.join(json.dumps(x, ensure_ascii=False) for x in out) + NL, encoding="utf-8")
    df = pd.DataFrame(rows)
    tab = (df.groupby(["model", "variant"]).cls.value_counts(normalize=True).unstack(fill_value=0) * 100)
    for c in ("valid", "nodesc", "untranslated"):
        if c not in tab:
            tab[c] = 0.0
    tab["leaked"] = df.groupby(["model", "variant"]).leaked.mean() * 100
    tab = tab.round(1)
    tab.to_csv(ROOT / "data/results/tab_translation_validity.csv")
    bs = chr(92)
    mac = ["% auto-generated by scripts/clean_translations.py - do not edit",
           bs + "newcommand{" + bs + "TV}[3]{" + bs + "csname tv@#1@#2@#3" + bs + "endcsname}"]
    for (m, v), r in tab.iterrows():
        k = KEY.get(m, re.sub(r"[^A-Za-z]", "", m))
        for fld in ("valid", "nodesc", "untranslated", "leaked"):
            mac.append(bs + "expandafter" + bs + "def" + bs + "csname tv@%s@%s@%s" % (k, v.replace("_", ""), fld) + bs + "endcsname{%.0f}" % r[fld])
    (ROOT / "paper/translation_numbers.tex").write_text(NL.join(mac) + NL, encoding="utf-8")
    print(tab[["valid", "nodesc", "untranslated", "leaked"]].to_string())


if __name__ == "__main__":
    main()
