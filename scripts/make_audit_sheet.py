#!/usr/bin/env python3
"""Create the native-speaker audit workbooks (one per annotator) for the Bangla renderings.

Sample: 60 tasks (40 functional + 20 security), stratified random (seed 42) x 3 renderings
(BN, BN_MIX, BANGLISH) = 180 items. Each annotator gets the same items in a different random order and
fills dropdown columns independently; scripts/ingest_audit.py then computes agreement and writes
paper/audit_numbers.tex. Annotators should NOT see each other's sheets.
"""
import json
import random
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parent.parent
tasks = [json.loads(l) for f in ("functional.jsonl", "security.jsonl")
         for l in (ROOT / "data/tasks" / f).read_text(encoding="utf-8").splitlines() if l.strip()]
rng = random.Random(42)
func = [t for t in tasks if t["kind"] == "functional"]
sec = [t for t in tasks if t["kind"] == "security"]
sample = rng.sample(func, 40) + rng.sample(sec, 20)
items = [(t, r) for t in sample for r in ("bn", "bn_mix", "banglish")]
LABEL = {"bn": "BN (formal Bangla)", "bn_mix": "BN_MIX (code-mixed)", "banglish": "BANGLISH (romanized)"}

INSTR = [
    "BanglaCodeBench - native-speaker audit of Bangla renderings",
    "",
    "You see an ENGLISH task description and one RENDERING of it (formal Bangla / code-mixed Bangla / Banglish).",
    "Judge ONLY the rendering against the English text. Do the audit alone, without discussing items with the other annotator.",
    "",
    "Column meaning_preserved:  Y = same requirements (nothing added/removed/changed);  MINOR = small wording difference that",
    "  does not change what a programmer must implement;  N = a requirement is missing, added or changed (numbers, literals,",
    "  order, edge cases, return value, constraints).",
    "Column naturalness (1-5): 1 = clearly machine-translated / unnatural for a Bangladeshi developer; 5 = fully natural for that register.",
    "Column register_ok: Y if the text really matches the register (BN = formal Bangla; BN_MIX = Bangla grammar with English technical words;",
    "  BANGLISH = Bangla in Latin letters, chat-style), else N.",
    "Column suggested_fix: write your corrected version if meaning_preserved is MINOR/N or naturalness <= 3 (optional otherwise).",
    "Save the file and send it back. Identifiers in `backticks` must stay English - that is intentional.",
]


def build(path: Path, annotator: str, seed: int):
    wb = Workbook()
    ws0 = wb.active
    ws0.title = "Instructions"
    for i, line in enumerate(INSTR, 1):
        ws0.cell(i, 1, line).font = Font(bold=(i == 1), size=13 if i == 1 else 11)
    ws0.column_dimensions["A"].width = 140
    ws = wb.create_sheet("Audit")
    head = ["item_id", "task_id", "rendering", "ENGLISH (original)", "RENDERING (to judge)", "meaning_preserved",
            "naturalness", "register_ok", "suggested_fix", "comment", "annotator"]
    ws.append(head)
    order = items[:]
    random.Random(seed).shuffle(order)
    for n, (t, r) in enumerate(order, 1):
        ws.append([f"A{n:03d}", t["task_id"], LABEL[r], t["description"]["en"], t["description"][r],
                   None, None, None, None, None, annotator])
    widths = [8, 8, 20, 60, 60, 16, 12, 12, 50, 30, 12]
    for i, w in enumerate(widths):
        ws.column_dimensions[chr(65 + i)].width = w
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF"); c.fill = PatternFill("solid", fgColor="3C5488")
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
        for c in row[5:10]:
            c.fill = PatternFill("solid", fgColor="FFF8E1")
    n = len(order) + 1
    dv1 = DataValidation(type="list", formula1='"Y,MINOR,N"', allow_blank=True)
    dv2 = DataValidation(type="list", formula1='"1,2,3,4,5"', allow_blank=True)
    dv3 = DataValidation(type="list", formula1='"Y,N"', allow_blank=True)
    for dv, col in ((dv1, "F"), (dv2, "G"), (dv3, "H")):
        ws.add_data_validation(dv); dv.add(f"{col}2:{col}{n}")
    ws.freeze_panes = "D2"
    wb.save(path)


out = ROOT / "data/audit"
out.mkdir(parents=True, exist_ok=True)
build(out / "audit_annotator1.xlsx", "annotator1", 101)
build(out / "audit_annotator2.xlsx", "annotator2", 202)
(out / "audit_sample.json").write_text(json.dumps([{"task_id": t["task_id"], "rendering": r} for t, r in items]))
print(f"wrote {len(items)} items x 2 workbooks to {out}")
