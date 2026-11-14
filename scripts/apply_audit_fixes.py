#!/usr/bin/env python3
"""Apply the corrected renderings proposed by the AI audit (fix field), validating every change.

usage: python scripts/apply_audit_fixes.py data/audit/llm_opus_NEW_*.json --out data/audit/applied_fixes_wave2.json
A fix is applied only if the full task (incl. identifier/numeral/script rules) still validates; otherwise it is logged
as skipped. The change log keeps old and new text so that every edit is traceable."""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bcb.tasks import validate_task  # noqa: E402

NL = chr(10)
ap = argparse.ArgumentParser()
ap.add_argument("audits", nargs="+")
ap.add_argument("--out", required=True)
a = ap.parse_args()
items = []
for pat in a.audits:
    for f in sorted(glob.glob(pat)):
        items += json.loads(Path(f).read_text(encoding="utf-8"))
files = {"functional": ROOT / "data/tasks/functional.jsonl", "security": ROOT / "data/tasks/security.jsonl"}
rows = {k: [json.loads(l) for l in p.read_text(encoding="utf-8").split(NL) if l.strip()] for k, p in files.items()}
idx = {t["task_id"]: (k, i) for k, rs in rows.items() for i, t in enumerate(rs)}
applied, skipped = [], []
for it in items:
    fix = (it.get("fix") or "").strip()
    if not fix:
        continue
    k, i = idx[it["task_id"]]
    new = json.loads(json.dumps(rows[k][i]))
    old = new["description"][it["rendering"]]
    new["description"][it["rendering"]] = fix
    errs = validate_task(new)
    if errs:
        skipped.append({"task_id": it["task_id"], "rendering": it["rendering"], "errors": errs})
        continue
    rows[k][i] = new
    applied.append({"task_id": it["task_id"], "rendering": it["rendering"], "old": old, "new": fix, "meaning": it["meaning"],
                    "naturalness": it["naturalness"], "issues": it["issues"]})
for k, p in files.items():
    tmp = str(p) + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline=NL) as f:
        for r in rows[k]:
            f.write(json.dumps(r, ensure_ascii=False) + NL)
    os.replace(tmp, p)
Path(a.out).write_text(json.dumps({"applied": applied, "skipped": skipped}, ensure_ascii=False, indent=1), encoding="utf-8")
print("applied", len(applied), "skipped", len(skipped))
for s in skipped:
    print(" skipped", s)
