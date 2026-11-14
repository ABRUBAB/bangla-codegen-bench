#!/usr/bin/env python3
"""Merge second-wave tasks (data/tasks/_new/*.jsonl) into functional.jsonl / security.jsonl.
Drops tasks listed in --drop (duplicates of an earlier task), validates everything, keeps ids sorted."""
import argparse, glob, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bcb.tasks import validate_task

ap = argparse.ArgumentParser(); ap.add_argument("--drop", default="")
a = ap.parse_args()
drop = set(filter(None, a.drop.split(",")))
ROOT = Path(__file__).resolve().parent.parent
new = {}
for f in sorted(glob.glob(str(ROOT / "data/tasks/_new/*.jsonl"))):
    for l in open(f, encoding="utf-8"):
        if l.strip():
            t = json.loads(l)
            if t["task_id"] in drop:
                continue
            new[t["task_id"]] = t
bad = {k: validate_task(t) for k, t in new.items() if validate_task(t)}
if bad:
    print("INVALID:", bad); sys.exit(1)
for name, prefix in (("functional", "F"), ("security", "S")):
    p = ROOT / f"data/tasks/{name}.jsonl"
    rows = {json.loads(l)["task_id"]: json.loads(l) for l in open(p, encoding="utf-8") if l.strip()}
    added = {k: v for k, v in new.items() if k.startswith(prefix)}
    overlap = set(rows) & set(added)
    assert not overlap, overlap
    rows.update(added)
    tmp = str(p) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for k in sorted(rows):
            f.write(json.dumps(rows[k], ensure_ascii=False) + "\n")
    os.replace(tmp, p)
    print(name, "now", len(rows), "tasks (+", len(added), ")")
