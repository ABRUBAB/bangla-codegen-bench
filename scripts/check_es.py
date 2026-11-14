#!/usr/bin/env python3
"""Validate Spanish renderings: python scripts/check_es.py data/tasks/_es/es_X.json
The JSON maps task_id -> Spanish description (string). Each is merged into the existing task and validated with the
same hard rules as the other renderings (no Bangla script, identifiers and numerals preserved)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bcb.tasks import load_tasks, TASK_DIR, validate_task

tasks = {t["task_id"]: t for t in load_tasks([TASK_DIR / "functional.jsonl", TASK_DIR / "security.jsonl"])}
bad = 0
for p in sys.argv[1:]:
    es = json.loads(Path(p).read_text(encoding="utf-8"))
    for tid, text in es.items():
        t = json.loads(json.dumps(tasks[tid]))
        t["description"]["es"] = text
        errs = [e for e in validate_task(t) if e.startswith(("es", "ES"))]
        if not text.strip():
            errs.append("empty")
        if errs:
            bad += 1
            print("[FAIL]", tid, errs)
    print(p, len(es), "tasks checked")
print("problems:", bad)
sys.exit(1 if bad else 0)
