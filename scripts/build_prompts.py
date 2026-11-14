#!/usr/bin/env python3
"""Build data/prompts/prompts.jsonl from data/tasks/*.jsonl (all variants, controls, ablations)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bcb.prompts import build_prompts
from bcb.tasks import load_tasks, TASK_DIR, validate_task

def main():
    files = [p for p in sorted(TASK_DIR.glob("*.jsonl")) if not p.name.startswith("_")]
    tasks = load_tasks(files)
    bad = {t["task_id"]: validate_task(t) for t in tasks if validate_task(t)}
    if bad:
        print("schema problems:", bad); sys.exit(1)
    out = Path(__file__).resolve().parent.parent / "data" / "prompts" / "prompts.jsonl"
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for t in tasks:
            for p in build_prompts(t):
                p["signature"] = t["signature"]
                p["function_name"] = t["function_name"]
                f.write(json.dumps(p, ensure_ascii=False) + "\n"); n += 1
    print(f"{len(tasks)} tasks -> {n} prompts -> {out}")

if __name__ == "__main__":
    main()
