#!/usr/bin/env python3
"""Execute generated code in the sandbox: functional tests (all tasks) + exploit tests (security tasks).

usage: python scripts/run_execution.py --gens data/generations --out data/results/exec.jsonl [--models a,b]
Reads <gens>/<model>/gens_*.jsonl (dedupe by gen_id), writes one JSON line per generation:
  {gen_id, model_id, task_id, variant, condition, sample_idx, kind, status, functional_pass, exploit_pass,
   func_failure_type, exploit_failure_type, func_tail, exploit_tail, duration}
status: ok | no_code | harness_error.   exploit_pass is None when not applicable/not run.
"""
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bcb.execute import run_jobs
from bcb.tasks import load_tasks, TASK_DIR, build_jobs

def load_gens(root: Path, models):
    gens = {}
    for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
        if models and mdir.name not in models:
            continue
        for f in sorted(mdir.glob("gens_*.jsonl")):
            for line in f.read_text(encoding="utf-8").split("\n"):
                if line.strip():
                    r = json.loads(line)
                    if r.get("finish") == "ERROR":
                        continue
                    gens[r["gen_id"]] = r
    return gens

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", default="data/generations")
    ap.add_argument("--out", default="data/results/exec.jsonl")
    ap.add_argument("--models", default="")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--shard", type=int, default=40)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--exclude-tasks", default="", help="comma list of task ids to skip (pending a regeneration patch)")
    a = ap.parse_args()
    tasks = {t["task_id"]: t for t in load_tasks([p for p in TASK_DIR.glob("*.jsonl") if not p.name.startswith("_")])}
    gens = load_gens(Path(a.gens), set(filter(None, a.models.split(","))))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():
        for l in out.read_text(encoding="utf-8").split(chr(10)):
            if l.strip():
                r = json.loads(l)
                done[r["gen_id"]] = r.get("code_sha")
    sha = lambda g: hashlib.sha1((g.get("extracted_code") or "").encode("utf-8")).hexdigest()[:12]
    excl = set(filter(None, a.exclude_tasks.split(",")))
    # re-run a sample when it is new or its code changed since it was executed (regenerated prompts keep their gen_id)
    todo = [g for g in gens.values() if done.get(g["gen_id"], "MISSING") != sha(g) and g["task_id"] not in excl]
    print(f"{len(gens)} generations, {len(done)} already executed, {len(todo)} to run", flush=True)
    jobs, skipped = [], []
    for g in todo:
        t = tasks[g["task_id"]]
        if not g.get("extracted_code", "").strip():
            skipped.append(g); continue
        jobs.append(build_jobs(t, g["extracted_code"], g["gen_id"], timeout=a.timeout))
    res = run_jobs(jobs, workers=a.workers, shard_size=a.shard)
    with open(out, "a", encoding="utf-8") as fh:
        for g in skipped:
            t = tasks[g["task_id"]]
            fh.write(json.dumps({"gen_id": g["gen_id"], "model_id": g["model_id"], "task_id": g["task_id"],
                "variant": g["variant"], "condition": g["condition"], "sample_idx": g["sample_idx"], "kind": t["kind"],
                "code_sha": sha(g), "status": "no_code", "functional_pass": False, "exploit_pass": None,
                "func_failure_type": "no_code", "exploit_failure_type": None}) + "\n")
        for g in todo:
            if g in skipped: continue
            t = tasks[g["task_id"]]; r = res.get(g["gen_id"], {})
            f = r.get("results", {}).get("functional"); e = r.get("results", {}).get("exploit")
            fh.write(json.dumps({"gen_id": g["gen_id"], "model_id": g["model_id"], "task_id": g["task_id"],
                "variant": g["variant"], "condition": g["condition"], "sample_idx": g["sample_idx"], "kind": t["kind"],
                "code_sha": sha(g), "status": "harness_error" if (f is None or r.get("harness_error")) else "ok",
                "functional_pass": bool(f and f["passed"]),
                "exploit_pass": (bool(e["passed"]) if e is not None else None),
                "func_failure_type": f.get("failure_type") if f else "harness_error",
                "exploit_failure_type": e.get("failure_type") if e else None,
                "func_tail": (f or {}).get("tail", "")[-700:], "exploit_tail": (e or {}).get("tail", "")[-500:],
                "compile_error": r.get("compile_error"), "duration": (f or {}).get("duration")}, ensure_ascii=False) + "\n")
    print("done ->", out)

if __name__ == "__main__":
    main()
