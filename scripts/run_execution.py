#!/usr/bin/env python3
"""Execute generated code in the sandbox: functional tests (all tasks) + exploit tests (security tasks).

usage: python scripts/run_execution.py --gens data/generations --out data/results/exec.jsonl [--models a,b]
Reads <gens>/<model>/gens_*.jsonl (dedupe by gen_id), writes one JSON line per generation:
  {gen_id, model_id, task_id, variant, condition, sample_idx, kind, status, functional_pass, exploit_pass,
   func_failure_type, exploit_failure_type, func_tail, exploit_tail, duration}
status: ok | no_code | harness_error.   exploit_pass is None when not applicable/not run.

Identical code for the same task gives an identical verdict (the tests are fixed and deterministic), so every distinct
(task, code) pair is executed once and its verdict is copied to all samples with that code (rows carry "reused": true when
the verdict came from another sample). Results are appended after every batch of jobs, so an interrupted run loses at most
one batch and can simply be restarted.
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


def verdict(r: dict) -> dict:
    """Verdict fields of one sandbox result (as returned by bcb.execute.run_jobs)."""
    f = r.get("results", {}).get("functional")
    e = r.get("results", {}).get("exploit")
    return {"status": "harness_error" if (f is None or r.get("harness_error")) else "ok",
            "functional_pass": bool(f and f["passed"]),
            "exploit_pass": (bool(e["passed"]) if e is not None else None),
            "func_failure_type": f.get("failure_type") if f else "harness_error",
            "exploit_failure_type": e.get("failure_type") if e else None,
            "func_tail": (f or {}).get("tail", "")[-700:], "exploit_tail": (e or {}).get("tail", "")[-500:],
            "compile_error": r.get("compile_error"), "duration": (f or {}).get("duration")}


VERDICT_KEYS = ("status", "functional_pass", "exploit_pass", "func_failure_type", "exploit_failure_type", "func_tail",
                "exploit_tail", "compile_error", "duration")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", default="data/generations")
    ap.add_argument("--out", default="data/results/exec.jsonl")
    ap.add_argument("--models", default="")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--shard", type=int, default=40)
    ap.add_argument("--batch", type=int, default=1500, help="distinct (task, code) jobs per batch; results are written after each batch")
    ap.add_argument("--no-reuse", action="store_true", help="execute every sample even if identical code was already executed")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--exclude-tasks", default="", help="comma list of task ids to skip (pending a regeneration patch)")
    a = ap.parse_args()
    tasks = {t["task_id"]: t for t in load_tasks([p for p in TASK_DIR.glob("*.jsonl") if not p.name.startswith("_")])}
    gens = load_gens(Path(a.gens), set(filter(None, a.models.split(","))))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done, cache = {}, {}
    if out.exists():
        for l in out.read_text(encoding="utf-8").split(chr(10)):
            if l.strip():
                r = json.loads(l)
                if r.get("status") == "harness_error":
                    continue  # infrastructure failure, not a verdict: the sample is executed again
                done[r["gen_id"]] = r.get("code_sha")
                if r.get("status") == "ok":
                    cache[(r["task_id"], r.get("code_sha"))] = {k: r.get(k) for k in VERDICT_KEYS}
    sha = lambda g: hashlib.sha1((g.get("extracted_code") or "").encode("utf-8")).hexdigest()[:12]
    excl = set(filter(None, a.exclude_tasks.split(",")))
    # re-run a sample when it is new or its code changed since it was executed (regenerated prompts keep their gen_id)
    todo = [g for g in gens.values() if done.get(g["gen_id"], "MISSING") != sha(g) and g["task_id"] not in excl]
    print(f"{len(gens)} generations, {len(done)} already executed, {len(todo)} to run", flush=True)

    def row(g, extra):
        t = tasks[g["task_id"]]
        d = {"gen_id": g["gen_id"], "model_id": g["model_id"], "task_id": g["task_id"], "variant": g["variant"],
             "condition": g["condition"], "sample_idx": g["sample_idx"], "kind": t["kind"], "code_sha": sha(g)}
        d.update(extra)
        return json.dumps(d, ensure_ascii=False) + "\n"

    groups, n_nocode, n_reused, n_fail = {}, 0, 0, 0
    with open(out, "a", encoding="utf-8") as fh:
        for g in todo:
            if not (g.get("extracted_code") or "").strip():
                fh.write(row(g, {"status": "no_code", "functional_pass": False, "exploit_pass": None,
                                 "func_failure_type": "no_code", "exploit_failure_type": None}))
                n_nocode += 1
                continue
            key = (g["task_id"], sha(g))
            if not a.no_reuse and key in cache:
                fh.write(row(g, dict(cache[key], reused=True)))
                n_reused += 1
            else:
                groups.setdefault(key, []).append(g)
        fh.flush()
        print(f"{n_nocode} samples without code, {n_reused} reuse the verdict of identical code, "
              f"{len(groups)} distinct (task, code) pairs to execute", flush=True)
        keys = list(groups)
        n_batches = max(1, (len(keys) + a.batch - 1) // a.batch)
        for b in range(n_batches):
            chunk = keys[b * a.batch:(b + 1) * a.batch]
            jobs = []
            for key in chunk:
                rep = groups[key][0]
                jobs.append(build_jobs(tasks[rep["task_id"]], rep["extracted_code"], rep["gen_id"], timeout=a.timeout))
            res = run_jobs(jobs, workers=a.workers, shard_size=a.shard)
            n_fail_batch = 0
            for key in chunk:
                grp = groups[key]
                v = verdict(res.get(grp[0]["gen_id"], {}))
                if v["status"] == "harness_error":  # never store infrastructure failures as verdicts
                    n_fail_batch += len(grp)
                    continue
                for i, g in enumerate(grp):
                    fh.write(row(g, v if i == 0 else dict(v, reused=True)))
            fh.flush()
            n_fail += n_fail_batch
            print(f"[exec] batch {b + 1}/{n_batches} written ({len(chunk)} distinct codes, {n_fail_batch} samples failed in the harness)", flush=True)
    print("done ->", out)
    if n_fail:
        print(f"NOT EXECUTED: {n_fail} samples hit a sandbox failure and were not stored; run this script again to retry them", flush=True)
        sys.exit(4)


if __name__ == "__main__":
    main()
