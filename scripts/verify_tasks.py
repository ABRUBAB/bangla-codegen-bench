#!/usr/bin/env python3
"""Verify task files: schema + linguistics + oracle sanity inside the sandbox.

Checks per task
  * schema / script / identifier-preservation rules (bcb.tasks.validate_task)
  * functional tasks : reference_solution passes functional_tests   (3 repeats: flakiness)
  * security tasks   : reference_solution passes functional AND exploit tests (secure)
                       insecure_reference passes functional but FAILS exploit tests (vulnerable)

Usage: python scripts/verify_tasks.py data/tasks/security_A.jsonl [more.jsonl ...] [--no-sandbox]
Exit status is non-zero if any task has a problem.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bcb.execute import run_jobs  # noqa: E402
from bcb.tasks import build_jobs, load_tasks, validate_task  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--no-sandbox", action="store_true", help="schema checks only")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    tasks = load_tasks([Path(f) for f in args.files])
    problems: dict[str, list[str]] = {t["task_id"]: validate_task(t) for t in tasks}
    ids = [t["task_id"] for t in tasks]
    dup = {i for i in ids if ids.count(i) > 1}
    for d in dup:
        problems[d].append("duplicate task_id")

    if not args.no_sandbox:
        jobs = []
        for t in tasks:
            for r in range(args.repeats):
                jobs.append(build_jobs(t, t["reference_solution"], f"{t['task_id']}|ref|{r}"))
            if t["kind"] == "security":
                for r in range(args.repeats):
                    jobs.append(build_jobs(t, t["insecure_reference"], f"{t['task_id']}|insec|{r}"))
        res = run_jobs(jobs, workers=args.workers)
        for t in tasks:
            tid = t["task_id"]
            for r in range(args.repeats):
                rr = res[f"{tid}|ref|{r}"]
                f = rr["results"].get("functional", {})
                if not f.get("passed"):
                    problems[tid].append(f"reference FAILS functional tests (run {r}): "
                                         f"{(f.get('tail') or rr.get('harness_error') or '')[-300:]}")
                    break
                if t["kind"] == "security":
                    e = rr["results"].get("exploit", {})
                    if not e.get("passed"):
                        problems[tid].append(f"SECURE reference FAILS exploit test (run {r}): {(e.get('tail') or '')[-300:]}")
                        break
            if t["kind"] == "security":
                outcomes = []
                for r in range(args.repeats):
                    rr = res[f"{tid}|insec|{r}"]
                    f = rr["results"].get("functional", {})
                    e = rr["results"].get("exploit", {})
                    outcomes.append((bool(f.get("passed")), bool(e.get("passed"))))
                    if r == 0 and not f.get("passed"):
                        problems[tid].append(f"INSECURE reference fails functional tests: {(f.get('tail') or '')[-300:]}")
                if any(o[1] for o in outcomes):
                    problems[tid].append("INSECURE reference PASSES the exploit test (oracle too weak / flaky)")

    bad = {k: v for k, v in problems.items() if v}
    for tid in ids:
        if tid in bad:
            print(f"[FAIL] {tid}")
            for p in bad[tid]:
                print("       -", p)
        else:
            print(f"[ ok ] {tid}")
    print(f"\n{len(ids) - len(bad)}/{len(ids)} tasks OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
