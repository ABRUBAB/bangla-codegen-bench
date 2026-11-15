#!/usr/bin/env python3
"""Unattended driver for the sanitised translate-first generation arms on Kaggle.

For each job (account, model, condition, translations file) it builds the kernel (scripts/make_kaggle_kernel.py), pushes it when the
account has a free GPU slot, polls until the kernel finishes and collects the result into data/generations/<model>/
(scripts/collect_kernel.py). One status line per event is printed (run with `nohup python -u scripts/tf_driver.py > log &`).
Authorised accounts only; tokens are read by scripts/kaggle_cli.py and never printed."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
Q3 = "data/generations/qwen3-8b/translations_clean.jsonl"


def own(m):
    return f"data/generations/{m}/translations_clean.jsonl"


# account -> (parallel GPU slots available to this driver, job list: (model, condition, tag, translations file))
PLAN = {
    "abdullahrubab1": (2, [("qwen2.5-coder-7b", "TF_CLEAN_Q3", "-tfq2", Q3), ("qwen2.5-coder-7b", "TF_CLEAN", "-tfc2", own("qwen2.5-coder-7b")),
                           ("qwen2.5-coder-3b", "TF_CLEAN_Q3", "-tfq2", Q3), ("qwen2.5-coder-3b", "TF_CLEAN", "-tfc2", own("qwen2.5-coder-3b")),
                           ("qwen2.5-coder-1.5b", "TF_CLEAN_Q3", "-tfq2", Q3), ("qwen2.5-coder-1.5b", "TF_CLEAN", "-tfc2", own("qwen2.5-coder-1.5b"))]),
    "absanim": (1, [("qwen3-8b", "TF_CLEAN", "-tfc2", Q3), ("deepseek-coder-6.7b", "TF_CLEAN", "-tfc2", own("deepseek-coder-6.7b"))]),
    "mitulkabirbadhon": (1, [("deepseek-coder-6.7b", "TF_CLEAN_Q3", "-tfq2", Q3), ("tigercoder-1b", "TF_CLEAN_Q3", "-tfq2", Q3),
                             ("tigercoder-1b", "TF_CLEAN", "-tfc2", own("tigercoder-1b"))]),
}


def slug(model, tag):
    return "bcb-gen-" + model.replace(".", "").replace("_", "-") + tag


def run(args, timeout=600):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args, 1, "", "timeout")


def status(acct, s):
    r = run([PY, "scripts/kaggle_cli.py", acct, "kernels", "status", f"{acct}/{s}"], 120)
    t = (r.stdout + r.stderr).upper()
    for k in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"):
        if f"KERNELWORKERSTATUS.{k}" in t or f'"{k}"' in t:
            return k
    return "UNKNOWN"


def push(acct, model, cond, tag, src):
    r = run([PY, "scripts/make_kaggle_kernel.py", model, "--user", acct, "--only", "EXTERNAL", "--external-translations", src,
             "--external-condition", cond, f"--tag={tag}"])
    p = run([PY, "scripts/kaggle_cli.py", acct, "kernels", "push", "-p", f"kaggle_work/{slug(model, tag)}"])
    ok = "successfully pushed" in (p.stdout + p.stderr)
    print(f"{time.strftime('%H:%M')} PUSH {acct}/{slug(model, tag)} cond={cond} ok={ok}", flush=True)
    return ok


def main():
    pending = {a: list(j) for a, (_, j) in PLAN.items()}
    slots = {a: n for a, (n, _) in PLAN.items()}
    running = {a: [] for a in PLAN}
    while any(pending.values()) or any(running.values()):
        for a in PLAN:
            for job in list(running[a]):
                st = status(a, slug(job[0], job[2]))
                if st == "COMPLETE":
                    r = run([PY, "scripts/collect_kernel.py", a, slug(job[0], job[2]), job[0]], 900)
                    print(f"{time.strftime('%H:%M')} DONE {slug(job[0], job[2])}: {r.stdout.strip()[-160:]}", flush=True)
                    running[a].remove(job)
                elif st in ("ERROR", "CANCEL"):
                    print(f"{time.strftime('%H:%M')} FAILED {a}/{slug(job[0], job[2])} status={st}", flush=True)
                    running[a].remove(job)
            while pending[a] and len(running[a]) < slots[a]:
                job = pending[a].pop(0)
                if push(a, *job):
                    running[a].append(job)
        time.sleep(180)
    print(f"{time.strftime('%H:%M')} ALL DONE", flush=True)


if __name__ == "__main__":
    main()
