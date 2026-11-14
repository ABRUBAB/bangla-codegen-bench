#!/usr/bin/env python3
"""In-container job runner for the BanglaCodeBench execution sandbox.

Protocol (JSON Lines):
  stdin  : one job per line
           {"job_id": str, "solution": str, "tests": {"functional": str, "exploit": str?},
            "timeout": float?}
  stdout : one result per line
           {"job_id": str, "results": {<test_name>: {"passed": bool, "failure_type": str|None,
            "exit_code": int, "duration": float, "tail": str}}, "compile_error": str|None}

Isolation (the *container* supplies network=none, read-only root FS, cgroup memory/pids
limits and a non-root user; this runner adds per-process POSIX resource limits and a
wall-clock timeout, and kills the whole process group on expiry).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

try:  # POSIX only; absent on Windows (LOCAL authoring mode for TRUSTED code only)
    import resource
except ImportError:  # pragma: no cover
    resource = None

DEFAULT_TIMEOUT = 10.0
MEM_BYTES = 1024 * 1024 * 1024
FSIZE_BYTES = 20 * 1024 * 1024
TAIL_CHARS = 1800


def _limits(cpu_seconds: int):
    if resource is None:
        return None

    def _apply():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (MEM_BYTES, MEM_BYTES))
        resource.setrlimit(resource.RLIMIT_FSIZE, (FSIZE_BYTES, FSIZE_BYTES))
        resource.setrlimit(resource.RLIMIT_NPROC, (128, 128))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return _apply


_FAIL_PATTERNS = [
    ("syntax_error", re.compile(r"SyntaxError|IndentationError|TabError")),
    ("import_error", re.compile(r"ModuleNotFoundError|ImportError: (?!cannot import name '\w+' from 'solution')")),
    ("missing_symbol", re.compile(r"cannot import name '\w+' from 'solution'|has no attribute|NameError")),
    ("signature_mismatch", re.compile(r"TypeError: .*(positional argument|unexpected keyword|takes \d+|missing \d+)")),
    ("wrong_answer", re.compile(r"AssertionError|assert ")),
    ("runtime_error", re.compile(r"\b(TypeError|ValueError|KeyError|IndexError|ZeroDivisionError|RecursionError|"
                                 r"OverflowError|MemoryError|RuntimeError|OSError|StopIteration|UnboundLocalError)\b")),
]


def _classify(output: str) -> str:
    for name, pat in _FAIL_PATTERNS:
        if pat.search(output):
            return name
    return "other"


def _kill(proc) -> None:
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        pass


def _run_one_test(workdir: str, test_name: str, test_code: str, timeout: float) -> dict:
    fname = f"test_{test_name}.py"
    with open(os.path.join(workdir, fname), "w", encoding="utf-8") as fh:
        fh.write(test_code)
    env = {
        "PATH": os.environ.get("BCB_JOB_PATH", "/usr/local/bin:/usr/bin:/bin"),
        **({"SYSTEMROOT": os.environ["SYSTEMROOT"]} if "SYSTEMROOT" in os.environ else {}),
        "HOME": workdir,
        "TMPDIR": workdir,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": workdir,
        "LANG": "C.UTF-8",
    }
    if os.environ.get("BCB_LOCAL") == "1":  # authoring mode: keep user-site packages visible
        env = {**os.environ, **{k: v for k, v in env.items() if k not in ("PATH",)}}
        env["PATH"] = os.environ.get("PATH", "")
    cmd = [sys.executable, "-m", "pytest", "-q", "-x", "--tb=short", "-p", "no:cacheprovider", fname]
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, cwd=workdir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", preexec_fn=_limits(int(timeout) + 5),
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        _kill(proc)
        out, _ = proc.communicate()
        return {"passed": False, "failure_type": "timeout", "exit_code": -9,
                "duration": round(time.time() - t0, 3), "tail": (out or "")[-TAIL_CHARS:]}
    finally:
        _kill(proc)
    passed = code == 0
    tail = (out or "")[-TAIL_CHARS:]
    if code in (-9, 137):
        ftype = "memory_or_cpu_limit"
    elif passed:
        ftype = None
    else:
        ftype = _classify(out or "")
    return {"passed": passed, "failure_type": ftype, "exit_code": code,
            "duration": round(time.time() - t0, 3), "tail": tail}


def run_job(job: dict) -> dict:
    workdir = tempfile.mkdtemp(prefix="job_", dir=os.environ.get("TMPDIR", "/tmp"))
    try:
        compile_error = None
        try:
            compile(job["solution"], "solution.py", "exec")
        except SyntaxError as exc:  # still run tests so the failure is recorded uniformly
            compile_error = f"{type(exc).__name__}: {exc}"
        with open(os.path.join(workdir, "solution.py"), "w", encoding="utf-8") as fh:
            fh.write(job["solution"])
        timeout = float(job.get("timeout") or DEFAULT_TIMEOUT)
        results = {}
        for name, code in job["tests"].items():
            if not code:
                continue
            results[name] = _run_one_test(workdir, name, code, timeout)
        return {"job_id": job["job_id"], "results": results, "compile_error": compile_error}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        job = json.loads(line)
        try:
            res = run_job(job)
        except Exception as exc:  # harness failure must never kill the batch
            res = {"job_id": job.get("job_id"), "results": {}, "compile_error": None,
                   "harness_error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(res, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
