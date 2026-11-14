"""Host-side driver for the Docker execution sandbox.

All model-generated code is executed *only* inside the `bcb-sandbox` container:
  --network none        no network access (exploit tests cannot exfiltrate anything)
  --read-only + tmpfs   immutable root FS, small scratch space
  --cap-drop ALL, no-new-privileges, non-root user, cgroup memory / pids / cpu limits
plus per-test wall-clock timeout and POSIX rlimits (see sandbox/runner.py).
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List

import os

IMAGE = "bcb-sandbox"
SANDBOX_DIR = Path(__file__).resolve().parent / "sandbox"
# Backend: "docker" (default, preferred) or "wsl" (a dedicated, network-less WSL2 distro with
# automount/interop disabled; used when Docker Desktop is unavailable).
BACKEND = os.environ.get("BCB_BACKEND", "docker")
WSL_DISTRO = os.environ.get("BCB_WSL_DISTRO", "bcb-sandbox")


def build_image(image: str = IMAGE) -> None:
    subprocess.run(["docker", "build", "-t", image, str(SANDBOX_DIR)], check=True)


def _docker_cmd(image: str, memory: str, cpus: str) -> List[str]:
    if BACKEND == "local":
        # !! TRUSTED code only (reference solutions / authoring). No isolation, no resource limits.
        os.environ["BCB_LOCAL"] = "1"
        return [sys.executable, str(SANDBOX_DIR / "runner.py")]
    if BACKEND == "wsl":
        # root only to create the empty network namespace; the job runs as uid 1001.
        return [
            "wsl", "-d", WSL_DISTRO, "-u", "root", "--",
            "unshare", "--net", "--pid", "--fork", "--mount-proc", "--",
            "setpriv", "--reuid=1001", "--regid=1001", "--clear-groups", "--no-new-privs",
            "/usr/bin/python3", "/opt/bcb/runner.py",
        ]
    return [
        "docker", "run", "--rm", "-i", "--network", "none",
        "--memory", memory, "--memory-swap", memory, "--cpus", cpus,
        "--pids-limit", "256", "--read-only",
        "--tmpfs", "/tmp:rw,exec,nosuid,size=512m",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        image,
    ]


def _run_shard(shard: List[dict], image: str, memory: str, cpus: str) -> List[dict]:
    payload = "\n".join(json.dumps(j, ensure_ascii=False) for j in shard) + "\n"
    per_job = max(float(j.get("timeout") or 10.0) * max(1, len(j["tests"])) + 5.0 for j in shard)
    budget = per_job * len(shard) + 120.0
    try:
        proc = subprocess.run(
            _docker_cmd(image, memory, cpus), input=payload.encode("utf-8"),
            capture_output=True, timeout=budget,
        )
        out = proc.stdout.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"").decode("utf-8", errors="replace")
    results = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return results


def run_jobs(
    jobs: Iterable[dict], workers: int = 4, shard_size: int = 25, image: str = IMAGE,
    memory: str = "1536m", cpus: str = "1.0", verbose: bool = True,
) -> Dict[str, dict]:
    """Run jobs in parallel containers. Returns {job_id: result}. Jobs lost to a container
    crash are retried one-by-one; if they still fail they get a `harness_error` entry."""
    jobs = list(jobs)
    shards = [jobs[i:i + shard_size] for i in range(0, len(jobs), shard_size)]
    done: Dict[str, dict] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_shard, s, image, memory, cpus): s for s in shards}
        for n, fut in enumerate(as_completed(futs), 1):
            for r in fut.result():
                done[r["job_id"]] = r
            if verbose:
                el = time.time() - t0
                print(f"[sandbox] shard {n}/{len(shards)}  jobs done {len(done)}/{len(jobs)}  "
                      f"{el:.0f}s elapsed", file=sys.stderr, flush=True)
    missing = [j for j in jobs if j["job_id"] not in done]
    for j in missing:  # retry individually
        res = _run_shard([j], image, memory, cpus)
        if res:
            done[j["job_id"]] = res[0]
        else:
            done[j["job_id"]] = {"job_id": j["job_id"], "results": {}, "compile_error": None,
                                 "harness_error": "container_crash_or_timeout"}
    return done


def docker_available() -> bool:
    if BACKEND == "local":
        return True
    if BACKEND == "wsl":
        try:
            r = subprocess.run(["wsl", "-d", WSL_DISTRO, "-u", "root", "--", "test", "-f", "/opt/bcb/runner.py"],
                               capture_output=True, timeout=60)
            return r.returncode == 0
        except Exception:
            return False
    try:
        r = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"],
                           capture_output=True, timeout=20)
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


def image_exists(image: str = IMAGE) -> bool:
    if BACKEND in ("local", "wsl"):
        return docker_available()
    try:
        r = subprocess.run(["docker", "image", "inspect", image], capture_output=True, timeout=30)
        return r.returncode == 0
    except Exception:
        return False
