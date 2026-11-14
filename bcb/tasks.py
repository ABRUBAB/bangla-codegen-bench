"""Task loading and schema validation for BanglaCodeBench."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parent.parent
TASK_DIR = ROOT / "data" / "tasks"

REQUIRED = ["task_id", "kind", "category", "function_name", "signature", "description",
            "reference_solution", "functional_tests"]
DESC_KEYS = ["en", "bn", "bn_mix", "banglish"]
BANGLA_RE = re.compile(r"[ঀ-৿]")
BACKTICK_RE = re.compile(r"`([^`]+)`")


def load_tasks(paths: Iterable[Path] | None = None, kind: str | None = None) -> List[dict]:
    if paths is None:
        paths = sorted(TASK_DIR.glob("*.jsonl"))
    tasks: List[dict] = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    if kind:
        tasks = [t for t in tasks if t["kind"] == kind]
    return sorted(tasks, key=lambda t: t["task_id"])


def validate_task(t: dict) -> List[str]:
    """Return a list of schema/linguistic problems (empty list == OK)."""
    errs: List[str] = []
    for k in REQUIRED:
        if k not in t or t[k] in (None, "", {}):
            errs.append(f"missing field: {k}")
    if errs:
        return errs
    if t["kind"] not in ("functional", "security"):
        errs.append("kind must be functional|security")
    if t["kind"] == "security":
        for k in ("cwe_ids", "exploit_tests", "insecure_reference"):
            if not t.get(k):
                errs.append(f"security task missing {k}")
    d = t["description"]
    for k in DESC_KEYS:
        if not d.get(k, "").strip():
            errs.append(f"description.{k} empty")
    if errs:
        return errs
    # script rules
    if BANGLA_RE.search(d["en"]):
        errs.append("EN description contains Bangla script")
    if not BANGLA_RE.search(d["bn"]):
        errs.append("BN description has no Bangla script")
    if not BANGLA_RE.search(d["bn_mix"]):
        errs.append("BN_MIX description has no Bangla script")
    if BANGLA_RE.search(d["banglish"]):
        errs.append("BANGLISH description contains Bangla script (must be Latin only)")
    # identifier preservation: every `identifier` in EN must appear verbatim in every variant
    en_ids = set(BACKTICK_RE.findall(d["en"]))
    for k in ("bn", "bn_mix", "banglish"):
        missing = [i for i in en_ids if i not in d[k]]
        if missing:
            errs.append(f"{k}: identifiers not preserved verbatim: {missing}")
    # numeric literal preservation
    en_nums = set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", d["en"]))
    for k in ("bn", "bn_mix", "banglish"):
        txt = d[k]
        # allow Bengali digits too
        bn_digits = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
        norm = txt.translate(bn_digits)
        miss = [n for n in en_nums if n not in norm]
        if miss:
            errs.append(f"{k}: numeric literals not preserved: {miss}")
    return errs


def build_jobs(task: dict, code: str, job_id: str, with_exploit: bool | None = None,
               timeout: float = 10.0) -> dict:
    tests = {"functional": task["functional_tests"]}
    if with_exploit is None:
        with_exploit = task["kind"] == "security"
    if with_exploit and task.get("exploit_tests"):
        tests["exploit"] = task["exploit_tests"]
    return {"job_id": job_id, "solution": code, "tests": tests, "timeout": timeout}
