#!/usr/bin/env python3
"""Bandit static analysis of generated code (never executes the code).

Writes each extracted solution to a scratch dir as <gen_id>.py, runs one `bandit -r` pass, and writes
data/results/bandit.jsonl: {gen_id, issues:[{test_id, severity, confidence, cwe, line}]}.
"""
import argparse, json, subprocess, sys, tempfile, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_execution import load_gens  # noqa

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", default="data/generations")
    ap.add_argument("--out", default="data/results/bandit.jsonl")
    ap.add_argument("--models", default="")
    a = ap.parse_args()
    gens = load_gens(Path(a.gens), set(filter(None, a.models.split(","))))
    tmp = Path(tempfile.mkdtemp(prefix="bcb_bandit_"))
    try:
        for gid, g in gens.items():
            (tmp / f"{gid}.py").write_text(g.get("extracted_code") or "", encoding="utf-8")
        r = subprocess.run([sys.executable, "-m", "bandit", "-r", str(tmp), "-f", "json", "-q", "--exit-zero"],
                           capture_output=True, text=True, timeout=7200)
        j = json.loads(r.stdout or "{}")
        issues = {}
        for it in j.get("results", []):
            gid = Path(it["filename"]).stem
            issues.setdefault(gid, []).append({"test_id": it["test_id"], "severity": it["issue_severity"],
                "confidence": it["issue_confidence"], "cwe": (it.get("issue_cwe") or {}).get("id"), "line": it["line_number"]})
        errs = {Path(e["filename"]).stem for e in j.get("errors", [])}
        out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            for gid in gens:
                fh.write(json.dumps({"gen_id": gid, "issues": issues.get(gid, []), "parse_error": gid in errs}) + "\n")
        print(f"bandit: {len(gens)} files, {sum(1 for v in issues.values() if v)} with findings, {len(errs)} unparsable -> {out}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    main()
