#!/usr/bin/env python3
"""Download a finished Kaggle generation kernel and add its gens_*.jsonl to data/generations/<model_id>/.
usage: python scripts/collect_kernel.py <account> <kernel-slug> <model_id>
(translations.jsonl of delta/top-up runs, if any, must be merged by hand: they are NOT copied here.)"""
import collections
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    acct, slug, model = sys.argv[1:4]
    out = ROOT / "kaggle_work" / f"out_{slug}"
    out.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, str(ROOT / "scripts/kaggle_cli.py"), acct, "kernels", "output", f"{acct}/{slug}", "-p", str(out)],
                       capture_output=True, text=True)
    files = sorted((out / "gens" / model).glob("gens_*.jsonl"))
    if not files:
        print(f"{slug}: NO gens found; {r.stdout[-200:]} {r.stderr[-200:]}")
        return
    dst = ROOT / "data/generations" / model
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy(f, dst)
        recs = [json.loads(x) for x in f.read_text(encoding="utf-8").split("\n") if x.strip()]
        st = collections.Counter(x["extraction_status"] for x in recs)
        print(f"{slug} -> {model}/{f.name}: {len(recs)} gens, conditions {dict(collections.Counter(x['condition'] for x in recs))}, "
              f"no_code {100 * st['no_code'] / max(1, len(recs)):.0f}%")


if __name__ == "__main__":
    main()
