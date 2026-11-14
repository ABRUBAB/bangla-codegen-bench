#!/usr/bin/env python3
"""Print the status of all project kernels across the three authorised Kaggle accounts.
Tokens are read from the files in the project folder (never printed)."""
import os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
def tok_ac(n):
    for l in (ROOT / "kggle api keys.txt").read_text().splitlines():
        if l.startswith(f"ac_{n} key"):
            return l.split(":", 1)[1].strip()
TOK = {"abdullahrubab1": tok_ac(1), "absanim": tok_ac(2), "mitulkabirbadhon": (ROOT / "Kaggle_Api_key.txt").read_text().strip()}
KERNELS = [k.split("/") for k in sys.argv[1:]] or []
for user, slug in KERNELS:
    env = dict(os.environ, KAGGLE_API_TOKEN=TOK[user], PYTHONUTF8="1")
    r = subprocess.run([sys.executable, "-m", "kaggle.cli", "kernels", "status", f"{user}/{slug}"], capture_output=True, text=True, env=env, timeout=90)
    print(f"{user}/{slug}: {r.stdout.strip().split('status ')[-1] if r.stdout else r.stderr.strip()[-80:]}")
