#!/usr/bin/env python3
"""Run the kaggle CLI as one of the three authorised accounts without printing or exporting the token.
Usage: python scripts/kaggle_cli.py <abdullahrubab1|absanim|mitulkabirbadhon> kernels output <user>/<slug> -p <dir>
Tokens are read from the files in the project folder (outside the repository)."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def token(user):
    if user == "mitulkabirbadhon":
        return (ROOT / "Kaggle_Api_key.txt").read_text().strip()
    n = {"abdullahrubab1": 1, "absanim": 2}[user]
    for line in (ROOT / "kggle api keys.txt").read_text().splitlines():
        if line.startswith(f"ac_{n} key"):
            return line.split(":", 1)[1].strip()
    raise SystemExit(f"no token for {user}")


if __name__ == "__main__":
    user = sys.argv[1]
    env = dict(os.environ, KAGGLE_API_TOKEN=token(user), PYTHONUTF8="1")
    sys.exit(subprocess.run([sys.executable, "-m", "kaggle.cli", *sys.argv[2:]], env=env).returncode)
