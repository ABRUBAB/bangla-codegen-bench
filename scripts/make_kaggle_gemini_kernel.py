#!/usr/bin/env python3
"""Create a Kaggle CPU kernel (internet on) that runs scripts/generate_gemini.py. Google's API is
unreachable from the authors' network, so Gemini generation is executed on Kaggle. The generator script is
embedded (base64) in the kernel source so that no dataset round-trip is needed; prompts + the API key come
from private Kaggle datasets (bcb-data, bcb-secrets)."""
import argparse
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TEMPLATE = '''import base64, glob, os, subprocess, sys
data = glob.glob("/kaggle/input/**/__PROMPTS__", recursive=True)[0]
key = glob.glob("/kaggle/input/**/gemini_key.txt", recursive=True)[0]
script = "/kaggle/working/generate_gemini.py"
open(script, "wb").write(base64.b64decode("__B64__"))
os.makedirs("/kaggle/working/gens", exist_ok=True)
cmd = [sys.executable, script, "--model", "__MODEL__", "--model-id", "__MODEL__", "--prompts", data,
       "--out", "/kaggle/working/gens", "--key-file", key, "--workers", "__WORKERS__"] + "__EXTRA__".split()
r = subprocess.run(cmd)
print("exit", r.returncode)
sys.exit(r.returncode)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--user", default="abdullahrubab1")
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default="")
    ap.add_argument("--think-budget", type=int, default=0)
    ap.add_argument("--prompts-name", default="prompts.jsonl")
    a = ap.parse_args()
    slug = f"bcb-gen-{a.model.replace('.', '').replace('_', '-')}{a.tag}"
    d = ROOT / "kaggle_work" / slug
    d.mkdir(parents=True, exist_ok=True)
    extra = " ".join(x for x in (f"--limit {a.limit}" if a.limit else "", f"--only {a.only}" if a.only else "",
                                 f"--think-budget {a.think_budget}") if x)
    b64 = base64.b64encode((ROOT / "scripts" / "generate_gemini.py").read_bytes()).decode()
    code = (TEMPLATE.replace("__B64__", b64).replace("__MODEL__", a.model)
            .replace("__WORKERS__", str(a.workers)).replace("__PROMPTS__", a.prompts_name).replace("__EXTRA__", extra))
    (d / "kernel.py").write_text(code)
    meta = {"id": f"{a.user}/{slug}", "title": slug, "code_file": "kernel.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [f"{a.user}/bcb-data", f"{a.user}/bcb-secrets"],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(d)


if __name__ == "__main__":
    main()
