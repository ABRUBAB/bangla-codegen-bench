#!/usr/bin/env python3
"""Create a Kaggle kernel directory that runs scripts/generate_hf.py for one model.

usage: python scripts/make_kaggle_kernel.py <model_key> [--user abdullahrubab1] [--limit N] [--extra "..."]
Then: KAGGLE_API_TOKEN=... kaggle kernels push -p kaggle_work/<kernel> --accelerator NvidiaTeslaT4
"""
import argparse, json, textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BATCH = {"qwen2.5-coder-1.5b": 24, "qwen2.5-coder-3b": 16, "qwen2.5-coder-7b": 10,
         "deepseek-coder-6.7b": 10, "qwen3-8b": 8, "tigercoder-1b": 24, "tigercoder-9b": 4}
EXTRA = {"qwen3-8b": "--qwen3-nothink", "tigercoder-9b": "--quant 4bit"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_key")
    ap.add_argument("--user", default="abdullahrubab1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    models = json.loads((ROOT / "data/models.json").read_text())
    m = models[a.model_key]
    slug = f"bcb-gen-{a.model_key.replace('.', '').replace('_', '-')}{a.tag}"
    d = ROOT / "kaggle_work" / slug
    d.mkdir(parents=True, exist_ok=True)
    extra = EXTRA.get(a.model_key, "")
    lim = f"--limit {a.limit}" if a.limit else ""
    only = f"--only {a.only}" if a.only else ""
    code = textwrap.dedent(f'''\
        import os, subprocess, sys, shutil, glob, json
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "bitsandbytes"], check=False)
        data = glob.glob("/kaggle/input/**/prompts.jsonl", recursive=True)[0]
        script = glob.glob("/kaggle/input/**/generate_hf.py", recursive=True)[0]
        os.makedirs("/kaggle/working/gens", exist_ok=True)
        cmd = [sys.executable, script, "--hf-id", "{m['hf_id']}", "--revision", "{m['revision']}",
               "--model-id", "{a.model_key}", "--prompts", data, "--out", "/kaggle/working/gens",
               "--batch-size", "{BATCH[a.model_key]}", "--max-new-tokens", "768"] + "{extra} {lim} {only}".split()
        print(" ".join(cmd), flush=True)
        r = subprocess.run(cmd)
        print("exit", r.returncode)
        sys.exit(r.returncode)
        ''')
    (d / "kernel.py").write_text(code)
    meta = {"id": f"{a.user}/{slug}", "title": slug, "code_file": "kernel.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [f"{a.user}/bcb-data"],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(d)

if __name__ == "__main__":
    main()
