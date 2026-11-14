#!/usr/bin/env python3
"""Create a self-contained Kaggle GPU kernel that runs scripts/generate_hf.py for one model.

The generator script and the prompt file are embedded (gzip+base64) in the kernel source, so no Kaggle dataset
version has to be uploaded (uploads from the authors' network are slow and racy).

usage: python scripts/make_kaggle_kernel.py <model_key> [--prompts data/prompts/prompts.jsonl] [--tag -x] [--only DIRECT]
Then: KAGGLE_API_TOKEN=... kaggle kernels push -p kaggle_work/<kernel> --accelerator NvidiaTeslaT4
"""
import argparse
import base64
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BATCH = {"qwen2.5-coder-1.5b": 24, "qwen2.5-coder-3b": 16, "qwen2.5-coder-7b": 10,
         "deepseek-coder-6.7b": 10, "qwen3-8b": 8, "tigercoder-1b": 24, "tigercoder-9b": 4}
EXTRA = {"qwen3-8b": "--qwen3-nothink", "tigercoder-9b": "--quant 4bit"}

TEMPLATE = '''import base64, gzip, os, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "bitsandbytes"], check=False)
def put(path, b64, gz):
    raw = base64.b64decode(b64)
    open(path, "wb").write(gzip.decompress(raw) if gz else raw)
put("/kaggle/working/generate_hf.py", "__SCRIPT__", False)
put("/kaggle/working/prompts.jsonl", "__PROMPTS__", True)
__EXTPUT__
os.makedirs("/kaggle/working/gens", exist_ok=True)
cmd = [sys.executable, "/kaggle/working/generate_hf.py", "--hf-id", "__HF__", "--revision", "__REV__",
       "--model-id", "__MID__", "--prompts", "/kaggle/working/prompts.jsonl", "--out", "/kaggle/working/gens",
       "--batch-size", "__BS__", "--max-new-tokens", "768"] + "__EXTRA__".split()
print(" ".join(cmd), flush=True)
r = subprocess.run(cmd)
print("exit", r.returncode)
sys.exit(r.returncode)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_key")
    ap.add_argument("--user", default="abdullahrubab1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--prompts", default="data/prompts/prompts.jsonl")
    ap.add_argument("--external-translations", default="", help="translations.jsonl of another model (cross-model translate-first)")
    ap.add_argument("--external-condition", default="TRANSLATE_FIRST_X")
    a = ap.parse_args()
    m = json.loads((ROOT / "data/models.json").read_text())[a.model_key]
    slug = f"bcb-gen-{a.model_key.replace('.', '').replace('_', '-')}{a.tag}"
    d = ROOT / "kaggle_work" / slug
    d.mkdir(parents=True, exist_ok=True)
    ext_put, ext_args = "", ""
    if a.external_translations:
        eb = base64.b64encode(gzip.compress((ROOT / a.external_translations).read_bytes(), 9)).decode()
        ext_put = f'put("/kaggle/working/ext_translations.jsonl", "{eb}", True)'
        ext_args = f"--external-translations /kaggle/working/ext_translations.jsonl --external-condition {a.external_condition}"
    extra = " ".join(x for x in (EXTRA.get(a.model_key, ""), f"--limit {a.limit}" if a.limit else "",
                                 f"--only {a.only}" if a.only else "", ext_args) if x)
    sb = base64.b64encode((ROOT / "scripts/generate_hf.py").read_bytes()).decode()
    pb = base64.b64encode(gzip.compress((ROOT / a.prompts).read_bytes(), 9)).decode()
    code = (TEMPLATE.replace("__EXTPUT__", ext_put).replace("__SCRIPT__", sb).replace("__PROMPTS__", pb).replace("__HF__", m["hf_id"])
            .replace("__REV__", m["revision"]).replace("__MID__", a.model_key)
            .replace("__BS__", str(BATCH[a.model_key])).replace("__EXTRA__", extra))
    (d / "kernel.py").write_text(code)
    meta = {"id": f"{a.user}/{slug}", "title": slug, "code_file": "kernel.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
            "model_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(d, f"kernel.py = {len(code) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
