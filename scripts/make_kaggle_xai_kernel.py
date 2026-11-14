#!/usr/bin/env python3
"""Kaggle GPU kernel for the white-box XAI probes (scripts/xai_attribution.py) on one model."""
import argparse, base64, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = '''import base64, glob, os, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate"], check=False)
prompts = glob.glob("/kaggle/input/**/prompts.jsonl", recursive=True)[0]
tasks = glob.glob("/kaggle/input/**/tasks_all.jsonl", recursive=True)[0]
script = "/kaggle/working/xai_attribution.py"
open(script, "wb").write(base64.b64decode("__B64__"))
os.makedirs("/kaggle/working/xai", exist_ok=True)
cmd = [sys.executable, script, "--hf-id", "__HF__", "--revision", "__REV__", "--model-id", "__MID__",
       "--prompts", prompts, "--tasks", tasks, "--out", "/kaggle/working/xai"] + "__EXTRA__".split()
r = subprocess.run(cmd); print("exit", r.returncode); sys.exit(r.returncode)
'''
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model_key"); ap.add_argument("--user", default="abdullahrubab1")
    ap.add_argument("--tag", default=""); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--variants", default="EN,BN,BN_MIX,BANGLISH,ID_ONLY,BN_SPELL,BN_TYPO,EN_TYPO")
    a = ap.parse_args()
    m = json.loads((ROOT / "data/models.json").read_text())[a.model_key]
    slug = f"bcb-xai-{a.model_key.replace('.', '').replace('_', '-')}{a.tag}"
    d = ROOT / "kaggle_work" / slug; d.mkdir(parents=True, exist_ok=True)
    b64 = base64.b64encode((ROOT / "scripts/xai_attribution.py").read_bytes()).decode()
    extra = f"--variants {a.variants}" + (f" --limit {a.limit}" if a.limit else "")
    code = (TEMPLATE.replace("__B64__", b64).replace("__HF__", m["hf_id"]).replace("__REV__", m["revision"])
            .replace("__MID__", a.model_key).replace("__EXTRA__", extra))
    (d / "kernel.py").write_text(code)
    meta = {"id": f"{a.user}/{slug}", "title": slug, "code_file": "kernel.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
            "dataset_sources": [f"{a.user}/bcb-data"], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2)); print(d)
if __name__ == "__main__":
    main()
