#!/usr/bin/env python3
"""Self-contained Kaggle GPU kernel for the white-box XAI probes (scripts/xai_attribution.py) on one model.
Script, prompts and tasks are embedded (gzip+base64) so no dataset upload is needed."""
import argparse, base64, gzip, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = '''import base64, gzip, os, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate"], check=False)
def put(path, b64, gz):
    raw = base64.b64decode(b64); open(path, "wb").write(gzip.decompress(raw) if gz else raw)
put("/kaggle/working/xai_attribution.py", "__SCRIPT__", False)
put("/kaggle/working/prompts.jsonl", "__PROMPTS__", True)
put("/kaggle/working/tasks_all.jsonl", "__TASKS__", True)
os.makedirs("/kaggle/working/xai", exist_ok=True)
cmd = [sys.executable, "/kaggle/working/xai_attribution.py", "--hf-id", "__HF__", "--revision", "__REV__", "--model-id", "__MID__",
       "--prompts", "/kaggle/working/prompts.jsonl", "--tasks", "/kaggle/working/tasks_all.jsonl", "--out", "/kaggle/working/xai"] + "__EXTRA__".split()
r = subprocess.run(cmd); print("exit", r.returncode); sys.exit(r.returncode)
'''
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model_key"); ap.add_argument("--user", default="abdullahrubab1")
    ap.add_argument("--tag", default=""); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--variants", default="EN,BN,BN_MIX,BANGLISH,ES,ID_ONLY,BN_SPELL,BN_TYPO,EN_TYPO")
    a = ap.parse_args()
    m = json.loads((ROOT / "data/models.json").read_text())[a.model_key]
    slug = f"bcb-xai-{a.model_key.replace('.', '').replace('_', '-')}{a.tag}"
    d = ROOT / "kaggle_work" / slug; d.mkdir(parents=True, exist_ok=True)
    enc = lambda b, z: base64.b64encode(gzip.compress(b, 9) if z else b).decode()
    tasks = (ROOT / "data/tasks/functional.jsonl").read_bytes() + (ROOT / "data/tasks/security.jsonl").read_bytes()
    extra = f"--variants {a.variants}" + (f" --limit {a.limit}" if a.limit else "")
    code = (TEMPLATE.replace("__SCRIPT__", enc((ROOT / "scripts/xai_attribution.py").read_bytes(), False))
            .replace("__PROMPTS__", enc((ROOT / "data/prompts/prompts.jsonl").read_bytes(), True))
            .replace("__TASKS__", enc(tasks, True)).replace("__HF__", m["hf_id"]).replace("__REV__", m["revision"])
            .replace("__MID__", a.model_key).replace("__EXTRA__", extra))
    (d / "kernel.py").write_text(code)
    meta = {"id": f"{a.user}/{slug}", "title": slug, "code_file": "kernel.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
            "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2)); print(d, len(code) // 1024, "KB")
if __name__ == "__main__":
    main()
