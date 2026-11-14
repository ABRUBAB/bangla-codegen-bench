#!/usr/bin/env python3
"""Read-only inventory of a machine for running the benchmark: GPUs, RAM, disk, Python, and installed local models
(Ollama, LM Studio, Hugging Face cache). Prints a short report and writes center_pc_report.json next to this script.
It does not send anything anywhere and changes nothing. Run:  python center_pc_probe.py"""
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, shell=isinstance(cmd, str)).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"(unavailable: {type(e).__name__})"


def dir_models(root, depth=2):
    p = Path(root).expanduser()
    if not p.exists():
        return None
    out = []
    for q in sorted(p.glob("*/" * 1 + "*"))[:200] if depth == 2 else sorted(p.glob("*"))[:200]:
        out.append(str(q.relative_to(p)))
    return out


def main():
    rep = {"os": platform.platform(), "python": sys.version.split()[0]}
    rep["gpu"] = run("nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader")
    rep["gpu_processes"] = run("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader")
    try:
        import psutil
        rep["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        rep["ram_free_gb"] = round(psutil.virtual_memory().available / 1e9, 1)
    except ImportError:
        rep["ram_gb"] = "(pip install psutil to read)"
    rep["disk_free_gb"] = {str(Path(d)): round(shutil.disk_usage(d).free / 1e9, 1) for d in ["C:/", "D:/", "E:/"] if Path(d).exists()} \
        if os.name == "nt" else {"/": round(shutil.disk_usage("/").free / 1e9, 1)}
    rep["ollama"] = run("ollama list") if shutil.which("ollama") else "(ollama not found)"
    rep["lm_studio_models"] = dir_models("~/.lmstudio/models") or dir_models("~/.cache/lm-studio/models")
    rep["hf_cache"] = [str(p.name) for p in (Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub").glob("models--*")]
    rep["torch"] = run([sys.executable, "-c", "import torch;print(torch.__version__, torch.cuda.is_available())"])
    rep["vllm_or_llamacpp"] = {"vllm": bool(shutil.which("vllm")), "llama-server": bool(shutil.which("llama-server"))}
    text = json.dumps(rep, indent=2, ensure_ascii=False)
    (Path(__file__).parent / "center_pc_report.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
