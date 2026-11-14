#!/usr/bin/env python3
"""Resumable code generation against ANY local OpenAI-compatible server (Ollama, LM Studio, vLLM, llama.cpp server, ...),
with the same conditions and output records as generate_hf.py / generate_gemini.py, so that the rest of the pipeline works unchanged.

Examples
  # Ollama (default port 11434): the model name is whatever `ollama list` prints
  python generate_local_server.py --base-url http://localhost:11434/v1 --model qwen2.5-coder:14b --model-id qwen2.5-coder-14b \
        --prompts ../data/prompts/prompts.jsonl --out ../data/generations --workers 2
  # LM Studio (default port 1234)
  python generate_local_server.py --base-url http://localhost:1234/v1 --model <name shown by LM Studio> --model-id <short-id> ...

Conditions fixed by the protocol: system prompt below, temperature 0.2, top_p 0.95, 3 samples, 768 completion tokens,
non-reasoning mode (pass --extra-json '{"think": false}' or a chat-template switch for models that think by default).
The script never sends anything outside the machine unless --base-url points elsewhere. Interrupt and re-run any time: finished samples are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

SYSTEM_CODE = ("You are an expert Python programmer. Return only a single Python code block solving "
               "the requested task. Do not include extra conversational text.")
SYSTEM_TRANSLATE = ("You are a precise technical translator. Translate the programming task below into clear "
                    "English. Keep every code block and every `backticked` identifier, literal and signature "
                    "exactly as written. Output only the translated task text, nothing else.")
TF_VARIANTS = {"BN", "BN_MIX", "BANGLISH", "ES"}
FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S | re.I)


def extract_code(text: str):
    blocks = FENCE.findall(text)
    if blocks:
        return max(blocks, key=len).strip("\n") + "\n", "fenced"
    m = re.search(r"```(?:python|py)?\s*\n(.*)$", text, re.S | re.I)
    if m:
        return m.group(1).strip("\n") + "\n", "unterminated_fence"
    if re.search(r"^\s*(def |import |from |class )", text, re.M):
        return text.strip("\n") + "\n", "raw_code"
    return "", "no_code"


def gid(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


class Server:
    def __init__(self, base_url, model, api_key="local", extra=None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.extra = model, extra or {}
        self.sess = requests.Session()
        self.sess.headers["Authorization"] = f"Bearer {api_key}"

    def call(self, system, user, temperature, top_p, max_tokens, seed=None, retries=4):
        body = {"model": self.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens, "stream": False, **self.extra}
        if seed is not None:
            body["seed"] = int(seed)
        last = None
        for attempt in range(retries):
            t0 = time.time()
            try:
                r = self.sess.post(self.url, json=body, timeout=600)
                if r.status_code == 200:
                    j = r.json()
                    ch = (j.get("choices") or [{}])[0]
                    u = j.get("usage") or {}
                    return {"text": (ch.get("message") or {}).get("content") or "", "finish": ch.get("finish_reason", "UNKNOWN"),
                            "prompt_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0),
                            "thought_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
                            "latency": time.time() - t0}
                last = f"HTTP {r.status_code}: {r.text[:200]}"
            except requests.RequestException as exc:
                last = f"{type(exc).__name__}: {exc}"
            time.sleep(min(30, 2 ** attempt))
        return {"text": "", "finish": "ERROR", "error": last, "prompt_tokens": 0, "output_tokens": 0, "thought_tokens": 0, "latency": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", required=True, help="name as the server knows it")
    ap.add_argument("--model-id", required=True, help="short id used in the result files, e.g. qwen2.5-coder-14b")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--api-key", default="local")
    ap.add_argument("--extra-json", default="{}", help="extra request fields, e.g. '{\"think\": false}'")
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--no-translate", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts (smoke test)")
    ap.add_argument("--only", default="", help="DIRECT and/or TRANSLATE_FIRST")
    args = ap.parse_args()

    g = Server(args.base_url, args.model, args.api_key, json.loads(args.extra_json))
    out_dir = Path(args.out) / args.model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    for f in out_dir.glob("gens_*.jsonl"):
        for line in f.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                rec = json.loads(line)
                if rec.get("finish") != "ERROR":
                    done.add(rec["gen_id"])
    prompts = [json.loads(l) for l in Path(args.prompts).read_text(encoding="utf-8").split("\n") if l.strip()]
    if args.limit:
        prompts = prompts[: args.limit]
    only = set(filter(None, args.only.split(",")))
    lock = threading.Lock()
    fh = open(out_dir / f"gens_{int(time.time())}.jsonl", "a", encoding="utf-8")
    (out_dir / "run_info.json").write_text(json.dumps({
        "model": args.model, "server": args.base_url, "temperature": args.temperature, "top_p": args.top_p,
        "max_tokens": args.max_tokens, "n_samples": args.n_samples, "seed": args.seed, "extra": json.loads(args.extra_json),
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))

    def emit(rec):
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()

    def gen_one(p, condition, s, tr=None):
        eff = tr["translation"] if tr else p["text"]
        r = g.call(SYSTEM_CODE, eff, args.temperature, args.top_p, args.max_tokens, seed=args.seed + s)
        code, status = extract_code(r["text"])
        emit({"gen_id": gid(args.model_id, p["task_id"], p["variant"], condition, s), "model_id": args.model_id,
              "task_id": p["task_id"], "variant": p["variant"], "condition": condition, "sample_idx": s,
              "seed": args.seed + s, "prompt_text": p["text"], "effective_prompt": eff,
              "translation_tokens_in": tr["tr_in"] if tr else 0, "translation_tokens_out": tr["tr_out"] if tr else 0,
              "signature_reinjected": tr["sig_reinjected"] if tr else False,
              "prompt_tokens": r["prompt_tokens"], "output_tokens": r["output_tokens"],
              "thought_tokens": r["thought_tokens"], "finish": r["finish"], "gen_seconds_share": r["latency"],
              "error": r.get("error"), "raw_output": r["text"], "extracted_code": code, "extraction_status": status})
        return r["finish"] != "ERROR"

    def run(items, condition, label):
        jobs = [(p, s, tr) for p, tr in items for s in range(args.n_samples)
                if gid(args.model_id, p["task_id"], p["variant"], condition, s) not in done]
        print(f"[{label}] {len(jobs)} samples to generate", flush=True)
        ok = n = 0
        t0 = time.time()
        with ThreadPoolExecutor(args.workers) as ex:
            futs = [ex.submit(gen_one, p, condition, s, tr) for p, s, tr in jobs]
            for f in as_completed(futs):
                n += 1; ok += bool(f.result())
                if n % 25 == 0:
                    print(f"[{label}] {n}/{len(jobs)} ok={ok} {time.time() - t0:.0f}s", flush=True)

    if not only or "DIRECT" in only:
        run([(p, None) for p in prompts], "DIRECT", "DIRECT")

    if not args.no_translate and (not only or "TRANSLATE_FIRST" in only):
        tr_path = out_dir / "translations.jsonl"
        trans = {}
        if tr_path.exists():
            for l in tr_path.read_text(encoding="utf-8").split("\n"):
                if l.strip():
                    r = json.loads(l); trans[(r["task_id"], r["variant"])] = r
        src = [p for p in prompts if p["variant"] in TF_VARIANTS]

        def tr_one(p):
            r = g.call(SYSTEM_TRANSLATE, p["text"], 0.0, 1.0, 1500, seed=args.seed)
            if r["finish"] == "ERROR":
                return None
            t = r["text"].strip(); reinj = False
            if p["signature"].strip() not in t:
                t += f"\n\n```python\n{p['signature'].strip()}\n```"; reinj = True
            return {"task_id": p["task_id"], "variant": p["variant"], "translation": t,
                    "tr_in": r["prompt_tokens"], "tr_out": r["output_tokens"] + r["thought_tokens"], "sig_reinjected": reinj}
        pend = [p for p in src if (p["task_id"], p["variant"]) not in trans]
        print(f"[translate] {len(pend)} prompts", flush=True)
        with ThreadPoolExecutor(args.workers) as ex, open(tr_path, "a", encoding="utf-8") as tfh:
            for rec in ex.map(tr_one, pend):
                if rec:
                    trans[(rec["task_id"], rec["variant"])] = rec
                    tfh.write(json.dumps(rec, ensure_ascii=False) + "\n"); tfh.flush()
        items = [(p, trans[(p["task_id"], p["variant"])]) for p in src if (p["task_id"], p["variant"]) in trans]
        run(items, "TRANSLATE_FIRST", "TRANSLATE_FIRST")
    fh.close()
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
