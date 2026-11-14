#!/usr/bin/env python3
"""Resumable code generation with the Gemini API (REST), same conditions/records as generate_hf.py.

Key: env GEMINI_API_KEY, or a text file given by --key-file (e.g. a private Kaggle dataset).
Usage: python generate_gemini.py --model gemini-3.1-flash-lite --model-id gemini-3.1-flash-lite \
          --prompts prompts.jsonl --out gens --workers 6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
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
BASE = "https://generativelanguage.googleapis.com/v1beta/models"


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


class Gemini:
    """Round-robins over several API keys; each key has its own minimum call interval (requests per minute cap)."""

    def __init__(self, model, keys, think_budget=0, rpm_per_key=12):
        self.model, self.keys, self.think = model, list(keys), think_budget
        self.interval = 60.0 / max(1, rpm_per_key)
        self.next_ok = [0.0] * len(self.keys)
        self.lock = threading.Lock()
        self.sess = requests.Session()

    def _acquire(self):
        with self.lock:
            i = min(range(len(self.keys)), key=lambda j: self.next_ok[j])
            now = time.time()
            wait = max(0.0, self.next_ok[i] - now)
            self.next_ok[i] = max(now, self.next_ok[i]) + self.interval
        if wait:
            time.sleep(wait)
        return self.keys[i]

    def call(self, system, user, temperature, top_p, max_tokens=2048, seed=None, retries=14):
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": temperature, "topP": top_p, "maxOutputTokens": max_tokens}}
        if seed is not None:
            body["generationConfig"]["seed"] = int(seed)
        if self.think is not None:
            body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": self.think}
        url = f"{BASE}/{self.model}:generateContent"
        last = None
        for attempt in range(retries):
            t0 = time.time()
            try:
                r = self.sess.post(url, params={"key": self._acquire()}, json=body, timeout=120)
                if r.status_code == 200:
                    j = r.json()
                    cand = (j.get("candidates") or [{}])[0]
                    parts = (cand.get("content") or {}).get("parts") or []
                    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                    u = j.get("usageMetadata", {})
                    return {"text": text, "finish": cand.get("finishReason", "UNKNOWN"),
                            "prompt_tokens": u.get("promptTokenCount", 0),
                            "output_tokens": u.get("candidatesTokenCount", 0),
                            "thought_tokens": u.get("thoughtsTokenCount", 0),
                            "latency": time.time() - t0}
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                if r.status_code == 400 and "thinking" in r.text.lower() and self.think is not None:
                    self.think = None  # model does not accept thinkingConfig; retry without it
                    body["generationConfig"].pop("thinkingConfig", None)
                    continue
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(min(120, 8 + 6 * attempt + random.random() * 4))
                    continue
                break
            except requests.RequestException as exc:
                last = f"{type(exc).__name__}: {exc}"
                time.sleep(min(60, 2 ** attempt))
        return {"text": "", "finish": "ERROR", "error": last, "prompt_tokens": 0, "output_tokens": 0,
                "thought_tokens": 0, "latency": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key-file", default="")
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--think-budget", type=int, default=0, help="-1 = do not send thinkingConfig")
    ap.add_argument("--rpm-per-key", type=int, default=12)
    ap.add_argument("--no-translate", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    keys = []
    if os.environ.get("GEMINI_API_KEY"):
        keys.append(os.environ["GEMINI_API_KEY"].strip())
    if args.key_file:
        for ln in Path(args.key_file).read_text().split(chr(10)):
            ln = ln.strip()
            if not ln:
                continue
            keys.append(ln.split("=", 1)[1].strip() if "=" in ln else ln)  # lines like "acc 1 = KEY" or a bare key
    assert keys, "no API key"
    g = Gemini(args.model, keys, None if args.think_budget < 0 else args.think_budget, args.rpm_per_key)

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
        "model": args.model, "temperature": args.temperature, "top_p": args.top_p, "think_budget": args.think_budget,
        "n_samples": args.n_samples, "seed": args.seed, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))

    def emit(rec):
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()

    def gen_one(p, condition, s, tr=None):
        eff = tr["translation"] if tr else p["text"]
        r = g.call(SYSTEM_CODE, eff, args.temperature, args.top_p, 2048, seed=args.seed + s)
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
                if n % 50 == 0:
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
                    "tr_in": r["prompt_tokens"], "tr_out": r["output_tokens"] + r["thought_tokens"],
                    "sig_reinjected": reinj}
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
