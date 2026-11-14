#!/usr/bin/env python3
"""Batched, resumable code generation with Hugging Face open-weights models (Kaggle T4x2 or local).

Conditions
  DIRECT           the prompt is sent to the model as-is
  TRANSLATE_FIRST  (variants BN, BN_MIX, BANGLISH) the *same* model first translates the prompt into
                   English (greedy), then generates code from the translation (3 samples)

Output: <out>/<model_id>/gens_*.jsonl  (one JSON object per generated sample, flushed per batch).
Re-running skips every gen_id already present, so a crashed/timed-out session can simply be restarted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_CODE = ("You are an expert Python programmer. Return only a single Python code block solving "
               "the requested task. Do not include extra conversational text.")
SYSTEM_TRANSLATE = ("You are a precise technical translator. Translate the programming task below into clear "
                    "English. Keep every code block and every `backticked` identifier, literal and signature "
                    "exactly as written. Output only the translated task text, nothing else.")
TF_VARIANTS = {"BN", "BN_MIX", "BANGLISH", "ES"}
FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S | re.I)


def extract_code(text: str) -> tuple[str, str]:
    blocks = FENCE.findall(text)
    if blocks:
        return max(blocks, key=len).strip("\n") + "\n", "fenced"
    m = re.search(r"```(?:python|py)?\s*\n(.*)$", text, re.S | re.I)  # unterminated fence (truncated)
    if m:
        return m.group(1).strip("\n") + "\n", "unterminated_fence"
    if re.search(r"^\s*(def |import |from |class )", text, re.M):
        return text.strip("\n") + "\n", "raw_code"
    return "", "no_code"


def gid(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def chat_text(tok, system: str, user: str, qwen3_nothink: bool) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kw = {"enable_thinking": False} if qwen3_nothink else {}
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    except Exception:
        # templates without system role support (e.g. Gemma)
        msgs = [{"role": "user", "content": system + "\n\n" + user}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)


@torch.inference_mode()
def generate_batch(model, tok, texts, n, do_sample, max_new, temperature, top_p, seed):
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    torch.manual_seed(seed)
    t0 = time.time()
    kw = dict(max_new_tokens=max_new, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    if do_sample:
        kw.update(do_sample=True, temperature=temperature, top_p=top_p, num_return_sequences=n)
    else:
        kw.update(do_sample=False, num_return_sequences=1)
    out = model.generate(**enc, **kw)
    dt = time.time() - t0
    plen = enc["input_ids"].shape[1]
    prompt_lens = enc["attention_mask"].sum(1).tolist()
    res = []
    per = n if do_sample else 1
    for i in range(len(texts)):
        for s in range(per):
            seq = out[i * per + s, plen:]
            ids = seq.tolist()
            eos = tok.eos_token_id
            eos_set = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
            cut, finished = len(ids), False
            for j, t in enumerate(ids):
                if t in eos_set or t == tok.pad_token_id:
                    cut, finished = j, True
                    break
            text = tok.decode(ids[:cut], skip_special_tokens=True)
            res.append({"prompt_tokens": int(prompt_lens[i]), "output_tokens": cut, "text": text,
                        "finish": "stop" if finished else "length", "batch_seconds": dt / len(texts)})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--quant", default="none", choices=["none", "8bit", "4bit"])
    ap.add_argument("--no-translate", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N prompts")
    ap.add_argument("--qwen3-nothink", action="store_true")
    ap.add_argument("--only", default="", help="comma list of conditions: DIRECT,TRANSLATE_FIRST")
    args = ap.parse_args()

    out_dir = Path(args.out) / args.model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    for f in out_dir.glob("gens_*.jsonl"):
        for line in f.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                done.add(json.loads(line)["gen_id"])
    print(f"[resume] {len(done)} samples already on disk", flush=True)

    prompts = [json.loads(l) for l in Path(args.prompts).read_text(encoding="utf-8").split("\n") if l.strip()]
    if args.limit:
        prompts = prompts[: args.limit]
    only = set(filter(None, args.only.split(",")))

    tok = AutoTokenizer.from_pretrained(args.hf_id, revision=args.revision, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    qkw = {}
    if args.quant != "none":
        from transformers import BitsAndBytesConfig
        qkw["quantization_config"] = (BitsAndBytesConfig(load_in_8bit=True) if args.quant == "8bit" else
                                      BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                                         bnb_4bit_quant_type="nf4"))
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_id, revision=args.revision, torch_dtype=getattr(torch, args.dtype),
        device_map="auto", trust_remote_code=True, **qkw)
    model.eval()
    info = {"hf_id": args.hf_id, "revision": args.revision, "dtype": args.dtype, "quant": args.quant,
            "torch": torch.__version__, "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens,
            "n_samples": args.n_samples, "seed": args.seed, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    print(info, flush=True)

    shard = out_dir / f"gens_{int(time.time())}.jsonl"
    fh = open(shard, "a", encoding="utf-8")

    def emit(rec):
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        done.add(rec["gen_id"])

    def run_pass(items, condition):
        """items: list of dicts with prompt_text/effective_prompt/etc. Sorted by length for efficiency."""
        todo = [it for it in items
                if any(gid(args.model_id, it["task_id"], it["variant"], condition, s) not in done
                       for s in range(args.n_samples))]
        todo.sort(key=lambda it: len(it["effective_prompt"]))
        bs = args.batch_size
        i, bidx, t_start = 0, 0, time.time()
        while i < len(todo):
            batch = todo[i:i + bs]
            texts = [chat_text(tok, SYSTEM_CODE, it["effective_prompt"], args.qwen3_nothink) for it in batch]
            try:
                res = generate_batch(model, tok, texts, args.n_samples, True, args.max_new_tokens,
                                     args.temperature, args.top_p, args.seed + bidx)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                print(f"[oom] batch size -> {bs}", flush=True)
                continue
            for bi, it in enumerate(batch):
                for s in range(args.n_samples):
                    r = res[bi * args.n_samples + s]
                    code, status = extract_code(r["text"])
                    emit({
                        "gen_id": gid(args.model_id, it["task_id"], it["variant"], condition, s),
                        "model_id": args.model_id, "task_id": it["task_id"], "variant": it["variant"],
                        "condition": condition, "sample_idx": s, "seed": args.seed + bidx,
                        "prompt_text": it["prompt_text"], "effective_prompt": it["effective_prompt"],
                        "translation_tokens_in": it.get("tr_in", 0), "translation_tokens_out": it.get("tr_out", 0),
                        "signature_reinjected": it.get("sig_reinjected", False),
                        "prompt_tokens": r["prompt_tokens"], "output_tokens": r["output_tokens"],
                        "finish": r["finish"], "gen_seconds_share": r["batch_seconds"],
                        "raw_output": r["text"], "extracted_code": code, "extraction_status": status})
            i += len(batch)
            bidx += 1
            print(f"[{condition}] {i}/{len(todo)} prompts  bs={bs}  {time.time() - t_start:.0f}s", flush=True)

    # ---- DIRECT ----
    if not only or "DIRECT" in only:
        direct = [dict(p, effective_prompt=p["text"], prompt_text=p["text"]) for p in prompts]
        run_pass(direct, "DIRECT")

    # ---- TRANSLATE_FIRST ----
    if not args.no_translate and (not only or "TRANSLATE_FIRST" in only):
        tf_src = [p for p in prompts if p["variant"] in TF_VARIANTS]
        tr_path = out_dir / "translations.jsonl"
        trans = {}
        if tr_path.exists():
            for l in tr_path.read_text(encoding="utf-8").split("\n"):
                if l.strip():
                    r = json.loads(l); trans[(r["task_id"], r["variant"])] = r
        pending = [p for p in tf_src if (p["task_id"], p["variant"]) not in trans]
        pending.sort(key=lambda p: len(p["text"]))
        bs = max(1, args.batch_size)
        i = 0
        t0 = time.time()
        with open(tr_path, "a", encoding="utf-8") as tfh:
            while i < len(pending):
                batch = pending[i:i + bs]
                texts = [chat_text(tok, SYSTEM_TRANSLATE, p["text"], args.qwen3_nothink) for p in batch]
                try:
                    res = generate_batch(model, tok, texts, 1, False, 700, 0.0, 1.0, args.seed)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); bs = max(1, bs // 2); continue
                for p, r in zip(batch, res):
                    t = r["text"].strip()
                    reinj = False
                    sig = p["signature"].strip()
                    if sig not in t:
                        t = t + f"\n\n```python\n{sig}\n```"
                        reinj = True
                    rec = {"task_id": p["task_id"], "variant": p["variant"], "translation": t,
                           "tr_in": r["prompt_tokens"], "tr_out": r["output_tokens"], "sig_reinjected": reinj}
                    tfh.write(json.dumps(rec, ensure_ascii=False) + "\n"); tfh.flush()
                    trans[(p["task_id"], p["variant"])] = rec
                i += len(batch)
                print(f"[translate] {i}/{len(pending)}  {time.time() - t0:.0f}s", flush=True)
        items = []
        for p in tf_src:
            tr = trans[(p["task_id"], p["variant"])]
            items.append(dict(p, prompt_text=p["text"], effective_prompt=tr["translation"],
                              tr_in=tr["tr_in"], tr_out=tr["tr_out"], sig_reinjected=tr["sig_reinjected"]))
        run_pass(items, "TRANSLATE_FIRST")
    fh.close()
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
