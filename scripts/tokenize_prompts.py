#!/usr/bin/env python3
"""Token-count every prompt with real tokenizers (RQ3: tokenization cost).

For each (prompt, tokenizer): tokens of the full user prompt and of the natural-language
description alone (isolates the language effect from the shared signature/wrapper), plus
chars / UTF-8 bytes / grapheme clusters / whitespace words of the description.

Output: data/results/token_counts.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import regex
import tiktoken
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bcb.tasks import load_tasks, TASK_DIR  # noqa: E402

HF_TOKENIZERS = {
    "qwen2.5": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen3": "Qwen/Qwen3-8B",
    "deepseek-coder": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "deepseek-v2": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "tigercoder": "md-nishat-008/TigerCoder-1B",
    "llama3": "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    # current-generation open models (2026)
    "qwen3.8": "Qwen/Qwen3.8-27B",
    "gemma4": "google/gemma-4-31B-it",
    "ministral3": "mistralai/Ministral-3-14B-Instruct-2512",
    "glm4.7": "zai-org/GLM-4.7-Flash",
}
TIKTOKEN = {"gpt-4 (cl100k)": "cl100k_base", "gpt-4o (o200k)": "o200k_base"}


def main():
    prompts = [json.loads(l) for l in (ROOT / "data/prompts/prompts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    tasks = {t["task_id"]: t for t in load_tasks([p for p in TASK_DIR.glob("*.jsonl") if not p.name.startswith("_")])}
    lang_key = {"EN": "en", "EN_TYPO": "en", "BN": "bn", "BN_SPELL": "bn", "BN_TYPO": "bn", "BN_MIX": "bn_mix",
                "BANGLISH": "banglish", "ES": "es", "ES_TYPO": "es"}

    encoders = {}
    for name, enc in TIKTOKEN.items():
        e = tiktoken.get_encoding(enc)
        encoders[name] = lambda s, e=e: len(e.encode(s, disallowed_special=()))
    for name, hf in HF_TOKENIZERS.items():
        try:
            kw = {"fix_mistral_regex": True} if name == "ministral3" else {}  # known regex bug in the published tokenizer
            tk = AutoTokenizer.from_pretrained(hf, trust_remote_code=True, **kw)
            encoders[name] = lambda s, tk=tk: len(tk.encode(s, add_special_tokens=False))
            print("loaded tokenizer", name, "vocab", getattr(tk, "vocab_size", "?"), flush=True)
        except Exception as exc:
            print("SKIP tokenizer", name, type(exc).__name__, str(exc)[:120], flush=True)

    rows = []
    for p in prompts:
        base = p["variant"].replace("_HINT", "")
        desc = None
        if base in lang_key and not p["variant"].endswith("_HINT"):
            d = tasks[p["task_id"]]["description"][lang_key[base]]
            desc = d
            # reconstruct the description as it appears after perturbation: take the prose paragraph
            # between the code block and the final instruction line
            parts = p["text"].split("```")
            if len(parts) >= 3:
                prose = parts[2].strip().rsplit("\n\n", 1)[0].strip()
                desc = prose
        rec = {"task_id": p["task_id"], "variant": p["variant"], "kind": p["kind"],
               "chars_full": len(p["text"]), "bytes_full": len(p["text"].encode("utf-8"))}
        if desc is not None:
            rec.update(desc_chars=len(desc), desc_bytes=len(desc.encode("utf-8")),
                       desc_graphemes=len(regex.findall(r"\X", desc)), desc_words=len(desc.split()))
        for name, f in encoders.items():
            rec[f"full__{name}"] = f(p["text"])
            if desc is not None:
                rec[f"desc__{name}"] = f(desc)
        rows.append(rec)
    out = ROOT / "data/results/token_counts.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    print("wrote", out, len(rows), "rows;", len(encoders), "tokenizers")


if __name__ == "__main__":
    main()
