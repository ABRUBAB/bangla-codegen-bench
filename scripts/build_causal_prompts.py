#!/usr/bin/env python3
"""Build the causal-control prompts (bcb/causal.py) -> data/prompts/prompts_causal.jsonl.
Token counts use the Qwen2.5-Coder tokenizer (the reference tokenizer of the study)."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bcb.causal import causal_prompts  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def main():
    tk = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
    count = lambda s: len(tk(s, add_special_tokens=False)["input_ids"])
    tasks = []
    for n in ("functional", "security"):
        tasks += [json.loads(x) for x in (ROOT / f"data/tasks/{n}.jsonl").read_text(encoding="utf-8").split("\n") if x.strip()]
    recs = []
    for t in tasks:
        recs += causal_prompts(t, count)
    out = ROOT / "data/prompts/prompts_causal.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")
    from collections import Counter
    c = Counter(r["variant"] for r in recs)
    pad = [abs(r["actual_tokens"] - r["target_tokens"]) / r["target_tokens"] for r in recs if r["variant"] == "EN_PAD"]
    print(dict(c), "| EN_PAD mean |token error| = %.1f%%" % (100 * sum(pad) / max(1, len(pad))))


if __name__ == "__main__":
    main()
