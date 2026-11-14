#!/usr/bin/env python3
"""White-box explainability probes on an open code LLM (run on a GPU, e.g. Kaggle T4x2).

For every (task, variant) in {EN, BN, BN_MIX, BANGLISH, ID_ONLY, ...} the chat prompt is followed by the
*reference solution* (teacher forcing). We compute
  1. nll          mean token NLL of the reference solution given the prompt  (a likelihood-based
                  comprehension/generation-difficulty proxy; no sampling noise)
  2. attribution  gradient x input saliency of that NLL w.r.t. the prompt-token embeddings, aggregated
                  per prompt segment: head | signature | description | tail  (L1 share of |g . x|)
  3. hidden       last-prompt-token hidden state at every layer and mean-pooled description-token state,
                  to measure cross-lingual representation alignment (cosine vs EN of the same task)
Outputs (npz/jsonl): xai_<model>.jsonl (per prompt scalars + segment shares), xai_<model>_hidden.npz,
                     xai_<model>_tokens_<task>_<variant>.json (token-level saliency for example tasks)
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = ("You are an expert Python programmer. Return only a single Python code block solving the "
          "requested task. Do not include extra conversational text.")
VARIANTS = ["EN", "EN_TYPO", "BN", "BN_SPELL", "BN_MIX", "BN_TYPO", "BANGLISH", "ES", "ID_ONLY"]
EXAMPLES = {"F025", "F007", "S001", "S011"}


def segments(user_text: str):
    """Character spans (start, end, name) inside the user prompt."""
    spans = []
    m = re.search(r"```python\n.*?```", user_text, re.S)
    if m:
        spans.append((0, m.start(), "head"))
        spans.append((m.start(), m.end(), "signature"))
        rest = user_text[m.end():]
        parts = rest.rstrip().rsplit("\n\n", 1)
        if len(parts) == 2:
            d_end = m.end() + len(parts[0])
            spans.append((m.end(), d_end, "description"))
            spans.append((d_end, len(user_text), "tail"))
        else:
            spans.append((m.end(), len(user_text), "tail"))
    else:
        spans.append((0, len(user_text), "description"))
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--tasks", required=True, help="jsonl with task_id, reference_solution")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=1400)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    variants = a.variants.split(",")

    tasks = {}
    for l in Path(a.tasks).read_text(encoding="utf-8").splitlines():
        if l.strip():
            t = json.loads(l); tasks[t["task_id"]] = t
    prompts = [json.loads(l) for l in Path(a.prompts).read_text(encoding="utf-8").splitlines() if l.strip()]
    prompts = [p for p in prompts if p["variant"] in variants]
    if a.limit:
        prompts = prompts[: a.limit]

    tok = AutoTokenizer.from_pretrained(a.hf_id, revision=a.revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.hf_id, revision=a.revision, torch_dtype=torch.float16,
                                                 device_map="auto", trust_remote_code=True)
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    emb_layer = model.get_input_embeddings()
    dev = emb_layer.weight.device

    rows, hid_last, hid_desc, keys = [], [], [], []
    t0 = time.time()
    for n, p in enumerate(prompts):
        tid, var = p["task_id"], p["variant"]
        ref = tasks[tid]["reference_solution"].strip("\n")
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p["text"]}]
        prompt_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        sol_str = f"```python\n{ref}\n```"
        full = prompt_str + sol_str
        enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc["input_ids"][: a.max_len]
        offs = enc["offset_mapping"][: a.max_len]
        n_prompt = len(tok(prompt_str, add_special_tokens=False)["input_ids"])
        if len(ids) <= n_prompt + 1:
            continue
        # segment id per prompt token via character offsets
        u0 = prompt_str.find(p["text"])
        segs = segments(p["text"])
        seg_of = []
        for (cs, ce) in offs[:n_prompt]:
            name = "template"
            if cs >= u0 and ce <= u0 + len(p["text"]):
                rel = cs - u0
                for s0, s1, nm in segs:
                    if s0 <= rel < s1:
                        name = nm
                        break
            seg_of.append(name)
        input_ids = torch.tensor([ids], device=dev)
        emb = emb_layer(input_ids).detach().clone().requires_grad_(True)
        outp = model(inputs_embeds=emb, output_hidden_states=True)
        logits = outp.logits[0, :-1].float()
        tgt = input_ids[0, 1:].to(logits.device)
        nll_tok = torch.nn.functional.cross_entropy(logits, tgt, reduction="none")
        sol_nll = nll_tok[n_prompt - 1:]
        loss = sol_nll.mean()
        loss.backward()
        g = emb.grad[0, :n_prompt].float()
        x = emb[0, :n_prompt].detach().float()
        attr = (g * x).sum(-1)                      # signed gradient x input
        mag = attr.abs()
        seg_mass = {}
        for nm in ("head", "signature", "description", "tail", "template"):
            idx = [i for i, s in enumerate(seg_of) if s == nm]
            seg_mass[nm] = float(mag[idx].sum()) if idx else 0.0
        tot = sum(v for k, v in seg_mass.items() if k != "template") or 1.0
        row = {"task_id": tid, "variant": var, "nll": float(loss), "n_prompt_tokens": n_prompt,
               "n_solution_tokens": int(sol_nll.numel())}
        row.update({f"attr_{k}": v / tot for k, v in seg_mass.items() if k != "template"})
        n_per = {k: sum(1 for s in seg_of if s == k) for k in ("head", "signature", "description", "tail")}
        row.update({f"ntok_{k}": v for k, v in n_per.items()})
        rows.append(row)
        # hidden states: last prompt token and mean over description tokens, all layers
        hs = torch.stack([h[0] for h in outp.hidden_states]).float().detach()   # L+1, T, D
        last = hs[:, n_prompt - 1, :].cpu().numpy()
        didx = [i for i, s in enumerate(seg_of) if s == "description"]
        dmean = hs[:, didx, :].mean(1).cpu().numpy() if didx else np.zeros_like(last)
        hid_last.append(last.astype(np.float16)); hid_desc.append(dmean.astype(np.float16)); keys.append(f"{tid}|{var}")
        if tid in EXAMPLES and var in ("EN", "BN", "BN_MIX", "BANGLISH"):
            toks = tok.convert_ids_to_tokens(ids[:n_prompt])
            (out / f"tokens_{a.model_id}_{tid}_{var}.json").write_text(json.dumps({
                "tokens": toks, "segments": seg_of, "attr": [float(v) for v in attr.cpu()], "nll": float(loss)},
                ensure_ascii=False))
        del outp, emb, hs, logits
        torch.cuda.empty_cache()
        if n % 25 == 0:
            print(f"{n}/{len(prompts)}  {time.time() - t0:.0f}s  nll={float(loss):.3f}", flush=True)
    with open(out / f"xai_{a.model_id}.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    np.savez_compressed(out / f"xai_{a.model_id}_hidden.npz", keys=np.array(keys),
                        last=np.stack(hid_last), desc=np.stack(hid_desc))
    print("[done]", len(rows), "prompts")


if __name__ == "__main__":
    main()
