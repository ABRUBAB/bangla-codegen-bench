"""Interventional (causal) prompt controls that separate *language* from *token length*.

All variants use the English wrapper (instruction head/tail) so that the only thing that moves is the description:

  EN_PAD   English description + a requirement-free context paragraph, padded until the description has the same number of
           tokens as the formal-Bangla description under a reference tokenizer (token inflation without a language change).
  BN_DESC  formal-Bangla description inside the English wrapper (the 100 % end of the dose-response; BN additionally has a
           Bangla wrapper).
  HYB33 / HYB67  about one third / two thirds of the sentences of the English description are replaced by their formal-Bangla
           counterparts (sentence alignment is checked; tasks whose EN and BN sentence counts differ, or with < 3 sentences,
           are skipped). The sets are nested: HYB67 contains the sentences of HYB33.

If the Bangla penalty were a pure token-length effect, EN_PAD would lose as much accuracy as BN_DESC; if it is a language effect,
EN_PAD stays at the English level and the hybrids interpolate with the share of Bangla sentences."""
from __future__ import annotations

import random
import re
from typing import Callable, Dict, List, Optional

from .prompts import wrap

FILLER = [
    "This function belongs to a small utility module that several developers of the team maintain together.",
    "The module is imported by a few different services, and the same helper is sometimes reused in unrelated scripts.",
    "The surrounding code base is reviewed in the usual way before changes are merged, and the review notes are kept in the team's tracker.",
    "Nothing in this paragraph adds a requirement; it only describes the setting in which the function lives.",
    "Other helpers in the same module were written over several years by different people, so their style varies a little.",
    "The team keeps a short changelog for the module and updates it whenever a helper is added or renamed.",
    "Colleagues from other teams sometimes read the module to learn how the shared helpers are organised.",
    "Documentation for the module is generated automatically and published on the internal wiki.",
    "A new helper is normally added after a short discussion in the weekly meeting of the team.",
    "The module has no special dependencies beyond the tooling that every project of the team already uses.",
    "Questions about the module can be asked in the team's chat channel during working hours.",
    "The maintainers of the module rotate every few months, which is why the notes in the repository are kept up to date.",
]


def split_en(t: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z`(])", t.strip()) if s]


def split_bn(t: str) -> List[str]:
    return [s for s in re.split(r"(?<=[।.!?])\s+", t.strip()) if s]


def pad_english(en: str, target_tokens: int, count: Callable[[str], int], seed: str) -> Optional[str]:
    """English description with a context paragraph whose size brings the token count closest to target_tokens."""
    base = count(en)
    if target_tokens <= base:
        return None
    rnd = random.Random(seed)
    order = FILLER[:]
    rnd.shuffle(order)
    chosen: List[str] = []
    best, best_diff = en, abs(target_tokens - base)
    i = 0
    while i < 4 * len(order):
        chosen.append(order[i % len(order)] if i < len(order) else order[i % len(order)])
        text = "Context: " + " ".join(chosen) + "\n\n" + en
        diff = abs(count(text) - target_tokens)
        if diff < best_diff:
            best, best_diff = text, diff
        elif count(text) > target_tokens:
            break
        i += 1
    return best


def causal_prompts(task: dict, count: Callable[[str], int], seed_base: int = 42) -> List[dict]:
    d, sig, tid = task["description"], task["signature"], task["task_id"]
    recs: List[dict] = []

    def add(variant: str, text: str, **extra):
        recs.append({"task_id": tid, "variant": variant, "text": text, "generation": "causal_control",
                     "kind": task["kind"], "signature": sig, "function_name": task.get("function_name"), **extra})

    add("BN_DESC", wrap("en", sig, d["bn"]))
    padded = pad_english(d["en"], count(d["bn"]), count, f"{tid}-pad-{seed_base}")
    if padded is not None:
        add("EN_PAD", wrap("en", sig, padded), target_tokens=count(d["bn"]), actual_tokens=count(padded))
    en_s, bn_s = split_en(d["en"]), split_bn(d["bn"])
    n = len(en_s)
    if n >= 3 and len(bn_s) == n:
        order = list(range(n))
        random.Random(f"{tid}-{seed_base}").shuffle(order)
        for name, frac in (("HYB33", 1 / 3), ("HYB67", 2 / 3)):
            k = min(n - 1, max(1, round(frac * n)))
            swap = set(order[:k])
            text = " ".join(bn_s[i] if i in swap else en_s[i] for i in range(n))
            add(name, wrap("en", sig, text), n_sentences=n, n_bangla=k)
    return recs
