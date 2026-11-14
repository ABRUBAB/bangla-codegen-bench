#!/usr/bin/env python3
"""Automated translation validation (runs on Kaggle: Google APIs are blocked locally).

For every task and each rendering r in {bn, bn_mix, banglish}:
  1. back-translation of the rendering into English by judge model A (temperature 0)
  2. two independent LLM judges (model A and model B) read the English original and the rendering and return
     JSON {verdict: equivalent|minor_difference|major_difference, naturalness: 1-5, issues: [...]}
Output jsonl rows: task_id, rendering, back_translation, judge_<A|B>: {...}
These are *automatic* checks and are reported as such; they do not replace the native-speaker audit.
"""
import argparse, json, re, time, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

BASE = "https://generativelanguage.googleapis.com/v1beta/models"
SYS_BT = ("You are a precise technical translator. Translate the following programming-task description into "
          "English as literally and completely as possible. Keep `backticked` items unchanged. Output only the "
          "English text.")
SYS_JUDGE = ("You are a bilingual (Bangla/English) software-engineering reviewer. You get an ENGLISH task description "
             "and a RENDERING of it in Bangla, code-mixed Bangla or romanized Bangla (Banglish). Decide whether the "
             "rendering imposes exactly the same requirements: nothing added, nothing omitted, no changed constraint, "
             "number, literal, ordering or edge-case behaviour. Also rate how natural it sounds to a Bangladeshi "
             "developer (1=unnatural machine translation, 5=fully natural). Reply with ONLY compact JSON: "
             '{"verdict":"equivalent|minor_difference|major_difference","naturalness":<1-5>,"issues":["..."]}')


def call(model, key, system, user, max_tokens=1200, retries=8):
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": max_tokens,
                                 "thinkingConfig": {"thinkingBudget": 0}}}
    for a in range(retries):
        try:
            r = requests.post(f"{BASE}/{model}:generateContent", params={"key": key}, json=body, timeout=120)
            if r.status_code == 200:
                c = (r.json().get("candidates") or [{}])[0]
                return "".join(p.get("text", "") for p in (c.get("content") or {}).get("parts", []))
            if r.status_code == 400 and "thinking" in r.text.lower():
                body["generationConfig"].pop("thinkingConfig", None); continue
            time.sleep(min(60, 2 ** a))
        except requests.RequestException:
            time.sleep(min(60, 2 ** a))
    return ""


def parse_json(t):
    m = re.search(r"\{.*\}", t, re.S)
    try:
        return json.loads(m.group(0)) if m else {"verdict": "parse_error"}
    except Exception:
        return {"verdict": "parse_error", "raw": t[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--model-a", default="gemini-3.8-flash")
    ap.add_argument("--model-b", default="gemini-3.5-flash")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    key = Path(a.key_file).read_text().strip()
    tasks = [json.loads(l) for l in Path(a.tasks).read_text(encoding="utf-8").splitlines() if l.strip()]
    items = [(t["task_id"], r, t["description"]["en"], t["description"][r]) for t in tasks for r in ("bn", "bn_mix", "banglish")]
    lock = threading.Lock()
    fh = open(a.out, "w", encoding="utf-8")

    def work(it):
        tid, r, en, cand = it
        bt = call(a.model_a, key, SYS_BT, cand).strip()
        prompt = f"ENGLISH:\n{en}\n\nRENDERING ({r}):\n{cand}"
        ja = parse_json(call(a.model_a, key, SYS_JUDGE, prompt))
        jb = parse_json(call(a.model_b, key, SYS_JUDGE, prompt))
        rec = {"task_id": tid, "rendering": r, "en": en, "candidate": cand, "back_translation": bt,
               "judge_a": ja, "judge_b": jb, "model_a": a.model_a, "model_b": a.model_b}
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, items))
    fh.close()
    print("done", len(items))


if __name__ == "__main__":
    main()
