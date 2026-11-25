"""Build the prompt-preparation figure (stages of one example task) as a XeLaTeX source and PDF.

The figure shows task F039 at every preparation stage, using the real prompts from
data/prompts/prompts.jsonl, the AI-audit verdicts, the unit test and real outputs of
Qwen2.5-Coder 1.5B (data/generations) with their sandbox verdicts (data/results/exec.jsonl).

Usage (from the repo root):  python scripts/make_fig_preprocess.py
Needs XeLaTeX (MiKTeX/TeX Live) and the fonts Segoe UI, Nirmala UI (Bengali) and Consolas (Windows).
Output: figures/src/fig_preprocess.tex and figures/fig_preprocess.pdf
"""
from __future__ import annotations

import difflib
import glob
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = "F039"
MODEL = "qwen2.5-coder-1.5b"
BS = chr(92)  # backslash, kept out of literals on purpose


def esc(s: str) -> str:
    out = []
    for ch in s:
        if ch == BS:
            out.append(BS + "textbackslash{}")
        elif ch in "&%$#_{}":
            out.append(BS + ch)
        elif ch == "~":
            out.append(BS + "textasciitilde{}")
        elif ch == "^":
            out.append(BS + "textasciicircum{}")
        elif ch == "'":
            out.append(BS + "textquotesingle{}")
        else:
            out.append(ch)
    return "".join(out)


def is_bn(ch: str) -> bool:
    return "ঀ" <= ch <= "৿" or ch in "।॥‌‍"


def runs(text: str) -> str:
    """Wrap Bengali-script runs in the Bengali font; keep the rest in the main font."""
    out, cur, cur_bn = [], "", None
    for ch in text:
        b = is_bn(ch) if not ch.isspace() else cur_bn
        if cur_bn is None or b == cur_bn:
            cur += ch
            cur_bn = b if cur_bn is None else cur_bn
        else:
            out.append((cur_bn, cur))
            cur, cur_bn = ch, b
    if cur:
        out.append((cur_bn, cur))
    return "".join("{" + BS + "bnf " + esc(t) + "}" if b else esc(t) for b, t in out)


def token_tex(tok: str) -> str:
    parts = re.split(r"(`[^`]+`)", tok)
    res = []
    for p in parts:
        if not p:
            continue
        if p.startswith("`"):
            res.append(BS + "cd{" + esc(p[1:-1]) + "}")
        else:
            res.append(runs(p))
    return "".join(res)


def prose(text: str, changed: set[int] | None = None) -> str:
    toks = text.split()
    out = []
    for i, t in enumerate(toks):
        tt = token_tex(t)
        if changed and i in changed:
            tt = BS + "nz{" + tt + "}"
        out.append(tt)
    return " ".join(out)


def changed_tokens(a: str, b: str) -> set[int]:
    wa, wb = a.split(), b.split()
    sm = difflib.SequenceMatcher(a=wa, b=wb, autojunk=False)
    ch = set()
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "equal":
            ch.update(range(j1, j2))
    return ch


def code(lines: list[str]) -> str:
    rows = []
    for ln in lines:
        lead = len(ln) - len(ln.lstrip(" "))
        body = esc(ln.lstrip(" ")).replace(" ", BS + " ")
        rows.append(BS + "mbox{" + BS + "hspace*{" + f"{lead * 0.35:.2f}" + "em}" + body + "}")
    return "{" + BS + "cfont " + (BS + BS + "\n").join(rows) + "}"


def split_prompt(text: str):
    """wrapper sentence, signature, description, closing instruction"""
    head, rest = text.split("```python\n", 1)
    sig, tail = rest.split("\n```\n\n", 1)
    paras = tail.split("\n\n")
    return head.strip(), sig.strip(), paras[0].strip(), paras[-1].strip() if len(paras) > 1 else ""


def main() -> None:
    P = {r["variant"]: r["text"] for r in map(json.loads, open(ROOT / "data/prompts/prompts.jsonl", encoding="utf-8"))
         if r["task_id"] == TASK}
    task = [json.loads(l) for l in open(ROOT / "data/tasks/functional.jsonl", encoding="utf-8") if f'"{TASK}"' in l][0]
    gens = {}
    for f in sorted(glob.glob(str(ROOT / f"data/generations/{MODEL}/gens_*.jsonl"))):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if r.get("task_id") == TASK and r.get("condition") == "DIRECT":
                gens[(r["variant"], r["sample_idx"])] = r
    ex = {}
    for l in open(ROOT / "data/results/exec.jsonl", encoding="utf-8"):
        if f'"{TASK}"' in l and MODEL in l:
            r = json.loads(l)
            ex[r["gen_id"]] = r

    def npass(v):
        return sum(bool(ex[g["gen_id"]]["functional_pass"]) for (vv, _), g in gens.items() if vv == v)

    d = {v: split_prompt(P[v]) for v in P}
    tests = [ln.strip() for ln in task["functional_tests"].splitlines() if ln.strip().startswith("assert")]
    en_code = gens[("EN", 0)]["extracted_code"].rstrip().splitlines()
    bn_code = gens[("BN", 0)]["extracted_code"].rstrip().splitlines()
    bn_code = [ln for ln in bn_code if ln.strip() and not ln.strip().startswith("#")]
    bn_tail = ex[gens[("BN", 0)]["gen_id"]]["func_tail"]
    m = re.search(r"AssertionError: (assert .*?)\n", bn_tail)
    bn_err = m.group(1) if m else "AssertionError"
    p_en, p_bn = npass("EN"), npass("BN")
    n_en = sum(1 for (v, _) in gens if v == "EN")

    audit = {}
    for f in glob.glob(str(ROOT / "data/audit/llm_opus_*.json")):
        dd = json.load(open(f, encoding="utf-8"))
        items = dd if isinstance(dd, list) else dd.get("items", [])
        for it in items:
            if isinstance(it, dict) and it.get("task_id") == TASK:
                audit[it["rendering"]] = it
    all_same = all(audit[k]["meaning"] == "Y" for k in ("bn", "bn_mix", "banglish", "es"))
    nat = sorted({audit[k]["naturalness"] for k in ("bn", "bn_mix", "banglish", "es")})
    verbatim = sorted(set(re.findall(r"`([^`]+)`", d["EN"][2])), key=lambda s: d["EN"][2].index("`" + s + "`"))
    ok_verb = all(all(("`" + t + "`") in d[v][2] for t in verbatim) for v in ("BN", "BN_MIX", "BANGLISH", "ES"))
    assert all_same and ok_verb

    L = []
    a = L.append
    a(BS + r"documentclass[border=0pt]{standalone}")
    a(BS + r"usepackage{fontspec}")
    a(BS + r"setmainfont{Segoe UI}")
    a(BS + r"newfontfamily" + BS + r"bnf{Nirmala UI}[Script=Bengali,Scale=0.97]")
    a(BS + r"newfontfamily" + BS + r"cfont{Consolas}[Scale=0.95]")
    a(BS + r"usepackage{xcolor}")
    a(BS + r"usepackage[most]{tcolorbox}")
    a(BS + r"usetikzlibrary{arrows.meta,calc}")
    a(BS + r"definecolor{hd}{HTML}{1F3A5F}" + BS + r"definecolor{bg}{HTML}{F4F6F9}" + BS + r"definecolor{nzc}{HTML}{C0392B}")
    a(BS + r"definecolor{ok}{HTML}{1E7B34}" + BS + r"definecolor{lab}{HTML}{55606E}" + BS + r"definecolor{codebg}{HTML}{FFFFFF}")
    a(BS + r"newcommand{" + BS + r"cd}[1]{{" + BS + r"cfont #1}}")
    a(BS + r"newcommand{" + BS + r"nz}[1]{" + BS + r"textcolor{nzc}{#1}}")
    a(BS + r"newcommand{" + BS + r"lb}[1]{{" + BS + r"color{lab}" + BS + r"bfseries #1}}")
    a(BS + r"newfontfamily" + BS + r"symf{Segoe UI Symbol}")
    a(BS + r"newcommand{" + BS + r"okm}{" + BS + r"textcolor{ok}{{" + BS + "symf ✓}}~}")
    a(BS + r"newtcolorbox{stage}[3]{enhanced,valign=top,halign=flush left,colback=white,colframe=hd!70,boxrule=0.5pt,arc=1.2mm,"
      r"left=1.4mm,right=1.4mm,top=1.0mm,bottom=1.0mm,fonttitle=" + BS + r"bfseries" + BS + r"footnotesize,coltitle=white,colbacktitle=hd,"
      r"title={#2},remember as=#1,fontupper=" + BS + r"fontsize{6.6}{8.1}" + BS + r"selectfont}")
    a(BS + r"newtcolorbox{sub}[1][]{enhanced,colback=bg,colframe=bg,boxrule=0pt,arc=0.6mm,left=0.9mm,right=0.9mm,top=0.5mm,bottom=0.6mm,before skip=0.8mm,after skip=0.5mm,halign=flush left,#1}")
    a(BS + r"setlength{" + BS + r"parindent}{0pt}")
    a(BS + r"begin{document}")
    a(BS + r"begin{tcbraster}[raster columns=3,raster equal height=rows,raster column skip=4.6mm,raster row skip=7mm,raster width=173.8mm]")
    H1, H2 = "71mm", "66mm"

    # ---- Stage 1
    a(BS + r"begin{stage}{s1}{1\quad Task specification (English)}{" + H1 + "}")
    a(BS + r"lb{Signature} (identical in every version)")
    a(BS + r"begin{sub}" + code([d["EN"][1]]) + BS + r"end{sub}")
    a(BS + r"lb{Description}")
    a(BS + r"begin{sub}" + prose(d["EN"][2]) + BS + r"end{sub}")
    a(BS + r"lb{Unit test} (pytest)")
    a(BS + r"begin{sub}" + code(tests) + BS + r"end{sub}")
    a(BS + r"lb{Reference solution}")
    a(BS + r"begin{sub}" + code(task["reference_solution"].strip().splitlines()) + BS + r"end{sub}")
    a(BS + r"end{stage}")

    # ---- Stage 2
    a(BS + r"begin{stage}{s2}{2\quad Translation into four versions}{" + H1 + "}")
    for v, nm in (("BN", "Formal Bangla (BN)"), ("BN_MIX", "Code-mixed Bangla (BN\\_MIX)"),
                  ("BANGLISH", "Romanized Bangla (BANGLISH)"), ("ES", "Spanish control (ES)")):
        a(BS + "lb{" + nm + "}")
        a(BS + r"begin{sub}" + prose(d[v][2]) + BS + r"end{sub}")
    a(BS + r"end{stage}")

    # ---- Stage 3
    a(BS + r"begin{stage}{s3}{3\quad Validation and audit}{" + H1 + "}")
    a(BS + r"lb{Automatic gate}")
    a(BS + r"begin{sub}" + BS + "okm Script check: Bengali in BN and BN" + BS + "_MIX, Latin in BANGLISH and ES." + BS + BS +
      BS + "okm Every code item of the English text is kept verbatim in all four versions: " +
      ", ".join(BS + "cd{" + esc(t) + "}" for t in verbatim) + "." + BS + r"end{sub}")
    a(BS + r"lb{AI audit} (Claude Opus, all 852 versions)")
    a(BS + r"begin{sub}" + BS + "okm Meaning compared with English: same requirements in all four versions; naturalness "
      + "/".join(str(x) for x in nat) + " of 5." + BS + BS + "Proposed corrections are applied only after passing the gate again." + BS + r"end{sub}")
    a(BS + r"lb{Oracle gate} (sandbox)")
    a(BS + r"begin{sub}" + BS + "okm The reference solution passes the unit test in 3 of 3 runs. (Security tasks: the secure "
      "reference must also resist the exploit test and the insecure reference must fail it.)" + BS + r"end{sub}")
    a(BS + r"lb{Native-speaker check} (both authors)")
    a(BS + r"begin{sub}" + BS + "okm All three Bangla versions read against the English text: no requirement added, dropped or changed."
      + BS + r"end{sub}")
    a(BS + r"end{stage}")

    # ---- Stage 4
    a(BS + r"begin{stage}{s4}{4\quad Noise injection (rule-based)}{" + H2 + "}")
    a(r"{" + BS + r"color{lab}Edited words in " + BS + r"nz{red}; code items are protected.}")
    for v, base, nm in (("EN_TYPO", "EN", "EN\\_TYPO: 5\\% keyboard typos"), ("BN_SPELL", "BN", "BN\\_SPELL: Bangla spelling confusions"),
                        ("BN_TYPO", "BN", "BN\\_TYPO: 5\\% grapheme-level typos")):
        ch = changed_tokens(d[base][2], d[v][2])
        txt = d[v][2]
        if v == "BN_TYPO":  # show the first sentence only (up to the danda)
            cut = txt.index("।") + 1
            txt = txt[:cut] + " ..."
        a(BS + "lb{" + nm + "}")
        a(BS + r"begin{sub}" + prose(txt, ch) + BS + r"end{sub}")
    a(BS + r"lb{Control prompts}")
    a(BS + r"begin{sub}ID" + BS + "_ONLY: signature only, no description. EN" + BS + "_PAD: English padded to the token count of the Bangla text. "
      + BS + "textit{x}" + BS + "_HINT: security reminder added (security tasks)." + BS + r"end{sub}")
    a(BS + r"end{stage}")

    # ---- Stage 5
    sysmsg = ("You are an expert Python programmer. Return only a single Python code block solving the requested task. "
              "Do not include extra conversational text.")
    a(BS + r"begin{stage}{s5}{5\quad Prompt assembly (chat template)}{" + H2 + "}")
    a(BS + r"lb{System message} (fixed, English)")
    a(BS + r"begin{sub}" + esc(sysmsg) + BS + r"end{sub}")
    a(BS + r"lb{User message} (formal Bangla example)")
    a(BS + r"begin{sub}" + prose(d["BN"][0]) + BS + BS + "[1pt]" + code(["```python", d["BN"][1], "```"]) + BS + BS + "[2pt]"
      + "{" + BS + r"color{lab}" + BS + r"itshape [formal Bangla description from stage 2]}" + BS + BS + "[2pt]" + prose(d["BN"][3]) + BS + r"end{sub}")
    a(BS + r"lb{What changes between versions}")
    a(BS + r"begin{sub}Only the natural-language text (opening line, description, closing line). The system message, the "
      "signature and every code item are identical in all versions." + BS + r"end{sub}")
    a(BS + r"lb{Decoding}")
    a(BS + r"begin{sub}Temperature 0.2, top-" + BS + r"textit{p} 0.95, 3 samples per prompt, at most 768 new tokens, thinking disabled."
      + BS + r"end{sub}")
    a(BS + r"end{stage}")

    # ---- Stage 6
    key_en = [ln for ln in en_code if "return" in ln][-1]
    key_bn = [ln for ln in bn_code if "return" in ln][-1]
    a(BS + r"begin{stage}{s6}{6\quad Generation, extraction, execution}{" + H2 + "}")
    a(r"{" + BS + r"color{lab}Model: Qwen2.5-Coder 1.5B. The largest fenced code block is extracted and its unit test is run in the sandbox.}" + BS + BS + "[1pt]")
    a(BS + r"lb{English prompt} " + BS + r"hfill" + BS + r"textcolor{ok}{" + BS + rf"bfseries {p_en} of {n_en} samples pass}}")
    a(BS + r"begin{sub}" + code(en_code) + BS + r"end{sub}")
    a(BS + r"lb{Formal Bangla prompt} " + BS + r"hfill" + BS + r"textcolor{nzc}{" + BS + rf"bfseries {p_bn} of {n_en} samples pass}}")
    a(BS + r"begin{sub}" + code(bn_code[:1] + ["  ..."] + ["  " + ln.strip() for ln in bn_code if "return" in ln][-1:]) + BS + r"end{sub}")
    a(BS + r"begin{sub}[colback=nzc!8,colframe=nzc!8]" + BS + r"cd{" + esc(bn_err) + "}" + BS + BS +
      r"The negation at the end of the Bangla sentence (" + BS + r"bnf{" + esc("থাকে না") + "}, `is not present') was missed: "
      r"the model adds the " + BS + r"cd{0b} prefix." + BS + r"end{sub}")
    a(BS + r"end{stage}")
    a(BS + r"end{tcbraster}")
    # arrows
    a(BS + r"begin{tikzpicture}[remember picture,overlay,>={Stealth[length=2.2mm]},line width=0.9pt,draw=hd!80]")
    a(BS + r"draw[->] ($(s1.east)+(0.2mm,0)$) -- ($(s2.west)+(-0.2mm,0)$);")
    a(BS + r"draw[->] ($(s2.east)+(0.2mm,0)$) -- ($(s3.west)+(-0.2mm,0)$);")
    a(BS + r"draw[->] ($(s4.east)+(0.2mm,0)$) -- ($(s5.west)+(-0.2mm,0)$);")
    a(BS + r"draw[->] ($(s5.east)+(0.2mm,0)$) -- ($(s6.west)+(-0.2mm,0)$);")
    a(BS + r"draw[->,rounded corners=2mm] ($(s3.south)+(0,-0.3mm)$) -- ++(0,-2.1mm) -| ($(s4.north)+(0,0.3mm)$);")
    a(BS + r"end{tikzpicture}")
    a(BS + r"end{document}")

    src = ROOT / "figures/src"
    src.mkdir(parents=True, exist_ok=True)
    tex = src / "fig_preprocess.tex"
    tex.write_text("\n".join(L) + "\n", encoding="utf-8")
    for _ in range(3):
        subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex.name], cwd=src, check=True,
                       stdout=subprocess.DEVNULL)
    shutil.copy(src / "fig_preprocess.pdf", ROOT / "figures/fig_preprocess.pdf")
    print("wrote figures/fig_preprocess.pdf", {"EN_pass": p_en, "BN_pass": p_bn, "n": n_en})


if __name__ == "__main__":
    main()
