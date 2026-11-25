#!/usr/bin/env python3
"""Build the workflow diagram (Figure 1) as a native, editable PowerPoint slide (python-pptx), in the three-band style of the
authors' reference figure (serif type, numbered dashed bands, thin outlined boxes, condition chips, line icons).
Numbers are read from the generated data files, so the diagram always matches the paper.
Export to PNG/PDF with scripts/export_pptx.ps1 (PowerPoint COM)."""
import csv
import json
import math
import re
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
WN = json.loads((ROOT / "figures/workflow_numbers.json").read_text()) if (ROOT / "figures/workflow_numbers.json").exists() else {}


def macro(key, default):
    p = ROOT / "paper/numbers.tex"
    if not p.exists():
        return default
    m = re.search(r"csname n@" + re.escape(key) + r"\\endcsname\{([^}]*)\}", p.read_text(encoding="utf-8"))
    return m.group(1) if m else default


def token_ratio(prefix, default):
    p = ROOT / "data/results/tab_token_ratio_median.csv"
    if not p.exists():
        return default
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r[""].startswith(prefix):
            return float(r["BN"])
    return default


N_TASKS, N_FUNC, N_SEC, N_CWE = WN.get("n_tasks", 213), WN.get("n_func", 138), WN.get("n_sec", 75), WN.get("n_cwe", 20)
N_PROMPTS, N_MODELS, N_TOK = WN.get("n_prompts", 2505), WN.get("n_models", 12), WN.get("n_tokenizers", 13)
N_EXEC = macro("nExec", "133,020")
SPDF = macro("sd-spdf-rel", "80")
TOKR = {"ds": token_ratio("DeepSeek-Coder (32k)", 8.3), "qw": token_ratio("Qwen2.5", 3.9),
        "g4o": token_ratio("GPT-4o", 1.5), "gm": token_ratio("Gemma-4", 1.2)}

# --- palette (close to the reference figure) ---------------------------------------------------
INK, TXT, GREY = "1B2540", "222222", "7A7A7A"
BLUE, DBLUE = "2B5FA8", "1F4E91"
RED, DRED = "C8262E", "B3202A"
GREEN, DGREEN = "1E6B45", "1E6B45"
TEAL, BRED, BBLUE = "1CA9A0", "D93636", "2D6CC0"
CHIP = {"EN": "2F6FDE", "EN_TYPO": "1FA3C7", "BN": "2BA05A", "BN_SPELL": "8DB32B", "BN_MIX": "F2A20F",
        "BN_TYPO": "E0562F", "BANGLISH": "7B4BC6", "ES": "C23B8E"}
FONT = "Times New Roman"
FS = 1.10  # global font scale (legibility at column-width reproduction)

W, H = 9.8, 7.66
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
slide = prs.slides.add_slide(prs.slide_layouts[6])
sh = slide.shapes


def rgb(h):
    return RGBColor.from_string(h)


# ================================================================================ primitives
def _line(shape, color, lw, dash=None):
    if color is None:
        shape.line.fill.background()
        return
    shape.line.color.rgb = rgb(color)
    shape.line.width = Pt(lw)
    if dash is not None:
        shape.line.dash_style = dash


def _fill(shape, fill):
    if fill is None:
        shape.fill.background()
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb = rgb(fill)


def fill_text(tf, paras, anchor="t", margins=(0, 0, 0, 0)):
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left, tf.margin_top, tf.margin_right, tf.margin_bottom = [Inches(m) for m in margins]
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    first = True
    for p in paras:
        para = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        para.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[p.get("align", "l")]
        para.space_before = Pt(0)
        para.space_after = Pt(p.get("after", 0))
        if p.get("ls"):
            para.line_spacing = p["ls"]
        runs = p["runs"] if "runs" in p else [(p["t"], {})]
        for t, f in runs:
            r = para.add_run()
            r.text = t
            fo = r.font
            fo.name = f.get("font", p.get("font", FONT))
            fo.size = Pt(f.get("s", p.get("s", 8)) * FS)
            fo.bold = f.get("b", p.get("b", False))
            fo.italic = f.get("i", p.get("i", False))
            fo.color.rgb = rgb(f.get("c", p.get("c", TXT)))
            if f.get("sub"):
                r.font._element.set("baseline", "-25000")
        if p.get("bullet"):
            pPr = para._p.get_or_add_pPr()
            ind = int(Inches(p.get("bi", 0.10)))
            pPr.set("marL", str(ind))
            pPr.set("indent", str(-ind))
            bf = etree.SubElement(pPr, qn("a:buFont"))
            bf.set("typeface", "Arial")
            bc = etree.SubElement(pPr, qn("a:buChar"))
            bc.set("char", "•")


def text(x, y, w, h, paras, anchor="t"):
    tb = sh.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    fill_text(tb.text_frame, paras, anchor)
    return tb


def rrect(x, y, w, h, line=None, fill="FFFFFF", lw=1.0, r=0.07, dash=None, paras=None, anchor="m", margins=(0.04, 0.0, 0.04, 0.0)):
    s = sh.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = max(0.0, min(0.5, r / min(w, h)))
    _fill(s, fill)
    _line(s, line, lw, dash)
    s.shadow.inherit = False
    if paras:
        fill_text(s.text_frame, paras, anchor, margins)
    return s


def oval(x, y, w, h, fill=None, line=None, lw=1.0, paras=None):
    s = sh.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(w), Inches(h))
    _fill(s, fill)
    _line(s, line, lw)
    s.shadow.inherit = False
    if paras:
        fill_text(s.text_frame, paras, "m")
    return s


def line(x1, y1, x2, y2, color=INK, lw=1.0, dash=None, arrow=False, cap_round=False):
    c = sh.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(lw)
    if dash is not None:
        c.line.dash_style = dash
    ln = c.line._get_or_add_ln()
    if cap_round:
        ln.set("cap", "rnd")
    if arrow:
        t = etree.SubElement(ln, qn("a:tailEnd"))
        t.set("type", "triangle")
        t.set("w", "med")
        t.set("len", "med")
    return c


def poly(pts, close=True, fill=None, color=INK, lw=1.5):
    e = [(int(Inches(px)), int(Inches(py))) for px, py in pts]
    fb = sh.build_freeform(e[0][0], e[0][1], scale=1.0)
    fb.add_line_segments(e[1:], close=close)
    s = fb.convert_to_shape()
    _fill(s, fill)
    _line(s, color, lw)
    ln = s.line._get_or_add_ln()
    ln.set("cap", "rnd")
    etree.SubElement(ln, qn("a:round"))
    s.shadow.inherit = False
    return s


def bez(p0, p1, p2, n=14):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0], (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1])
            for t in [i / n for i in range(1, n + 1)]]


def arc(cx, cy, rx, ry, a0, a1, n=18):
    return [(cx + rx * math.cos(math.radians(a0 + (a1 - a0) * i / n)), cy - ry * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]


# ================================================================================ composite pieces
def badge(cx, cy, d, color, label, size=12):
    oval(cx - d / 2, cy - d / 2, d, d, fill=color, paras=[dict(t=label, b=True, s=size, c="FFFFFF", align="c")])


def band(x, y, w, h, color, tcolor, num, title):
    rrect(x, y, w, h, line=color, fill=None, lw=1.25, r=0.10, dash=MSO_LINE.DASH)
    badge(x + 0.26, y + 0.25, 0.34, color, str(num), 13)
    text(x + 0.52, y + 0.05, w - 0.6, 0.40, [dict(t=title, b=True, s=14.5, c=tcolor)], anchor="m")


def title_block(x, y, w, title, sub=None, ts=10, ss=8):
    paras = [dict(t=title, b=True, s=ts, c=INK, align="c")]
    if sub:
        paras.append(dict(t=sub, s=ss, c=TXT, align="c"))
    text(x, y, w, 0.40, paras)


def chip(x, y, w, h, label, color, size=7.5):
    rrect(x, y, w, h, line=color, fill="FFFFFF", lw=1.2, r=0.04, paras=[dict(t=label, b=True, s=size, c=INK, align="c")], margins=(0.01, 0, 0.01, 0))


def pill(x, y, w, h, paras, line_color=INK, fill="FFFFFF", lw=1.0, r=0.05, anchor="m", margins=(0.05, 0.0, 0.05, 0.0)):
    return rrect(x, y, w, h, line=line_color, fill=fill, lw=lw, r=r, paras=paras, anchor=anchor, margins=margins)


def family_badge(x, y, letter, color, d=0.145):
    oval(x, y, d, d, fill=color, paras=[dict(t=letter, b=True, s=6.5, c="FFFFFF", align="c", font="Arial")])


# ================================================================================ line icons
def icon_code(x, y, s):
    rrect(x, y, s, s * 0.92, line=INK, fill="FFFFFF", lw=1.75, r=s * 0.12)
    text(x, y + s * 0.10, s, s * 0.5, [dict(t="</>", b=True, s=s * 20, c=INK, align="c", font="Consolas")], anchor="m")
    line(x + s * 0.22, y + s * 0.70, x + s * 0.78, y + s * 0.70, INK, 1.5, cap_round=True)


def icon_shield(x, y, s):
    w, h = s * 0.84, s
    X, Y = (lambda u: x + u * w), (lambda v: y + v * h)
    pts = [(X(0.5), Y(0.0)), (X(1.0), Y(0.14)), (X(1.0), Y(0.52))]
    pts += [(X(a), Y(b)) for a, b in bez((1.0, 0.52), (1.0, 0.86), (0.5, 1.0))]
    pts += [(X(a), Y(b)) for a, b in bez((0.5, 1.0), (0.0, 0.86), (0.0, 0.52))]
    pts += [(X(0.0), Y(0.14))]
    poly(pts, True, fill="FFFFFF", lw=1.75)
    oval(X(0.5) - s * 0.075, Y(0.36) - s * 0.075, s * 0.15, s * 0.15, fill=INK)
    poly([(X(0.5) - s * 0.035, Y(0.40)), (X(0.5) + s * 0.035, Y(0.40)), (X(0.5) + s * 0.05, Y(0.66)), (X(0.5) - s * 0.05, Y(0.66))], True, fill=INK, lw=0.5)


def icon_people(x, y, s, color=INK):
    for dx in (0.0, 0.46):
        cx = x + s * (0.27 + dx)
        oval(cx - s * 0.12, y, s * 0.24, s * 0.24, fill=color)
        pts = arc(cx, y + s * 0.80, s * 0.24, s * 0.46, 0, 180)
        poly(pts, True, fill=color, color=color, lw=0.5)


def doc_outline(x, y, w, h, fold=0.28, lw=1.5, color=INK):
    f = w * fold
    poly([(x, y), (x + w - f, y), (x + w, y + f), (x + w, y + h), (x, y + h)], True, fill="FFFFFF", color=color, lw=lw)
    poly([(x + w - f, y), (x + w - f, y + f), (x + w, y + f)], False, color=color, lw=lw)


def icon_doc_lines(x, y, s, n=3):
    w, h = s * 0.78, s
    doc_outline(x, y, w, h)
    for i in range(n):
        yy = y + h * (0.42 + 0.17 * i)
        line(x + w * 0.20, yy, x + w * (0.80 if i < n - 1 else 0.55), yy, INK, 1.25, cap_round=True)


def icon_doc_check(x, y, s):
    w, h = s * 0.78, s
    doc_outline(x, y, w, h)
    poly([(x + w * 0.20, y + h * 0.62), (x + w * 0.42, y + h * 0.80), (x + w * 0.80, y + h * 0.40)], False, color=INK, lw=1.75)


def icon_doc_search(x, y, s):
    w, h = s * 0.78, s
    doc_outline(x, y, w, h)
    for i in range(2):
        line(x + w * 0.18, y + h * (0.42 + 0.14 * i), x + w * 0.70, y + h * (0.42 + 0.14 * i), INK, 1.1, cap_round=True)
    cx, cy, r = x + w * 0.60, y + h * 0.68, s * 0.15
    oval(cx - r, cy - r, 2 * r, 2 * r, fill="FFFFFF", line=INK, lw=1.5)
    line(cx + r * 0.72, cy + r * 0.72, cx + r * 1.6, cy + r * 1.6, INK, 1.75, cap_round=True)


def icon_bug(x, y, s):
    cx = x + s * 0.5
    oval(cx - s * 0.20, y + s * 0.28, s * 0.40, s * 0.58, fill=INK)
    oval(cx - s * 0.12, y + s * 0.08, s * 0.24, s * 0.22, fill=INK)
    for dy in (0.40, 0.56, 0.72):
        line(cx - s * 0.20, y + s * dy, cx - s * 0.46, y + s * (dy - 0.06), INK, 1.4, cap_round=True)
        line(cx + s * 0.20, y + s * dy, cx + s * 0.46, y + s * (dy - 0.06), INK, 1.4, cap_round=True)
    line(cx - s * 0.06, y + s * 0.10, cx - s * 0.18, y - s * 0.02, INK, 1.2, cap_round=True)
    line(cx + s * 0.06, y + s * 0.10, cx + s * 0.18, y - s * 0.02, INK, 1.2, cap_round=True)
    line(cx, y + s * 0.36, cx, y + s * 0.84, "FFFFFF", 0.9)


def icon_monitor(x, y, s):
    rrect(x, y, s, s * 0.70, line=INK, fill="FFFFFF", lw=1.75, r=s * 0.07)
    text(x, y + s * 0.02, s, s * 0.60, [dict(t="</>", b=True, s=s * 17, c=INK, align="c", font="Consolas")], anchor="m")
    line(x + s * 0.5, y + s * 0.70, x + s * 0.5, y + s * 0.86, INK, 1.75)
    line(x + s * 0.28, y + s * 0.88, x + s * 0.72, y + s * 0.88, INK, 2.0, cap_round=True)


def icon_bars(x, y, s):
    for i, hh in enumerate((0.38, 0.68, 0.98)):
        bw = s * 0.24
        rrect(x + i * (bw + s * 0.07), y + s * (1 - hh), bw, s * hh, line=INK, fill="FFFFFF", lw=1.6, r=0.01)
    line(x - s * 0.04, y + s * 1.02, x + s * 0.88, y + s * 1.02, INK, 1.5, cap_round=True)


def icon_jsonl(x, y, s):
    for k in (2, 1):
        rrect(x + s * 0.09 * k, y - s * 0.09 * k + s * 0.18, s * 0.62, s * 0.78, line=INK, fill="FFFFFF", lw=1.4, r=s * 0.05)
    rrect(x, y + s * 0.18, s * 0.62, s * 0.78, line=INK, fill="FFFFFF", lw=1.75, r=s * 0.05)
    rrect(x - s * 0.04, y + s * 0.44, s * 0.70, s * 0.24, line=INK, fill="FFFFFF", lw=1.6, r=s * 0.03,
          paras=[dict(t="JSONL", b=True, s=s * 11.5, c=INK, align="c", font="Arial")], margins=(0, 0, 0, 0))


def icon_ai(x, y, s):
    rrect(x, y, s, s * 0.82, line=INK, fill="FFFFFF", lw=1.6, r=s * 0.16, paras=[dict(t="AI", b=True, s=s * 19, c=INK, align="c", font="Arial")], margins=(0, 0, 0, 0))
    line(x + s * 0.5, y, x + s * 0.5, y - s * 0.14, INK, 1.4, cap_round=True)
    oval(x + s * 0.5 - s * 0.05, y - s * 0.21, s * 0.10, s * 0.10, fill=INK)


# ================================================================================ content
# ---------------------------------------------------------------- band 1
B1Y, B1H = 0.08, 2.50
band(0.10, B1Y, 9.60, B1H, BLUE, DBLUE, 1, "Multilingual Task Synthesis & Sociolinguistic Perturbation Engine")
BY, BH = B1Y + 0.47, 1.95
X1, W1 = 0.22, 1.85
X2, W2 = 2.29, 3.25
X3, W3 = 5.76, 1.85
X4, W4 = 7.83, 1.75
for xx, ww in ((X1, W1), (X2, W2), (X3, W3), (X4, W4)):
    rrect(xx, BY, ww, BH, line=BLUE, fill="FFFFFF", lw=1.1, r=0.07)
for xa in (X1 + W1, X2 + W2, X3 + W3):
    line(xa + 0.02, BY + BH / 2, xa + 0.20, BY + BH / 2, DBLUE, 1.5, arrow=True)

# box 1: canonical tasks
title_block(X1, BY + 0.05, W1, "Canonical Tasks", f"(N = {N_TASKS})")
icon_code(X1 + 0.09, BY + 0.52, 0.42)
text(X1 + 0.62, BY + 0.44, 1.2, 0.74, [
    dict(t=f"{N_FUNC} Functional Tasks", b=True, s=8.5, c=INK, after=1),
    dict(t="80 adapted classics", s=7.5, bullet=True), dict(t="20 author-original", s=7.5, bullet=True),
    dict(t="38 harder, multi-step", s=7.5, bullet=True)])
line(X1 + 0.12, BY + 1.22, X1 + W1 - 0.12, BY + 1.22, GREY, 0.8, dash=MSO_LINE.DASH)
icon_shield(X1 + 0.14, BY + 1.36, 0.40)
text(X1 + 0.62, BY + 1.27, 1.2, 0.66, [
    dict(t=f"{N_SEC} Security-Critical Tasks", b=True, s=8.5, c=INK, after=1),
    dict(t=f"({N_CWE} CWE types)", s=7.5, after=1),
    dict(t="exploit-test oracles", s=7.5, bullet=True)])

# box 2: perturbation engine
title_block(X2, BY + 0.05, W2, "Sociolinguistic Perturbation Engine", f"(10 variants, N = {N_PROMPTS:,} prompts)")
rows = [("EN", "Canonical English"), ("EN_TYPO", "English with keyboard noise"), ("BN", "Standard Bangla (formal)"),
        ("BN_SPELL", "Bangla with phonetic spelling shifts"), ("BN_MIX", "Colloquial code-mixed Bangla"),
        ("BN_TYPO", "Bangla with grapheme-level typos"), ("BANGLISH", "Bangla in Latin script (chat style)"),
        ("ES", "Spanish control (clean / typos)")]
for i, (k, d) in enumerate(rows):
    yy = BY + 0.50 + i * 0.158
    chip(X2 + 0.12, yy, 0.82, 0.138, k, CHIP[k])
    rrect(X2 + 1.02, yy, W2 - 1.14, 0.138, line="C9C9C9", fill="F3F3F3", lw=0.7, r=0.04,
          paras=[dict(t=d, s=7.5, c=TXT)], margins=(0.07, 0, 0.02, 0))
text(X2 + 0.08, BY + 1.78, W2 - 0.16, 0.14, [dict(t="Controls: ID_ONLY · EN_PAD · BN_DESC · HYB33/67 · hints", i=True, s=7, c="444444", align="c")])

# box 3: validation and gates
title_block(X3, BY + 0.05, W3, "Validation & QA", "(852 non-English translations)")
icon_doc_check(X3 + 0.10, BY + 0.50, 0.34)
text(X3 + 0.54, BY + 0.46, 1.28, 0.56, [dict(t="Oracle gate", b=True, s=8, c=INK, after=1),
     dict(t="reference passes all tests; insecure one fails the exploit test", s=7)])
line(X3 + 0.12, BY + 1.06, X3 + W3 - 0.12, BY + 1.06, GREY, 0.8, dash=MSO_LINE.DASH)
icon_ai(X3 + 0.10, BY + 1.22, 0.32)
text(X3 + 0.54, BY + 1.10, 1.28, 0.44, [dict(t="LLM audit", b=True, s=8, c=INK, after=1),
     dict(t="852 translations, 43 fixes applied", s=7)])
line(X3 + 0.12, BY + 1.56, X3 + W3 - 0.12, BY + 1.56, GREY, 0.8, dash=MSO_LINE.DASH)
icon_people(X3 + 0.10, BY + 1.64, 0.28)
text(X3 + 0.54, BY + 1.60, 1.28, 0.32, [dict(t="Native-speaker check", b=True, s=8, c=INK, after=1),
     dict(t="all Bangla, both authors", s=7)])

# box 4: corpus
title_block(X4, BY + 0.05, W4, "Benchmark Corpus", f"(N = {N_PROMPTS:,} prompts)")
icon_jsonl(X4 + W4 / 2 - 0.34, BY + 0.46, 0.62)
text(X4 + 0.12, BY + 1.22, W4 - 0.18, 0.68, [
    dict(t="Standardised prompt format", s=7.5, bullet=True), dict(t="Shared signatures and tests", s=7.5, bullet=True),
    dict(t="+ 672 causal-control prompts", s=7.5, bullet=True), dict(t="Frozen and released", s=7.5, bullet=True)])

# ---------------------------------------------------------------- band 2
B2Y, B2H = 2.70, 3.12
band(0.10, B2Y, 9.60, B2H, RED, DRED, 2, "Dual-Gate Evaluation Harness & Socioeconomic Accounting")
CY, CH = B2Y + 0.47, 2.57
LX, LW = 0.22, 2.55
rrect(LX, CY, LW, CH, line=RED, fill="FFFFFF", lw=1.1, r=0.07)
title_block(LX, CY + 0.05, LW, "Inference Orchestration", f"({N_MODELS} open models, N = {N_EXEC} samples)")


def tier(y, label):
    text(LX + 0.10, y, LW - 0.2, 0.14, [dict(t=label, b=True, s=7.5, c=DRED)])


def model_row(y, letter, color, label):
    family_badge(LX + 0.12, y + 0.005, letter, color)
    text(LX + 0.32, y, LW - 0.40, 0.15, [dict(t=label, s=7.5, c=TXT)], anchor="m")


FAM = {"Q": "6C5CE7", "D": "2F6FDE", "T": "E8A317", "G": "1FA3C7", "M": "F2652A"}
y0 = CY + 0.50
tier(y0, "Tier 1 · Kaggle 2×T4 · fp16 · 3 samples")
for i, (l, t_) in enumerate((("Q", "Qwen2.5-Coder 1.5B · 3B · 7B"), ("D", "DeepSeek-Coder 6.7B"), ("Q", "Qwen3 8B"), ("T", "TigerCoder 1B (Bangla)"))):
    model_row(y0 + 0.155 + i * 0.158, l, FAM[l], t_)
y1 = y0 + 0.155 + 4 * 0.158 + 0.03
tier(y1, "Tier 2 · RTX 5090 · Q4 GGUF · 3 samples")
for i, (l, t_) in enumerate((("Q", "Qwen3.8 27B"), ("G", "Gemma-4 31B · E4B"), ("M", "Ministral-3 14B"), ("Q", "Qwen2.5-Coder 14B · 7B (Q4)"))):
    model_row(y1 + 0.155 + i * 0.158, l, FAM[l], t_)
yd = y1 + 0.155 + 4 * 0.158 + 0.04
line(LX + 0.12, yd, LX + LW - 0.12, yd, GREY, 0.8, dash=MSO_LINE.DASH)
icon_doc_lines(LX + 0.12, yd + 0.07, 0.27)
text(LX + 0.46, yd + 0.04, LW - 0.54, 0.34, [dict(t="T = 0.2 · top-p 0.95 · 768 tokens", s=7.5, c=TXT),
     dict(t="+ TRANSLATE_FIRST arm", s=7.5, c=TXT)])

line(LX + LW + 0.02, CY + CH / 2, LX + LW + 0.30, CY + CH / 2, RED, 2.0, arrow=True)

HX = LX + LW + 0.32
HW = 9.58 - HX
rrect(HX, CY, HW, CH, line=RED, fill="FFFFFF", lw=1.1, r=0.07)
text(HX, CY + 0.05, HW, 0.22, [dict(t="Tri-Track Verification Harness", b=True, s=10, c=INK, align="c")])
TY, TH = CY + 0.32, CH - 0.39
AX, AW = HX + 0.06, 2.00
BX, BW = AX + AW + 0.06, 2.12
CXX = BX + BW + 0.06
CW = HX + HW - 0.06 - CXX

# --- track A
rrect(AX, TY, AW, TH, line=TEAL, fill="FFFFFF", lw=1.2, r=0.06)
badge(AX + 0.17, TY + 0.17, 0.22, TEAL, "A", 8.5)
title_block(AX + 0.30, TY + 0.04, AW - 0.34, "Dynamic Execution Sandbox", "(Correctness Evaluation)", ts=8, ss=7.5)
icon_monitor(AX + 0.09, TY + 0.52, 0.52)
text(AX + 0.70, TY + 0.47, AW - 0.74, 0.62, [dict(t="Isolated sandbox", s=7.5, bullet=True), dict(t="10 s timeout", s=7.5, bullet=True),
     dict(t="1 GB memory limit", s=7.5, bullet=True), dict(t="No network access", s=7.5, bullet=True)])
line(AX + AW / 2, TY + 1.12, AX + AW / 2, TY + 1.30, INK, 1.1, arrow=True)
pill(AX + 0.14, TY + 1.32, AW - 0.28, 0.27, [dict(t="Pytest test suites", s=7.5, c=TXT, align="c")], margins=(0.30, 0, 0.04, 0))
icon_doc_check(AX + 0.20, TY + 1.345, 0.24)
line(AX + AW / 2, TY + 1.60, AX + AW / 2, TY + 1.76, INK, 1.1, arrow=True)
pill(AX + 0.14, TY + 1.78, AW - 0.28, 0.27, [dict(t="Pass@1 per model × variant", b=True, s=7.5, c=INK, align="c")])

# --- track B
rrect(BX, TY, BW, TH, line=BRED, fill="FFFFFF", lw=1.2, r=0.06)
badge(BX + 0.17, TY + 0.17, 0.22, BRED, "B", 8.5)
title_block(BX + 0.30, TY + 0.04, BW - 0.34, "Dual Security Gate", "(Static + Dynamic)", ts=8, ss=7.5)
pill(BX + 0.10, TY + 0.50, BW - 0.20, 0.40, [dict(t="Static analysis", b=True, s=7.5, c=INK), dict(t="Bandit, CWE-matched rules", s=7, c=TXT)], margins=(0.40, 0, 0.03, 0))
icon_doc_search(BX + 0.17, TY + 0.545, 0.30)
text(BX + BW / 2 - 0.1, TY + 0.91, 0.2, 0.14, [dict(t="+", b=True, s=10, c=INK, align="c")], anchor="m")
pill(BX + 0.10, TY + 1.06, BW - 0.20, 0.40, [dict(t="Dynamic exploit tests", b=True, s=7.5, c=INK), dict(t="benign attacks, sandboxed", s=7, c=TXT)], margins=(0.40, 0, 0.03, 0))
icon_bug(BX + 0.19, TY + 1.11, 0.26)
line(BX + BW / 2, TY + 1.47, BX + BW / 2, TY + 1.60, INK, 1.1, arrow=True)
text(BX, TY + 1.61, BW, 0.14, [dict(t="Exploit concordance analysis", b=True, s=7.5, c=INK, align="c")])
pill(BX + 0.10, TY + 1.78, BW - 0.20, 0.30, [dict(t="Static-Pass Dynamic-Fail (SPDF)", b=True, s=7.5, c=INK, align="c"),
     dict(t=f"{SPDF}% of vulnerable code missed", s=7, c=TXT, align="c")], line_color="E7A3A3", fill="FDEDED", margins=(0.02, 0, 0.02, 0))

# --- track C
rrect(CXX, TY, CW, TH, line=BBLUE, fill="FFFFFF", lw=1.2, r=0.06)
badge(CXX + 0.17, TY + 0.17, 0.22, BBLUE, "C", 8.5)
title_block(CXX + 0.30, TY + 0.04, CW - 0.34, "Token Cost Accounting", "(Socioeconomic Analysis)", ts=8, ss=7.5)
text(CXX + 0.06, TY + 0.50, CW - 0.12, 0.16, [dict(t=f"{N_TOK} tokenizers · Bangla : English tokens", s=7.5, c=TXT, align="c")])
toks = [("DeepSeek-Coder (32k)", TOKR["ds"], "2F6FDE"), ("Qwen2.5 / Qwen3 (151k)", TOKR["qw"], "6C5CE7"),
        ("GPT-4o o200k (200k)", TOKR["g4o"], "10A37F"), ("Gemma-4 (262k)", TOKR["gm"], "1FA3C7")]
for i, (nm, rr, col) in enumerate(toks):
    yy = TY + 0.70 + i * 0.245
    rrect(CXX + 0.10, yy, CW - 0.20, 0.205, line=INK, fill="FFFFFF", lw=1.0, r=0.05)
    oval(CXX + 0.17, yy + 0.047, 0.11, 0.11, fill=col)
    text(CXX + 0.34, yy, CW - 0.80, 0.205, [dict(t=nm, s=7.5, c=TXT)], anchor="m")
    text(CXX + CW - 0.62, yy, 0.46, 0.205, [dict(t=f"{rr:.1f}×", b=True, s=8, c=INK, align="r")], anchor="m")
line(CXX + CW / 2, TY + 1.69, CXX + CW / 2, TY + 1.80, INK, 1.1, arrow=True)
text(CXX, TY + 1.80, CW, 0.14, [dict(t="Token expansion ratio", b=True, s=7.5, c=INK, align="c")])
rrect(CXX + 0.12, TY + 1.95, CW - 0.24, 0.22, line="9DBDE8", fill="E8F1FC", lw=0.9, r=0.05, paras=[
    dict(runs=[("T", dict(s=8.5, i=True)), ("variant", dict(s=8.5, sub=True)), (" / T", dict(s=8.5, i=True)), ("EN", dict(s=8.5, sub=True)),
               ("   Bangla premium", dict(s=7))], align="c", c=INK)], margins=(0.02, 0, 0.02, 0))

# ---------------------------------------------------------------- band 3
B3Y, B3H = 5.94, 1.62
band(0.10, B3Y, 9.60, B3H, GREEN, DGREEN, 3, "Statistical Synthesis & Governance Engine")
GY, GH = B3Y + 0.45, 1.10
gx = [(0.22, 2.40), (2.82, 2.15), (5.17, 1.75), (7.12, 2.46)]
for xx, ww in gx:
    rrect(xx, GY, ww, GH, line=GREEN, fill="FFFFFF", lw=1.1, r=0.07)
for (xa, wa) in gx[:3]:
    line(xa + wa + 0.02, GY + GH / 2, xa + wa + 0.18, GY + GH / 2, GREEN, 1.5, arrow=True)

xx, ww = gx[0]
text(xx, GY + 0.04, ww, 0.2, [dict(t="Non-Parametric Inferential Statistics", b=True, s=9, c=INK, align="c")])
icon_bars(xx + 0.12, GY + 0.42, 0.42)
text(xx + 0.62, GY + 0.32, ww - 0.66, 0.74, [dict(t="Cluster bootstrap (2,000)", s=7.5, bullet=True),
     dict(t="Wilcoxon / McNemar (Holm)", s=7.5, bullet=True), dict(t="Logistic GEE, fixed effects", s=7.5, bullet=True)])
xx, ww = gx[1]
text(xx, GY + 0.04, ww, 0.2, [dict(t="Causal Controls & Explainability", b=True, s=9, c=INK, align="c")])
text(xx + 0.12, GY + 0.32, ww - 0.2, 0.72, [dict(t="Length-matched EN, hybrids", s=7.5, bullet=True),
     dict(t="Failure anatomy, SHAP model", s=7.5, bullet=True), dict(t="Gradient × input, alignment", s=7.5, bullet=True)])
xx, ww = gx[2]
text(xx, GY + 0.04, ww, 0.2, [dict(t="Pareto Trade-Off Modeling", b=True, s=9, c=INK, align="c")])
px0, py0, pw, ph = xx + 0.28, GY + 0.34, 0.72, 0.56
line(px0, py0, px0, py0 + ph, INK, 1.1)
line(px0, py0 + ph, px0 + pw, py0 + ph, INK, 1.1)
curve = [(px0 + 0.05 + 0.62 * t, py0 + ph - 0.04 - 0.44 * (1 - (1 - t) ** 2.2)) for t in [i / 24 for i in range(25)]]
for (ax, ay), (bx, by) in zip(curve[:-1], curve[1:]):
    line(ax, ay, bx, by, INK, 0.9, dash=MSO_LINE.DASH)
for t in (0.08, 0.28, 0.5, 0.72, 0.95):
    cxp, cyp = px0 + 0.05 + 0.62 * t, py0 + ph - 0.04 - 0.44 * (1 - (1 - t) ** 2.2)
    oval(cxp - 0.022, cyp - 0.022, 0.044, 0.044, fill=INK)
tb = text(xx - 0.05, py0 + 0.04, 0.50, 0.40, [dict(t="ΔPass@1", s=6.5, c=INK, align="c")], anchor="m")
tb.rotation = 270
text(xx + 0.10, py0 + ph + 0.02, 1.0, 0.12, [dict(t="Token-cost inflation", s=7, c=INK, align="c")])
line(xx + 1.04, GY + 0.46, xx + 1.16, GY + 0.46, INK, 0.9, dash=MSO_LINE.DASH)
text(xx + 1.18, GY + 0.36, 0.55, 0.20, [dict(t="Pareto frontier", s=6.5, c=INK)], anchor="m")
xx, ww = gx[3]
text(xx, GY + 0.04, ww, 0.2, [dict(t="Policy Frameworks & Recommendations", b=True, s=8.5, c=INK, align="c")])
icon_doc_lines(xx + 0.12, GY + 0.40, 0.42, n=4)
text(xx + 0.58, GY + 0.30, ww - 0.64, 0.76, [dict(t="Keep technical terms in English", s=7.5, bullet=True),
     dict(t="Security reminders in English", s=7.5, bullet=True), dict(t="Translate-first needs a translator", s=7.5, bullet=True),
     dict(t="Exploit tests, not only linters", s=7.5, bullet=True)])

out = ROOT / "figures"
out.mkdir(exist_ok=True)
prs.save(out / "workflow.pptx")
print("saved", out / "workflow.pptx")
