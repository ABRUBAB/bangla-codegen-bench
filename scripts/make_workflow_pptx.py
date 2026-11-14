#!/usr/bin/env python3
"""Build the workflow diagram (Figure 1) as a native, editable PowerPoint slide (python-pptx).
Numbers are read from figures/workflow_numbers.json so the diagram always matches the data.
Export to PNG/PDF with scripts/export_pptx.ps1 (PowerPoint COM)."""
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from lxml import etree

ROOT = Path(__file__).resolve().parent.parent
N = json.loads((ROOT / "figures/workflow_numbers.json").read_text()) if (ROOT / "figures/workflow_numbers.json").exists() else {}

def g(k, d):
    return N.get(k, d)

# --- NPG-inspired palette (colour-blind friendly) --------------------------------------------
NAVY, RED, TEAL, CYAN, CORAL, SLATE, PURPLE = "3C5488", "E64B35", "00A087", "4DBBD5", "F39B7F", "8491B4", "7E6BA8"
BAND = {1: ("EAF0F8", NAVY), 2: ("FDF0EC", RED), 3: ("E8F6F3", TEAL), 4: ("F1EEF8", PURPLE)}

W, H = 10.0, 8.35
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
slide = prs.slides.add_slide(prs.slide_layouts[6])


def rgb(h):
    return RGBColor.from_string(h)


def rect(x, y, w, h, fill, line, lw=1.0, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06, dash=False):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = rgb(fill)
    s.line.color.rgb = rgb(line); s.line.width = Pt(lw)
    if dash:
        s.line.dash_style = 4
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    s.shadow.inherit = False
    return s


def text(shape, paras, anchor=MSO_ANCHOR.TOP, margin=0.07):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(margin); tf.margin_top = tf.margin_bottom = Inches(0.04)
    first = True
    for p in paras:
        para = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        para.alignment = p.get("align", PP_ALIGN.LEFT)
        para.space_after = Pt(p.get("after", 1.5)); para.space_before = Pt(0)
        run = para.add_run(); run.text = p["t"]
        f = run.font; f.size = Pt(p.get("s", 9)); f.bold = p.get("b", False); f.italic = p.get("i", False)
        f.name = p.get("font", "Calibri"); f.color.rgb = rgb(p.get("c", "222222"))


def box(x, y, w, h, title, lines, accent, fill="FFFFFF", tsize=10.5, lsize=9, align=PP_ALIGN.LEFT):
    s = rect(x, y, w, h, fill, accent, 1.4)
    paras = [{"t": title, "s": tsize, "b": True, "c": accent, "after": 2.5, "align": align}]
    paras += [{"t": l, "s": lsize, "align": align, "font": "Nirmala UI" if any("ঀ" <= ch <= "৿" for ch in l) else "Calibri"} for l in lines]
    text(s, paras)
    return s


def arrow(x1, y1, x2, y2, color="555555", lw=1.75):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color); c.line.width = Pt(lw)
    ln = c.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd")); tail.set("type", "triangle"); tail.set("w", "med"); tail.set("len", "med")
    return c


def band(idx, y, h, title):
    fill, col = BAND[idx]
    rect(0.15, y, W - 0.3, h, fill, col, 1.5, radius=0.03, dash=True)
    badge = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.25), Inches(y + 0.07), Inches(0.36), Inches(0.36))
    badge.fill.solid(); badge.fill.fore_color.rgb = rgb(col); badge.line.fill.background()
    text(badge, [{"t": str(idx), "s": 12, "b": True, "c": "FFFFFF", "align": PP_ALIGN.CENTER}], anchor=MSO_ANCHOR.MIDDLE, margin=0)
    tb = slide.shapes.add_textbox(Inches(0.68), Inches(y + 0.05), Inches(W - 1.0), Inches(0.4))
    text(tb, [{"t": title, "s": 12.5, "b": True, "c": col}], anchor=MSO_ANCHOR.MIDDLE)
    return col


nt = g("n_tasks", 150); nf = g("n_func", 100); ns = g("n_sec", 50); ncwe = g("n_cwe", 14)
npr = g("n_prompts", 1400); nmod = g("n_models", 7); ntok = g("n_tokenizers", 9)
ngen = g("n_generations", "38,850")

# ---------------------------------------------------------------- Band 1
y1, hb = 0.12, 1.98
band(1, y1, hb, f"Benchmark construction: {nt} tasks → {npr:,} prompts")
bx, by, bh = 0.3, y1 + 0.52, 1.36
bw = 2.18; gap = 0.2
x = bx
box(x, by, bw, bh, f"Task authoring ({nt})", [f"• {nf} functional (adapted + original)", f"• {ns} security tasks, {ncwe} CWE types", "• neutral specs, no security cue", "• pytest + behavioural exploit tests"], NAVY)
arrow(x + bw, by + bh / 2, x + bw + gap, by + bh / 2)
x += bw + gap
box(x, by, bw, bh, "Four renderings", ["EN  ·  BN (formal)", "BN_MIX (code-mixed)  ·  BANGLISH", "গ্রাহকের তালিকা ফেরত দেয়", "• identifiers kept verbatim", "• LLM-drafted + auto-checked"], NAVY)
arrow(x + bw, by + bh / 2, x + bw + gap, by + bh / 2)
x += bw + gap
box(x, by, bw, bh, "Perturbation engine", ["7 variants: EN, EN_TYPO, BN,", "BN_SPELL, BN_MIX, BN_TYPO, BANGLISH", "+ ID_ONLY control (signature only)", "+ 4 security-hint ablations"], NAVY)
arrow(x + bw, by + bh / 2, x + bw + gap, by + bh / 2)
x += bw + gap
box(x, by, bw, bh, "Oracle verification gate", ["• reference passes all tests", "• insecure ref passes functional,", "   fails exploit test", f"• frozen: {npr:,} prompts"], TEAL, fill="F4FBF9")

# ---------------------------------------------------------------- Band 2
y2 = y1 + hb + 0.2
band(2, y2, hb - 0.2, f"Generation: {nmod} models, 2 arms, 3 samples per prompt ({ngen} completions)")
by2, bh2 = y2 + 0.52, 1.16
box(0.3, by2, 3.0, bh2, "Models", ["Qwen2.5-Coder 1.5B / 3B / 7B", "DeepSeek-Coder 6.7B · Qwen3-8B", "TigerCoder (Bangla-specialised)", "Gemini 3.8 Flash (API)"], RED)
arrow(3.3, by2 + bh2 / 2, 3.55, by2 + bh2 / 2)
box(3.55, by2, 2.55, bh2, "DIRECT arm", ["prompt → code model", "all variants and controls"], RED)
box(6.3, by2, 2.55, bh2, "TRANSLATE_FIRST arm", ["BN / BN_MIX / BANGLISH →", "same model translates to English", "→ code model"], CORAL)
box(8.95 - 0.05, by2, 0.85, bh2, "T=0.2", ["top-p", "0.95", "3 samples"], SLATE, tsize=9.5, lsize=8.5, align=PP_ALIGN.CENTER)

# ---------------------------------------------------------------- Band 3
y3 = y2 + hb - 0.2 + 0.2
band(3, y3, hb, "Evaluation harness (isolated Linux sandbox: no network, rlimits, 10 s timeout)")
by3, bh3 = y3 + 0.52, 1.36
tw = 2.95; tg = 0.17
x = 0.3
box(x, by3, tw, bh3, "A  Correctness", ["• functional pytest suites", "• sample-level Pass@1", "• failure-type logging"], TEAL)
x += tw + tg
box(x, by3, tw, bh3, "B  Security: dynamic + static", ["• exploit tests (behavioural oracle)", "• Bandit static scan (CWE-matched)", "• secure-correct vs vulnerable-correct", "• Static-Pass Dynamic-Fail (SPDF)"], TEAL)
x += tw + tg
box(x, by3, tw, bh3, f"C  Cost ({ntok} tokenizers)", ["• token ratio vs English", "• workflow tokens incl. translation", "• Gemini usage metadata + price"], CORAL)

# ---------------------------------------------------------------- Band 4
y4 = y3 + hb + 0.2
band(4, y4, hb - 0.12, "Statistics, explainability and recommendations")
by4, bh4 = y4 + 0.52, 1.24
box(0.3, by4, 2.95, bh4, "Inference", ["• task-cluster bootstrap (95% CI)", "• paired Wilcoxon / McNemar + Holm", "• GEE logistic regression"], PURPLE)
box(3.42, by4, 3.2, bh4, "Explainability (XAI)", ["• SHAP on a failure-risk model", "• gradient × input token attribution", "• layer-wise EN↔BN alignment"], PURPLE)
box(6.79, by4, 3.0, bh4, "Outputs", ["• accuracy–cost Pareto frontier", "• CWE-level security maps", "• deployment recommendations"], PURPLE)

# inter-band arrows
for yy_from, yy_to in [(y1 + hb, y2), (y2 + hb - 0.2, y3), (y3 + hb, y4)]:
    arrow(W / 2, yy_from - 0.02, W / 2, yy_to + 0.1, "555555", 2.25)

out = ROOT / "figures"
out.mkdir(exist_ok=True)
prs.save(out / "workflow.pptx")
print("saved", out / "workflow.pptx")
