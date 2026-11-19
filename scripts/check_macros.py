#!/usr/bin/env python3
"""List every number macro used in the paper text that is not defined (an undefined \\csname silently prints nothing).
Checks \\N{key}, \\XN{m}{v}{f}, \\TV{m}{v}{f}, \\CZ{m}{v}{f} and plain \\newcommand-style macros of the generated number files."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "paper"
BS = chr(92)


def keys(path, prefix):
    f = P / path
    if not f.exists():
        return set()
    return set(re.findall(re.escape(BS) + "csname " + prefix + r"(.*?)" + re.escape(BS) + "endcsname", f.read_text(encoding="utf-8")))


def main():
    nkeys = keys("numbers.tex", "n@")
    sgk = keys("scale_numbers.tex", "sg@")
    xn, tv, cz = keys("xai_numbers.tex", "xn@"), keys("translation_numbers.tex", "tv@"), keys("causal_numbers.tex", "cz@")
    defined_cmds = set()
    for f in P.glob("*numbers.tex"):
        defined_cmds |= set(re.findall(re.escape(BS) + r"newcommand\{" + re.escape(BS) + r"([A-Za-z]+)\}", f.read_text(encoding="utf-8")))
    bad = []
    for f in sorted(P.glob("*.tex")):
        if f.name.endswith("numbers.tex") or f.name == "main.tex":
            continue
        t = f.read_text(encoding="utf-8")
        for k in re.findall(re.escape(BS) + r"N\{([^}]+)\}", t):
            if k not in nkeys:
                bad.append((f.name, "\\N{" + k + "}"))
        for k in re.findall(re.escape(BS) + r"SG\{([^}]+)\}", t):
            if k not in sgk:
                bad.append((f.name, BS + "SG{" + k + "}"))
        for name, ks in (("XN", xn), ("TV", tv), ("CZ", cz)):
            for m in re.finditer(re.escape(BS) + name + r"\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}", t):
                k = "@".join(m.groups())
                if k.replace("_", "") not in ks and k not in ks:
                    bad.append((f.name, "\\%s{%s}" % (name, "}{".join(m.groups()))))
        for c in re.findall(re.escape(BS) + r"(fa[A-Z][A-Za-z]*|sens(?:All|Excl)[A-Za-z]+|llm[A-Za-z]+|N(?:Tasks|Func|Sec|Cwe|CweAll|Prompts|Models))\b", t):
            if c not in defined_cmds:
                bad.append((f.name, "\\" + c))
    for f, k in sorted(set(bad)):
        print(f"{f}: undefined {k}")
    print(f"{len(set(bad))} undefined macro uses")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
