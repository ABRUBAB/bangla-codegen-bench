# Errata and known ambiguities of the benchmark text (as run)

Found by the post-hoc failure analysis (`scripts/failure_anatomy.py`, LLM-classified; not human-verified). The texts below were
**not** changed after generation, so that the reported results correspond to the released prompts; a sensitivity analysis that
drops these tasks is reported in the paper. Fix them in a later release together with a regeneration of the affected cells.

| Task | Variants | Issue | Suggested wording |
|---|---|---|---|
| S009 | BN (and its noised variants) | The sentence "`0` স্ট্যাটাস কোড দিয়ে শেষ হলে" does not say explicitly that the exit/return code is meant; two failing samples compared output text instead. | "এক্সিট স্ট্যাটাস (রিটার্ন কোড) `0`" |
| F082 | BN, BN_SPELL, BN_TYPO | "রোমান হরফে লেখা" (written in Roman script) was read by some samples as *Roman numerals*; the English text says "romanized spellings". The Bangla is faithful but less explicit. | "ল্যাটিন হরফে (romanized) লেখা" |

Other negation / "strictly less than" failures (F039, F041, F074) were judged model errors, not translation problems.
