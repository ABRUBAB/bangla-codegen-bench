# BanglaCodeBench

Controlled benchmark and pipeline for measuring how the **language of the prompt** (English, formal Bangla,
code-mixed Bangla, Banglish and perturbed variants) changes the **correctness**, **executable security** and
**tokenization cost** of code written by LLMs.

Companion code and data of the paper *"Does the Language of the Prompt Change What Code LLMs Write? Correctness,
Executable Security and Tokenization Cost of English, Bangla and Banglish Prompts"* (A. Rubab, M. M. K. Badhon).

## What is in the box

| path | content |
|---|---|
| `data/tasks/functional.jsonl`, `security.jsonl` | 150 tasks: description in 4 renderings (en / bn / bn_mix / banglish), signature, reference solution, pytest tests; security tasks add an insecure reference and a behavioural exploit test |
| `data/prompts/prompts.jsonl` | the 1,400 frozen prompts (7 variants + `ID_ONLY` control + 4 security-hint ablations) |
| `data/generations/<model>/` | every raw model output (jsonl) incl. translations of the translate-first arm |
| `data/results/` | execution, Bandit and token-count results, statistics tables |
| `bcb/` | package: prompt builder / perturbation engine, task loader, statistics, sandbox driver and in-sandbox runner |
| `scripts/` | pipeline: `build_prompts.py`, `generate_hf.py`, `generate_gemini.py`, `run_execution.py`, `static_scan.py`, `tokenize_prompts.py`, `analyze.py`, `xai_*.py`, `make_figures.py`, `make_workflow_pptx.py` |
| `figures/` | all paper figures; `workflow.pptx` is the editable PowerPoint source of Figure 1 |
| `paper/` | LaTeX sources (Elsevier `cas-dc`) and verified bibliography |
| `docs/` | task schema, authoring brief, related-work notes |
| `data/audit/` | native-speaker audit workbooks (see below) |

## Design in one paragraph
Every language variant of a task shows the model the *same* code signature; only the natural-language description
changes, and each rendering must impose exactly the English requirements (identifiers/literals verbatim). Security
task descriptions are *neutral* (no security cue). Correctness is decided by pytest, security by behavioural exploit
tests (benign attacks that create marker files, return extra rows, ...), never by string matching. Before any model
is run, every task passes a gate: reference passes all tests; for security tasks the secure reference passes
functional + exploit tests and the insecure reference passes functional but fails the exploit test.

## Reproducing

```bash
pip install -r requirements.txt

# 1. prompts and token counts
python scripts/build_prompts.py
python scripts/tokenize_prompts.py

# 2. verify the benchmark in the sandbox (Docker image or the WSL fallback, see bcb/sandbox)
BCB_BACKEND=docker python scripts/verify_tasks.py data/tasks/functional.jsonl data/tasks/security.jsonl

# 3. generation (GPU, e.g. Kaggle T4x2) and Gemini API generation
python scripts/generate_hf.py --hf-id Qwen/Qwen2.5-Coder-7B-Instruct --model-id qwen2.5-coder-7b \
       --prompts data/prompts/prompts.jsonl --out data/generations
python scripts/generate_gemini.py --model gemini-3.8-flash --model-id gemini-3.8-flash \
       --prompts data/prompts/prompts.jsonl --out data/generations      # needs GEMINI_API_KEY

# 4. execute everything, static scan, statistics, XAI, figures
python scripts/run_execution.py --gens data/generations --out data/results/exec.jsonl
python scripts/static_scan.py   --gens data/generations --out data/results/bandit.jsonl
python scripts/analyze.py
python scripts/xai_analysis.py
python scripts/make_figures.py
```

**Safety.** Generated code is executed only inside the sandbox (`bcb/sandbox`): no network, unprivileged user,
resource limits, fresh temp dir per sample. Never run model outputs on a machine you care about.

**API keys.** No secrets are stored in this repository. Pass keys via environment variables.

## Native-speaker audit (important)
The Bangla/Banglish renderings were drafted with an LLM and checked automatically (schema, identifier and numeral
preservation, back-translation with two independent LLM judges). They are **not** human-certified until two native
speakers complete the audit: fill `data/audit/audit_annotator1.xlsx` and `audit_annotator2.xlsx` independently,
then run `python scripts/ingest_audit.py data/audit/audit_annotator1.xlsx data/audit/audit_annotator2.xlsx`;
this rewrites `paper/audit_numbers.tex` so the paper reports the real agreement statistics.

## Licence
Code: MIT. Task and prompt data: CC BY 4.0. Model outputs are research artefacts of third-party models and remain
subject to those models' terms.

## Citation
See `CITATION.cff`.
