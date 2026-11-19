# BanglaCodeBench

Controlled benchmark and pipeline for measuring how the **language of the prompt** (English, formal Bangla,
code-mixed Bangla, Banglish, a Spanish control and perturbed variants) changes the **correctness**, **executable
security** and **tokenization cost** of code written by LLMs.

Companion code and data of the paper *"Does the Language of the Prompt Change What Code LLMs Write? Correctness,
Executable Security and Tokenization Cost of English, Bangla and Banglish Prompts"* (A. Rubab, M. M. K. Badhon).
Repository: <https://github.com/ABRUBAB/bangla-codegen-bench>

## What is in the box

| path | content |
|---|---|
| `data/tasks/functional.jsonl`, `security.jsonl` | 213 tasks (138 functional, 75 security over 20 CWE categories): description in English, formal Bangla, code-mixed Bangla, Banglish and Spanish, signature, reference solution, pytest tests; security tasks add an insecure reference and a behavioural exploit test |
| `data/prompts/prompts.jsonl` | the 2,505 frozen main prompts (10 variants incl. the `ID_ONLY` control, + 5 security-hint ablations); `prompts_causal.jsonl`: interventional controls (`EN_PAD`, `BN_DESC`, `HYB33`, `HYB67`) |
| `data/generations/<model>/` | every raw model output (jsonl), the models' own translations (`translations.jsonl`) and their sanitised versions (`translations_clean.jsonl`) |
| `data/results/` | execution, Bandit and token-count results, statistics tables |
| `data/audit/` | AI audit of all renderings, applied fixes, `errata.md`, blinded human review workbooks |
| `data/analysis/`, `data/xai/` | failure pairs and their LLM-classified root causes; white-box probe outputs (large tensors are not tracked) |
| `bcb/` | package: prompt builder / perturbation engine, causal prompt controls, task loader, statistics, sandbox driver and in-sandbox runner |
| `scripts/` | pipeline: `build_prompts.py`, `build_causal_prompts.py`, `generate_hf.py`, `generate_gemini.py`, `generate_local_server.py` (any OpenAI-compatible local server), `run_execution.py`, `static_scan.py`, `tokenize_prompts.py`, `analyze.py`, `clean_translations.py`, `causal_analysis.py`, `scale_analysis.py` (size/recency analysis), `failure_anatomy.py`, `sensitivity.py`, `xai_*.py`, `make_numbers.py`, `make_figures.py`, `make_workflow_pptx.py` |
| `figures/` | all paper figures; `workflow.pptx` is the editable PowerPoint source of Figure 1 |
| `paper/` | LaTeX sources (Elsevier `cas-dc`) and verified bibliography; every number in the text comes from a generated macro file |
| `docs/` | task schema, authoring briefs, related-work notes, `CENTER_PC_RUNBOOK.md` (running more models on a local GPU) |

## Design in one paragraph
Every language variant of a task shows the model the *same* code signature; only the natural-language description
changes, and each rendering must impose exactly the English requirements (identifiers/literals verbatim). Security
task descriptions are *neutral* (no security cue). Correctness is decided by pytest, security by behavioural exploit
tests (benign attacks that create marker files, return extra rows, ...), never by string matching. Before any model
is run, every task passes a gate: reference passes all tests; for security tasks the secure reference passes
functional + exploit tests and the insecure reference passes functional but fails the exploit test. Interventional
prompts separate language from token length; the translate-first arm uses sanitised translations (code blocks removed)
because code models often solve the task inside the translation.

## Reproducing

```bash
pip install -r requirements.txt

# 1. prompts and token counts
python scripts/build_prompts.py
python scripts/build_causal_prompts.py
python scripts/tokenize_prompts.py

# 2. verify the benchmark in the sandbox (Docker image or the WSL fallback, see bcb/sandbox)
BCB_BACKEND=docker python scripts/verify_tasks.py data/tasks/functional.jsonl data/tasks/security.jsonl

# 3. generation. Tier 1 (fp16, 3 samples/prompt): Hugging Face Transformers on Kaggle T4x2 (generate_hf.py).
#    Tier 2 (4-bit GGUF, 1 sample/prompt): llama.cpp llama-server + generate_local_server.py, see docs/CENTER_PC_RUNBOOK.md.
#    Any OpenAI-compatible server (Ollama, LM Studio, vLLM) or the Gemini API (free-tier quota is too small for a full run) also works.
python scripts/generate_hf.py --hf-id Qwen/Qwen2.5-Coder-7B-Instruct --model-id qwen2.5-coder-7b \
       --prompts data/prompts/prompts.jsonl --out data/generations
python scripts/generate_gemini.py --model gemini-3.1-flash-lite --model-id gemini-3.1-flash-lite \
       --prompts data/prompts/prompts.jsonl --out data/generations      # needs GEMINI_API_KEY
python scripts/generate_local_server.py --base-url http://localhost:11434/v1 --model <name> --model-id <id> \
       --prompts data/prompts/prompts.jsonl --out data/generations

# 4. sanitised translate-first arms (own translation / fixed translator), then everything else
python scripts/clean_translations.py
python scripts/generate_hf.py ... --only EXTERNAL --external-translations data/generations/qwen3-8b/translations_clean.jsonl \
       --external-condition TF_CLEAN_Q3

# 5. execute everything, static scan, statistics, XAI, figures, paper
BCB_BACKEND=wsl bash scripts/final_pipeline.sh
```

**Safety.** Generated code is executed only inside the sandbox (`bcb/sandbox`): no network, unprivileged user,
resource limits, fresh temp dir per sample. Never run model outputs on a machine you care about.

**API keys.** No secrets are stored in this repository. Pass keys via environment variables.

## Native-speaker audit (important)
The Bangla, code-mixed, Banglish and Spanish renderings were drafted with an LLM, checked automatically (schema,
identifier and numeral preservation) and audited by Claude Opus agents (`data/audit/llm_opus_*.json`; statistics via
`python scripts/audit_tools.py llm-stats`). They are **not** human-certified until native speakers complete the review:
`python scripts/audit_tools.py build` writes the blinded review workbooks `data/audit/review_annotator{1,2}.xlsx`
(150-item fixed-seed sample; AI judgements pre-filled, independent empty columns per reviewer). After two people have
really completed them, run
`python scripts/audit_tools.py ingest data/audit/review_annotator1.xlsx data/audit/review_annotator2.xlsx`;
this rewrites `paper/audit_numbers.tex` so the paper reports the real agreement statistics. Until then the paper
describes the renderings as LLM-drafted, automatically checked and AI-audited only. Known ambiguities found after
generation are listed in `data/audit/errata.md`.

## Licence
Code: MIT. Task and prompt data: CC BY 4.0. Model outputs are research artefacts of third-party models and remain
subject to those models' terms.

## Citation
See `CITATION.cff`.
