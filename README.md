# BanglaCodeBench

Controlled benchmark and pipeline for measuring how the **language of the prompt** (English, formal Bangla,
code-mixed Bangla, Banglish, a Spanish control and perturbed variants) changes the **correctness**, **executable
security** and **tokenization cost** of code written by LLMs.

Companion code and data of the paper *"Does the Prompt Language Matter? An Empirical Study of Correctness, Security,
and Token Cost in Multilingual LLM Code Generation"* (A. Rubab, M. M. K. Badhon).
Repository: <https://github.com/ABRUBAB/bangla-codegen-bench>

## What is in the box

| path | content |
|---|---|
| `data/tasks/functional.jsonl`, `security.jsonl` | 213 tasks (138 functional, 75 security over 20 CWE categories): description in English, formal Bangla, code-mixed Bangla, Banglish and Spanish, signature, reference solution, pytest tests; security tasks add an insecure reference and a behavioural exploit test |
| `data/prompts/prompts.jsonl` | the 2,505 frozen main prompts (10 variants incl. the `ID_ONLY` control, + 5 security-hint ablations); `prompts_causal.jsonl`: interventional controls (`EN_PAD`, `BN_DESC`, `HYB33`, `HYB67`) |
| `data/generations/<model>/` | every raw model output (jsonl), the models' own translations (`translations.jsonl`) and their sanitised versions (`translations_clean.jsonl`) |
| `data/results/` | execution, Bandit and token-count results, statistics tables |
| `data/audit/` | AI audit of all translations, applied fixes, `errata.md`, unfilled review workbooks for an independent rating study |
| `data/analysis/`, `data/xai/` | failure pairs and their LLM-classified root causes; white-box probe outputs (large tensors are not tracked) |
| `bcb/` | package: prompt builder / perturbation engine, causal prompt controls, task loader, statistics, sandbox driver and in-sandbox runner |
| `scripts/` | pipeline: `build_prompts.py`, `build_causal_prompts.py`, `generate_hf.py`, `generate_gemini.py`, `generate_local_server.py` (any OpenAI-compatible local server), `run_execution.py`, `static_scan.py`, `tokenize_prompts.py`, `analyze.py`, `clean_translations.py`, `causal_analysis.py`, `scale_analysis.py` (size/recency analysis), `failure_anatomy.py`, `sensitivity.py`, `xai_*.py`, `make_numbers.py`, `make_figures.py`, `make_workflow_pptx.py` |
| `docs/` | task schema, authoring briefs, `CENTER_PC_RUNBOOK.md` (running more models on a local GPU) |

The scripts write their outputs into two folders that are not tracked in git: `figures/` (figures from
`make_figures.py` and the other figure scripts) and `paper/` (LaTeX macro and table files such as `numbers.tex` and
`tab_*.tex`, which carry every reported number). The manuscript itself is not part of this repository.

## Design in one paragraph
Every language variant of a task shows the model the *same* code signature; only the natural-language description
changes, and each translation must impose exactly the English requirements (identifiers/literals verbatim). Security
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

# 5. execute everything, static scan, statistics, XAI, number macros and figures
BCB_BACKEND=wsl bash scripts/final_pipeline.sh
```

**Large files.** The execution verdicts of all samples are stored compressed (`data/results/exec.jsonl.gz`); `scripts/final_pipeline.sh` unpacks them and
re-executes only samples that are new or whose code changed. Identical code for the same task is executed once and its verdict is reused (rows carry
`"reused": true`). Results are appended after every batch, so an interrupted run can be restarted. Heavy runs (sandbox execution, Bandit, statistics) were
performed on a separate GPU/CPU workstation; the sandbox is the WSL2 distro described in `bcb/sandbox/setup_wsl.sh` (or the Docker image).

**Safety.** Generated code is executed only inside the sandbox (`bcb/sandbox`): no network, unprivileged user,
resource limits, fresh temp dir per sample. Never run model outputs on a machine you care about.

**API keys.** No secrets are stored in this repository. Pass keys via environment variables.

## How the translations were checked
The formal Bangla, code-mixed, Banglish and Spanish translations were drafted with an LLM and then checked in three ways:
1. **Automatic gate:** expected script, and every identifier, literal and number of the English text kept verbatim.
2. **AI audit:** Claude Opus agents compared all 852 translations with the English task (`data/audit/llm_opus_*.json`;
   statistics via `python scripts/audit_tools.py llm-stats`); the 43 proposed corrections were applied
   (`data/audit/applied_fixes_*.json`).
3. **Native-speaker check:** after the prompts were frozen, both authors (native Bangla speakers) read the formal,
   code-mixed and Banglish versions of all 213 tasks against the English text, first independently and then together.
   They found no translation that adds, drops or changes a requirement, so no text was changed. This was a read-through
   by the authors, not a rating study, so no agreement statistic exists. The Spanish versions were not checked by a
   Spanish speaker.

For an independent rating study, `python scripts/audit_tools.py build` writes blinded review workbooks
(`data/audit/review_annotator{1,2}.xlsx`, a 150-item fixed-seed sample; they are released unfilled) and
`python scripts/audit_tools.py ingest A.xlsx B.xlsx` computes agreement statistics from two completed workbooks.
Known ambiguities found after generation are listed in `data/audit/errata.md`.

## Licence
Code: MIT. Task and prompt data: CC BY 4.0. Model outputs are research artefacts of third-party models and remain
subject to those models' terms.

## Citation
See `CITATION.cff`.
