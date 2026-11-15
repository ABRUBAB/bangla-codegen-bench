#!/usr/bin/env bash
# Re-run the whole evaluation after (new) generations have been copied into data/generations/<model>/.
# Usage:  BCB_BACKEND=wsl BCB_WSL_DISTRO=Ubuntu-24.04 bash scripts/final_pipeline.sh
set -euo pipefail
export PYTHONUTF8=1
cd "$(dirname "$0")/.."
python scripts/run_execution.py --gens data/generations --out data/results/exec.jsonl --workers 10 --shard 60
python scripts/static_scan.py --gens data/generations --out data/results/bandit.jsonl
python scripts/analyze.py --exec data/results/exec.jsonl --bandit data/results/bandit.jsonl --gens data/generations --out data/results --boot 2000
python scripts/xai_analysis.py --merged data/results/exec_merged.csv --tokens data/results/token_counts.csv --out data/results
python scripts/make_numbers.py
python scripts/make_extra_tables.py
python scripts/audit_tools.py llm-stats
# failure anatomy: `failure_anatomy.py pairs` + LLM classification (data/analysis/failure_causes_opus.json) are run by hand
[ -f data/analysis/failure_causes_opus.json ] && python scripts/failure_anatomy.py table
python scripts/sensitivity.py
python scripts/clean_translations.py
python scripts/causal_analysis.py
python scripts/make_figures.py
(cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null || true; \
 pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null)
echo "pipeline finished -> paper/main.pdf"
