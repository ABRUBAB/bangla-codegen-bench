# Running more models on the lab/center PC (free, no cloud)

Goal: generate code for the benchmark with whatever good models are already installed there, using exactly the study's protocol
(system prompt, temperature 0.2, top-p 0.95, 3 samples, 768 tokens, non-reasoning mode), then copy the results back.
Nothing is sent to the internet if the server is local. The PC may be shared with other work: use `--workers 1` and run only
when the GPU is free; the script is **resumable** (stop and re-run any time, finished samples are skipped).

## 1. Inventory (2 minutes, read-only)
```bash
python scripts/center_pc_probe.py
```
Send the printed report (GPU, RAM, installed models). It tells us which models are usable and how long a run will take.

## 2. Start a local server for one model
Any OpenAI-compatible server works. With Ollama (installed models: `ollama list`):
```bash
ollama serve            # if not already running; default http://localhost:11434
```
LM Studio: *Developer -> Start server* (default http://localhost:1234).

## 3. Smoke test (12 prompts, a few minutes)
```bash
python scripts/generate_local_server.py --base-url http://localhost:11434/v1 --model <name from ollama list> \
   --model-id <short-id> --prompts data/prompts/prompts.jsonl --out data/generations --limit 12 --n-samples 1 --workers 1
```
Open `data/generations/<short-id>/gens_*.jsonl`: `extracted_code` should contain Python code. Delete that folder before the real run.

Models that "think" by default (Qwen3, DeepSeek-R1 style): add `--extra-json '{"think": false}'` (Ollama) so that all models are
non-reasoning at test time; check that `thought_tokens` is 0.

## 4. Full run (direct + translate-first)
```bash
python scripts/generate_local_server.py --base-url http://localhost:11434/v1 --model <name> --model-id <short-id> \
   --prompts data/prompts/prompts.jsonl --out data/generations --workers 1
```
about 7,500 direct + 2,500 translate-first generations. Add `--no-translate` to skip the second arm (about 25 % less time).
Optional causal controls (about 2,000 more generations): `--prompts data/prompts/prompts_causal.jsonl --no-translate`.

## 5. Bring the results back
Copy the folder `data/generations/<short-id>/` (a few MB) to the main machine (USB / cloud drive). Do **not** copy keys or other files.
Then on the main machine: `bash scripts/final_pipeline.sh` executes the code in the sandbox and rebuilds all tables and figures.

## Notes for the paper
Each model needs its exact name and quantization (shown by `ollama show <name>`); record them in `data/models.json`. Quantized local
models are reported as such.
