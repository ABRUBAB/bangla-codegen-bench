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

## Route used for the tier-2 models of the paper (llama.cpp, 4-bit GGUF, one RTX 5090)
Hugging Face Transformers with bitsandbytes was too slow for 27-31B models (hybrid-attention kernels fell back to reference
implementations; about 9 h for one model). llama.cpp with continuous batching ran the same prompt set in 0.2-1.2 h per model.
1. Download a `llama-bXXXX-bin-win-cuda-*-x64.zip` build (b12444 was used) and the matching `cudart-*.zip` from the ggml-org/llama.cpp
   releases; unpack into one folder.
2. Download the Q4_K_M GGUF of each model (repositories and file names are in `data/models.json`, key `gguf`).
3. Start the server (GPU memory was kept below about 24 GB so that other jobs on the shared GPU are not starved; the number of parallel
   slots `-np` is the lever: 8 for Qwen3.8 27B, 4 for Gemma-4 31B, 12-16 for the smaller models, each slot with a 3,072-token context):
   `llama-server -m <model>.gguf -ngl 999 -c <3072*np> -np <np> --host 127.0.0.1 --port 8089 --jinja`
4. Run the client with one sample per prompt, the sanitised Qwen3-8B translations and the causal controls:
   `python scripts/generate_local_server.py --base-url http://127.0.0.1:8089/v1 --model <id> --model-id <id> --prompts data/prompts/prompts.jsonl --out data/generations --n-samples 1 --workers <np> --no-translate --external-translations data/generations/qwen3-8b/translations_clean.jsonl --external-condition TF_CLEAN_Q3 --extra-prompts data/prompts/prompts_causal.jsonl --extra-json '{"chat_template_kwargs": {"enable_thinking": false}}'`
   (`enable_thinking` only for the models that have a thinking mode: Qwen3.8 and Gemma-4.) The client is resumable and gives every
   sample its own reproducible seed; re-running with `--n-samples 3 --only DIRECT` adds samples 2 and 3 of the main prompts.
