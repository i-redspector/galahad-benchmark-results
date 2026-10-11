# Galahad Benchmark Challenge — Results

**Tester:** Abdullah Zulfiqar ([@redspector](https://www.freelancer.com/u/redspector) on Freelancer.com)
**Hardware:** NVIDIA A40 (46 GB VRAM), 96 vCPU, 503 GB RAM, local NVMe-backed overlay disk (RunPod, on-demand pod)
**Model:** Qwen3-4B-AWQ (4-bit AWQ, served via vLLM's OpenAI-compatible API) — chosen as "a comparable LLM" per the brief, sized to fit comfortably alongside the full working set required by the test
**vLLM versions tried:** 0.31.0 and 0.30.0 (Galahad's own docs list 0.30 as last verified)

## Summary

Baseline vLLM (no Galahad) ran clean: 36 unique prompts, 738,909 tokens total, 3 full runs, **36/36 correct answers, median revisit time 3.29s**. Full results are in `results/baseline_A.json`.

I could not get a working Galahad (B) run. Every attempt — across two different machines (a local WSL2 desktop and this RunPod A40), two vLLM versions, several `--gpu-memory-utilization` and `GALAHAD_DISK_BUDGET_GB` settings — crashed at the same point: allocating the KV cache block tables, with `CUDA out of memory`, even with 40+ GB genuinely free on the GPU. The evidence in `logs/` and below points at a bug in `GalahadConnector` itself, not at my setup. Details below.

I'm sending this because the brief asked for the comparison, and an honest "it doesn't start" is more useful to you than a result I made up. If there's a specific flag or environment I'm missing, I'd genuinely like to know — the baseline numbers alone show the model and harness both work fine.

## 1. Environment

| | |
|---|---|
| GPU | NVIDIA A40, 46,068 MiB VRAM |
| RAM | 503 GB total |
| Disk | Local overlay filesystem on the pod's NVMe, not a network volume |
| OS | Ubuntu 24.04, CUDA 13.0, driver 580.178.04 |
| Model | Qwen3-4B-AWQ, 4-bit AWQ quantization |
| Serving | vLLM 0.31.0 (primary) and 0.30.0 (retried, since Galahad's docs mention 0.30 as the last verified version) |

I also ran the whole pipeline earlier on a local desktop (RTX 5060 Ti, 8GB VRAM, WSL2) to work out the methodology before moving to a GPU with enough headroom to rule out VRAM size as a factor. The Galahad crash showed up identically on both machines, which is part of why I don't think this is environment-specific.

## 2. Step 1 — finding N

Started vLLM against Qwen3-4B-AWQ with `--max-model-len 32768` and `--gpu-memory-utilization 0.85`:

```
GPU KV cache size: 245,728 tokens, Maximum concurrency for 32,768 tokens per request: 7.50x
```

**N = 245,728 tokens.**

## 3. Step 2 — the working set

3×N = 737,184 tokens minimum. I generated 36 unique prompts, each built from random filler text (no copy-paste between prompts), each with one embedded 8-character secret code at a random position, each ending in "What is the secret code? Answer with only the code." Total: **738,909 tokens** — just over the 3×N floor.

The prompt generator is in `scripts/galahad_bench.py`. It uses the model's own tokenizer to hit the target token count precisely, not a word-count estimate.

## 4. Step 3 — baseline (A)

```
vllm serve /workspace/models/Qwen3-4B-AWQ --served-model-name Qwen3-4B-AWQ \
  --quantization awq --gpu-memory-utilization 0.85 --max-model-len 32768
```

Fill: all 36 prompts sent, 4 concurrent. Revisit: first 12 prompts (first third) re-sent sequentially. Repeated 3 times.

| Run | Median revisit time |
|---|---|
| 1 | 3.290s |
| 2 | 3.290s |
| 3 | 3.306s |

**Median of medians: 3.29s. Correct: 36/36.**

Since the working set (738,909 tokens) is roughly 3x the KV cache capacity, plain vLLM has to evict and recompute the earlier prompts by the time it gets back to them — that's exactly what the brief predicted, and the timing is consistent with a real recompute (close to the original fill time per prompt, not a cache hit). See `charts/baseline_revisit_times.png` and `results/baseline_A.json` for the full per-prompt breakdown.

## 5. Step 3 — Galahad (B)

```
vllm serve /workspace/models/Qwen3-4B-AWQ --served-model-name Qwen3-4B-AWQ \
  --quantization awq --gpu-memory-utilization 0.85 --max-model-len 32768 \
  --kv-transfer-config '{"kv_connector":"GalahadConnector","kv_role":"kv_both"}'
```

with `GALAHAD_CACHE_DIR` on local disk, a valid free licence (`galahad doctor` reports everything OK), and `galahad free --org ... --accept-non-commercial` run successfully beforehand.

The server gets as far as computing the same KV cache size as the baseline run, registers the Galahad connector, then crashes while allocating the KV cache's block tables:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 33.75 GiB. GPU 0 has a
total capacity of 44.42 GiB of which 41.27 GiB is free. Process 4056363 has 3.14 GiB
memory in use. Of the allocated memory 2.68 GiB is allocated by PyTorch...
```

This happened every time, regardless of:

- `--gpu-memory-utilization`: tried 0.70, 0.75, 0.80, 0.85
- `GALAHAD_DISK_BUDGET_GB`: tried 1, 5, 10, 20
- vLLM version: 0.31.0 and 0.30.0
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — vLLM itself rejects this combination with `GalahadConnector` unless `--enable-cumem-allocator` is also set
- `--enable-cumem-allocator` — allowed the config through, but then `[galahad] register_kv_caches failed (can't start new thread)` and the crash moved one step further down the same code path

## 6. What the crash is actually telling us

The one test that matters: I dropped `--max-model-len` from 32768 to 4096 — an 8x smaller context, which should need a dramatically smaller KV cache — and kept everything else the same.

| `--max-model-len` | Attempted allocation at crash |
|---|---|
| 32,768 | 33.75 GiB |
| 4,096 | 29.31 GiB |

An 8x drop in context length only reduced the requested allocation by about 13%. See `charts/galahad_crash_evidence.png`.

That's the actual finding: whatever size `GalahadConnector` is reporting to vLLM for its block tables isn't being driven by `--max-model-len` the way it should be. Plain vLLM, with the exact same flags minus the connector, allocates normally and serves the full baseline run without issue — so the GPU, the driver, and vLLM's own KV cache math are all fine. The ~30 GB request shows up specifically once `GalahadConnector` is in the picture.

I don't have visibility into why — `galahad_vllm/connector.*.so` is a compiled, closed-source module, so I can't trace the actual size calculation. Full raw logs for both the `max_model_len=32768` and `max_model_len=4096` runs are in `logs/`, including the "disk budget" lines Galahad prints just before the crash, in case that's a useful thread to pull on.

## 7. Reproducing this

```bash
git clone https://github.com/i-redspector/galahad-benchmark-results
cd galahad-benchmark-results
pip install vllm galahad-kv
galahad free --org <your-org> --email you@example.com --accept-non-commercial

# A
vllm serve <model-path> --quantization awq --gpu-memory-utilization 0.85 --max-model-len 32768

# in another terminal, once A is up:
python3 scripts/galahad_bench.py --n-tokens-target 737184 --prompt-tokens 20500 --runs 3 \
  --label baseline-A --out results_baseline_A.json

# B (crashes before it reaches the point where galahad_bench.py could run)
export GALAHAD_CACHE_DIR=/path/on/local/disk
vllm serve <model-path> --quantization awq --gpu-memory-utilization 0.85 --max-model-len 32768 \
  --kv-transfer-config '{"kv_connector":"GalahadConnector","kv_role":"kv_both"}'
```

## Files in this repo

- `results/baseline_A.json` — full raw output from the benchmark script (per-prompt timing, every run, all 36 correctness checks)
- `charts/baseline_revisit_times.png` — revisit time per prompt, all 3 baseline runs
- `charts/galahad_crash_evidence.png` — the allocation-size-vs-context-length comparison
- `logs/baseline_A_full.log` — full vLLM server log for the working baseline run
- `logs/galahad_crash_maxlen32768.log` — full log of the Galahad crash at the original settings
- `logs/galahad_crash_maxlen4096.log` — full log of the diagnostic run with an 8x smaller context
- `scripts/galahad_bench.py` — the benchmark harness (prompt generation, fill, revisit, timing, JSON output)
