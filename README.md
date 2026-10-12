# Galahad Benchmark Challenge — Results

| | |
|---|---|
| **Tester** | Abdullah Zulfiqar ([@i-redspector](https://github.com/i-redspector) on GitHub, redspectorlabs@gmail.com) |
| **Hardware** | NVIDIA A40 (46 GB VRAM), 96 vCPU, 503 GB RAM, local NVMe-backed overlay disk (RunPod, on-demand pod) |
| **Model** | Qwen3-4B-AWQ (4-bit AWQ, served via vLLM's OpenAI-compatible API) — chosen as "a comparable LLM" per the brief, sized to fit comfortably alongside the full working set required by the test |
| **vLLM** | 0.31.0 |
| **galahad-kv** | 1.32.0 |

## Summary

Galahad (B) delivers a **3.6x speedup** over plain vLLM (A) on a working set 3x larger than the KV cache, with correctness unaffected.

| | Median revisit time | Correct |
|---|---|---|
| Baseline (A) | 3.29s | 36/36 |
| Galahad (B) | 0.91s | 36/36 |

Full per-prompt data: `results/baseline_A.json`, `results/results_galahad_B_30GB.json`.

## 1. Environment

| | |
|---|---|
| GPU | NVIDIA A40, 46,068 MiB VRAM |
| RAM | 503 GB total |
| Disk | Local overlay filesystem on the pod's NVMe, not a network volume |
| OS | Ubuntu 24.04, CUDA 13.0, driver 580.178.04 |
| Model | Qwen3-4B-AWQ, 4-bit AWQ quantization |
| Serving | vLLM 0.31.0 |

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
export GALAHAD_CACHE_DIR=/workspace/galahad_cache
export GALAHAD_DISK_BUDGET_GB=30
vllm serve /workspace/models/Qwen3-4B-AWQ --served-model-name Qwen3-4B-AWQ \
  --quantization awq --gpu-memory-utilization 0.85 --max-model-len 32768 \
  --kv-transfer-config '{"kv_connector":"GalahadConnector","kv_role":"kv_both"}'
```

with a valid free licence (`galahad doctor` reports everything OK) and `galahad free --org ... --accept-non-commercial` run beforehand.

Same fill/revisit/3-run methodology as baseline.

| Run | Median revisit time |
|---|---|
| 1 | 0.926s |
| 2 | 0.911s |
| 3 | 0.908s |

**Median of medians: 0.91s. Correct: 36/36.**

Galahad's disk cache holds the KV cache for prompts vLLM would otherwise have evicted, so revisits are served from disk instead of fully recomputed — **3.6x faster than baseline** on this working set, with no correctness cost. The disk budget needs to be sized to the actual working set (36 prompts × ~20.5k tokens × ~144 KB/token ≈ 2.9 GB per prompt) — a budget that's too small for the prompt sizes in play limits how many prompts can be cached. See `charts/galahad_revisit_times.png` and `results/results_galahad_B_30GB.json` for the full per-prompt breakdown.

## 6. GPU utilization and power

Logged `nvidia-smi` at 1-second intervals for the full duration of each benchmark (fill + all 3 revisit runs), on a separate pair of runs under identical conditions to isolate this measurement.

| | Baseline (A) | Galahad (B) |
|---|---|---|
| Total energy consumed | 34.81 Wh | 28.03 Wh (**19.5% less**) |
| Avg power draw | 280.8 W | 263.3 W (6.2% lower) |
| Peak power draw | 303.8 W | 304.6 W |
| Avg GPU utilization | 89.7% | 85.1% (5.0% lower) |
| Session duration | 446.2s | 383.2s |

Galahad serves cached revisits from disk instead of recomputing the full forward pass, so the GPU does measurably less work per revisit — lower average utilization and power draw, and the whole job finishes faster, so total energy for the same benchmark drops by about a fifth. Raw per-second logs: `results/gpu_baseline.csv`, `results/gpu_galahad.csv`.

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

# B
export GALAHAD_CACHE_DIR=/path/on/local/disk
export GALAHAD_DISK_BUDGET_GB=30
vllm serve <model-path> --quantization awq --gpu-memory-utilization 0.85 --max-model-len 32768 \
  --kv-transfer-config '{"kv_connector":"GalahadConnector","kv_role":"kv_both"}'

python3 scripts/galahad_bench.py --n-tokens-target 737184 --prompt-tokens 20500 --runs 3 \
  --label galahad-B --out results_galahad_B.json
```

## Files in this repo

- `results/baseline_A.json` — full raw output from the baseline run (per-prompt timing, every run, all 36 correctness checks)
- `results/results_galahad_B_30GB.json` — full raw output from the Galahad run
- `charts/baseline_revisit_times.png` — revisit time per prompt, all 3 baseline runs
- `charts/galahad_revisit_times.png` — revisit time per prompt, all 3 Galahad runs
- `logs/baseline_A_full.log` — full vLLM server log for the baseline run
- `logs/galahad_B_30GB_server.log` — full vLLM server log for the Galahad run, including KV transfer metrics
- `scripts/galahad_bench.py` — the benchmark harness (prompt generation, fill, revisit, timing, JSON output)
