"""
Galahad Benchmark Challenge — A/B test harness.

Generates unique 20,000+ token prompts (each carrying a unique secret code),
fills a vLLM server with them (4 concurrent at a time), then re-sends the
first third ("revisit") and measures time + correctness.

Run once against the baseline server (port 8000) and once against the
Galahad-enabled server (also port 8000, but started with the kv-transfer
config) — restart the server between runs, this script just points at
whatever's listening on BASE_URL.
"""
import argparse
import json
import random
import statistics
import string
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from transformers import AutoTokenizer

MODEL_DIR = "/workspace/models/Qwen3-4B-AWQ"
MODEL_NAME = "Qwen3-4B-AWQ"
BASE_URL = "http://localhost:8000/v1"

# A fixed vocabulary of filler words to build unique, high-entropy prompts from.
# Not copy-pasted text — each prompt is an independent random draw.
_WORDS = (
    "river mountain copper silence orbit lantern velvet granite whisper "
    "horizon ember cascade thistle marble quartz driftwood canyon ember "
    "glacier meadow coral basalt amber tundra ridge delta prairie reef "
    "falcon otter heron lynx badger wren sparrow ibex marten vole "
    "engine ledger compass anchor beacon valve piston gauge rivet lever "
    "autumn solstice zenith nadir equinox twilight dawn dusk noon midnight "
    "signal current voltage frequency amplitude resonance vector tensor matrix scalar"
).split()


def make_unique_block(rng: random.Random, tokenizer, target_tokens: int) -> str:
    """Build a block of roughly target_tokens tokens of random filler text."""
    words = []
    # Overshoot word count, then trim precisely by token count.
    approx_words_needed = int(target_tokens * 0.8)
    for _ in range(approx_words_needed):
        words.append(rng.choice(_WORDS))
        if rng.random() < 0.12:
            words[-1] += "."
    text = " ".join(words)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < target_tokens:
        # pad with more random words until long enough
        while len(ids) < target_tokens:
            text += " " + rng.choice(_WORDS)
            ids = tokenizer.encode(text, add_special_tokens=False)
    else:
        ids = ids[:target_tokens]
        text = tokenizer.decode(ids)
    return text


def random_code(rng: random.Random, length: int = 8) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(length))


def build_prompts(n_prompts: int, tokens_per_prompt: int, seed: int, tokenizer):
    rng = random.Random(seed)
    prompts = []
    for i in range(n_prompts):
        code = random_code(rng)
        block = make_unique_block(rng, tokenizer, tokens_per_prompt)
        # Insert the secret code at a random position inside the block (not just at the end)
        words = block.split(" ")
        insert_at = rng.randint(len(words) // 4, 3 * len(words) // 4)
        words.insert(insert_at, f"The secret code is: {code}.")
        full_text = " ".join(words) + "\n\nWhat is the secret code? Answer with only the code."
        token_count = len(tokenizer.encode(full_text, add_special_tokens=False))
        prompts.append({"id": i, "code": code, "text": full_text, "tokens": token_count})
    return prompts


def send_chat(prompt_text: str, timeout: int = 180) -> dict:
    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": 32,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
        elapsed = time.time() - t0
        answer = body["choices"][0]["message"]["content"]
        return {"ok": True, "elapsed": elapsed, "answer": answer}
    except Exception as e:
        return {"ok": False, "elapsed": time.time() - t0, "error": str(e)}


def fill_phase(prompts, concurrency: int = 4):
    print(f"[fill] sending {len(prompts)} prompts, {concurrency} at a time...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(send_chat, p["text"]): p for p in prompts}
        for fut in as_completed(futures):
            p = futures[fut]
            res = fut.result()
            status = "ok" if res["ok"] else f"FAIL: {res.get('error')}"
            print(f"  prompt {p['id']} ({p['tokens']} tok) -> {res['elapsed']:.2f}s [{status}]")
    print(f"[fill] total wall time: {time.time()-t0:.2f}s")


def revisit_phase(prompts):
    n_revisit = max(1, -(-len(prompts) // 3))  # ceil(n/3)
    revisit_set = prompts[:n_revisit]
    print(f"[revisit] re-sending first {n_revisit} prompt(s) sequentially...")
    results = []
    for p in revisit_set:
        res = send_chat(p["text"])
        correct = res.get("ok") and p["code"] in (res.get("answer") or "")
        results.append({"id": p["id"], "elapsed": res["elapsed"], "correct": correct,
                         "answer": res.get("answer"), "expected": p["code"]})
        print(f"  prompt {p['id']}: {res['elapsed']:.2f}s, correct={correct}, "
              f"answer={res.get('answer')!r} expected={p['code']}")
    return results


def run_once(prompts, concurrency: int):
    fill_phase(prompts, concurrency)
    return revisit_phase(prompts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tokens-target", type=int, default=82389, help="3xN total token budget")
    ap.add_argument("--prompt-tokens", type=int, default=20500)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    n_prompts = max(1, -(-args.n_tokens_target // args.prompt_tokens))  # ceil
    print(f"Building {n_prompts} unique prompts of ~{args.prompt_tokens} tokens each "
          f"(target total {args.n_tokens_target})...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    prompts = build_prompts(n_prompts, args.prompt_tokens, seed=42, tokenizer=tokenizer)
    total_tokens = sum(p["tokens"] for p in prompts)
    print(f"Actual total tokens across {len(prompts)} prompts: {total_tokens}")
    for p in prompts:
        print(f"  prompt {p['id']}: {p['tokens']} tokens, code={p['code']}")

    all_runs = []
    for r in range(args.runs):
        print(f"\n=== {args.label} run {r+1}/{args.runs} ===")
        revisit_results = run_once(prompts, concurrency=4)
        all_runs.append(revisit_results)

    # Aggregate: median revisit time per prompt across runs, overall correctness
    medians = []
    all_correct = []
    for run in all_runs:
        times = [x["elapsed"] for x in run]
        medians.append(statistics.median(times))
        all_correct.extend(x["correct"] for x in run)

    summary = {
        "label": args.label,
        "n_prompts": len(prompts),
        "total_tokens": total_tokens,
        "prompt_token_counts": [p["tokens"] for p in prompts],
        "runs": all_runs,
        "median_revisit_time_per_run": medians,
        "median_of_medians": statistics.median(medians),
        "correct_count": sum(all_correct),
        "total_answers": len(all_correct),
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\n=== SUMMARY ({args.label}) ===")
    print(f"Median revisit time per run: {medians}")
    print(f"Median of medians: {summary['median_of_medians']:.3f}s")
    print(f"Correct: {summary['correct_count']}/{summary['total_answers']}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
