"""End-to-end prefill/decode benchmark for nano-vLLM's FA2 paths.

Run each KV-cache mode in a separate process so CUDA allocations and graph
captures from one mode cannot affect the other mode.
"""

import argparse
import hashlib
import json
import os
import random
import statistics
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from nanovllm import LLM, SamplingParams
from operator_backends import BACKENDS, configure_backend


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def timed_step(llm: LLM):
    torch.cuda.synchronize()
    start = time.perf_counter()
    output, token_delta = llm.step()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0, output, token_delta


def make_prompt(length: int, vocab_size: int, case_id: int) -> list[int]:
    # Distinct deterministic prompts prevent prefix-cache hits between repeats.
    rng = random.Random(20260824 + length * 1009 + case_id)
    return [rng.randrange(1, vocab_size) for _ in range(length)]


def run_request(llm: LLM, prompt: list[int], output_tokens: int, ignore_eos=True) -> dict:
    llm.add_request(
        prompt,
        SamplingParams(temperature=0.6, ignore_eos=ignore_eos, max_tokens=output_tokens),
    )
    ttft_ms, final_output, token_delta = timed_step(llm)
    # LLMEngine.step() computes this after postprocess appends the first
    # sampled completion token, so its prefill counter is prompt + 1.
    assert token_delta == len(prompt) + 1, (token_delta, len(prompt))

    decode_ms = []
    while not llm.is_finished():
        elapsed_ms, output, token_delta = timed_step(llm)
        assert token_delta == -1, token_delta
        decode_ms.append(elapsed_ms)
        if output:
            final_output = output

    return {
        "ttft_ms": ttft_ms,
        "decode_step_ms": decode_ms,
        "e2e_ms": ttft_ms + sum(decode_ms),
        "output": final_output,
        "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B"))
    parser.add_argument("--kv-quant", choices=("on", "off"), required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--prefill-repeats", type=int, default=5)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--decode-repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--rmsnorm-backend", choices=BACKENDS, default="cuda")
    parser.add_argument("--cuda-graph", choices=("on", "off"), default="on")
    parser.add_argument("--decode-staging", choices=("on", "off"), default="on")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.lengths) < 1 or args.decode_tokens < 2 or min(args.prefill_repeats, args.decode_repeats, args.warmup_runs) < 1:
        parser.error("positive lengths/repeats/warmups and at least two decode tokens are required")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    started = datetime.now(timezone.utc).isoformat()
    backend = configure_backend(args.rmsnorm_backend)

    kv_quant = args.kv_quant == "on"
    max_model_len = max(args.lengths) + args.decode_tokens
    llm = LLM(
        args.model,
        enforce_eager=args.cuda_graph == "off",
        kv_quant=kv_quant,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    # Disable only CPU staging reuse; the existing Graph fallback still uses
    # persistent GPU capture buffers and creates temporary metadata tensors.
    if args.decode_staging == "off" and hasattr(llm.model_runner, "decode_staging"):
        del llm.model_runner.decode_staging
    vocab_size = llm.model_runner.config.hf_config.vocab_size

    metadata = {
        "schema_version": 2,
        "operator_backend": backend,
        "operator_backends_sha256": hashlib.sha256(Path(__file__).with_name("operator_backends.py").read_bytes()).hexdigest(),
        "started_utc": started,
        "model": args.model,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "kv_quant": kv_quant,
        "decode_tokens": args.decode_tokens,
        "prefill_repeats": args.prefill_repeats,
        "decode_repeats": args.decode_repeats,
        "warmup_runs_per_length": args.warmup_runs,
        "seed": args.seed,
        "batch_size": 1,
        "cuda_graph": not llm.model_runner.enforce_eager,
        "decode_staging": hasattr(llm.model_runner, "decode_staging"),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "cuda_runtime": torch.version.cuda,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_config_sha256": hashlib.sha256((Path(args.model) / "config.json").read_bytes()).hexdigest(),
        "timing_scope": "scheduler + H2D preparation + model + sampler; tokenizer excluded",
    }
    runner = llm.model_runner
    cache_bytes = runner.kv_cache.numel() * runner.kv_cache.element_size()
    scale_bytes = runner.kv_scale.numel() * runner.kv_scale.element_size() if kv_quant else 0
    slots = runner.config.num_kvcache_blocks * runner.block_size
    metadata["cache"] = {
        "blocks": runner.config.num_kvcache_blocks,
        "block_size": runner.block_size,
        "capacity_tokens": slots,
        "kv_tensor_bytes": cache_bytes,
        "scale_tensor_bytes": scale_bytes,
        "bytes_per_token_all_layers": (cache_bytes + scale_bytes) // slots,
        "allocation_policy": "automatic cache capacity at the same GPU memory utilization; capacity may vary by process",
    }
    results = []
    case_id = 0
    for length in args.lengths:
        for warmup in range(args.warmup_runs):
            run_request(llm, make_prompt(length, vocab_size, -100 - warmup), args.decode_tokens)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated_before = torch.cuda.memory_allocated()
        # Prefill-only runs. max_tokens=1 makes each request finish in its
        # prefill step, while still measuring sampling/TTFT end to end.
        prefill_ms = []
        prefill_prompts = []
        for _ in range(args.prefill_repeats):
            prompt = make_prompt(length, vocab_size, case_id)
            case_id += 1
            measurement = run_request(llm, prompt, output_tokens=1)
            prefill_ms.append(measurement["ttft_ms"])
            prefill_prompts.append(measurement["prompt_sha256"])

        # Independent requests measure decode after their own prefill.
        full_runs = []
        for _ in range(args.decode_repeats):
            prompt = make_prompt(length, vocab_size, case_id)
            case_id += 1
            full = run_request(llm, prompt, output_tokens=args.decode_tokens)
            assert len(full["decode_step_ms"]) == args.decode_tokens - 1
            assert len(full["output"][0][1]) == args.decode_tokens
            full_runs.append(full)
        steps = [ms for full in full_runs for ms in full["decode_step_ms"]]
        row = {
            "input_tokens": length,
            "prefill_ms_median": statistics.median(prefill_ms),
            "prefill_ms_min": min(prefill_ms),
            "prefill_ms_p90": percentile(prefill_ms, 0.90),
            "prefill_tok_s": length / (statistics.median(prefill_ms) / 1000.0),
            "ttft_ms": statistics.median(full["ttft_ms"] for full in full_runs),
            "decode_steps": len(steps),
            "tpot_ms_median": statistics.median(steps) if steps else 0.0,
            "tpot_ms_p90": percentile(steps, 0.90) if steps else 0.0,
            "decode_tok_s": 1000.0 / statistics.median(steps) if steps else 0.0,
            "e2e_ms": statistics.median(full["e2e_ms"] for full in full_runs),
            "decode_tok_s_aggregate": len(steps) * 1000.0 / sum(steps),
            "e2e_output_tok_s": args.decode_repeats * args.decode_tokens * 1000.0 / sum(full["e2e_ms"] for full in full_runs),
            "memory": {
                "allocated_before_bytes": allocated_before,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "scope": "PyTorch allocator including preallocated cache and CUDA graphs; excludes non-PyTorch allocations",
            },
            "prefill_samples_ms": prefill_ms,
            "prefill_prompt_sha256": prefill_prompts,
            "requests": [{k: v for k, v in full.items() if k != "output"} for full in full_runs],
        }
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    # A fixed natural-language prompt is a smoke test, not part of timing.
    text_prompt = llm.tokenizer.apply_chat_template(
        [{"role": "user", "content": "Introduce yourself in one short sentence."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    text_prompt = llm.tokenizer.encode(text_prompt)
    quality = run_request(llm, text_prompt, output_tokens=64, ignore_eos=False)
    output_ids = quality["output"][0][1] if quality["output"] else []
    decoded = llm.tokenizer.decode(output_ids, skip_special_tokens=True)

    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report = {"metadata": metadata, "results": results, "output_smoke_test": decoded}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
