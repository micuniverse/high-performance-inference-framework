"""End-to-end prefill/decode benchmark for nano-vLLM's FA2 paths.

Run each KV-cache mode in a separate process so CUDA allocations and graph
captures from one mode cannot affect the other mode.
"""

import argparse
import json
import os
import random
import statistics
import time
from pathlib import Path

import torch

from nanovllm import LLM, SamplingParams


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


def run_request(llm: LLM, prompt: list[int], output_tokens: int) -> dict:
    llm.add_request(
        prompt,
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_tokens),
    )
    ttft_ms, _, token_delta = timed_step(llm)
    # LLMEngine.step() computes this after postprocess appends the first
    # sampled completion token, so its prefill counter is prompt + 1.
    assert token_delta == len(prompt) + 1, (token_delta, len(prompt))

    decode_ms = []
    final_output = []
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
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B"))
    parser.add_argument("--kv-quant", choices=("on", "off"), required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--prefill-repeats", type=int, default=5)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    kv_quant = args.kv_quant == "on"
    max_model_len = max(args.lengths) + args.decode_tokens
    llm = LLM(
        args.model,
        enforce_eager=False,
        kv_quant=kv_quant,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        max_num_seqs=1,
    )
    vocab_size = llm.model_runner.config.hf_config.vocab_size

    metadata = {
        "model": args.model,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "kv_quant": kv_quant,
        "decode_tokens": args.decode_tokens,
        "prefill_repeats": args.prefill_repeats,
        "timing_scope": "scheduler + H2D preparation + model + sampler; tokenizer excluded",
    }
    results = []
    case_id = 0
    for length in args.lengths:
        # Prefill-only runs. max_tokens=1 makes each request finish in its
        # prefill step, while still measuring sampling/TTFT end to end.
        prefill_ms = []
        for _ in range(args.prefill_repeats):
            prompt = make_prompt(length, vocab_size, case_id)
            case_id += 1
            measurement = run_request(llm, prompt, output_tokens=1)
            prefill_ms.append(measurement["ttft_ms"])

        # One full request measures decode after its own prefill.
        prompt = make_prompt(length, vocab_size, case_id)
        case_id += 1
        full = run_request(llm, prompt, output_tokens=args.decode_tokens)
        steps = full["decode_step_ms"]
        row = {
            "input_tokens": length,
            "prefill_ms_median": statistics.median(prefill_ms),
            "prefill_ms_min": min(prefill_ms),
            "prefill_ms_p90": percentile(prefill_ms, 0.90),
            "prefill_tok_s": length / (statistics.median(prefill_ms) / 1000.0),
            "ttft_ms": full["ttft_ms"],
            "decode_steps": len(steps),
            "tpot_ms_median": statistics.median(steps) if steps else 0.0,
            "tpot_ms_p90": percentile(steps, 0.90) if steps else 0.0,
            "decode_tok_s": 1000.0 / statistics.median(steps) if steps else 0.0,
            "e2e_ms": full["e2e_ms"],
        }
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    # A fixed natural-language prompt is a smoke test, not part of timing.
    text_prompt = llm.tokenizer.apply_chat_template(
        [{"role": "user", "content": "Introduce yourself in one short sentence."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    text_prompt = llm.tokenizer.encode(text_prompt)
    quality = run_request(llm, text_prompt, output_tokens=32)
    output_ids = quality["output"][0][1] if quality["output"] else []
    decoded = llm.tokenizer.decode(output_ids, skip_special_tokens=True)

    report = {"metadata": metadata, "results": results, "output_smoke_test": decoded}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
