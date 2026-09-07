"""Summarize all eight RMSNorm ablation processes and matched comparisons."""
import argparse
import json
import statistics
from pathlib import Path
from operator_backends import BACKENDS
from summarize_benchmarks import percentile


def summarize(folder):
    reports = {backend: [json.loads((folder / f"{backend}_round{i}.json").read_text()) for i in (1, 2)] for backend in BACKENDS}
    first = reports["cuda"][0]
    for backend, runs in reports.items():
        for run in runs:
            assert run["metadata"]["operator_backend"]["name"] == backend
            assert run["metadata"]["kv_quant"] is False
            for key in ("model", "model_config_sha256", "benchmark_sha256", "operator_backends_sha256", "git_commit", "seed", "decode_tokens", "prefill_repeats", "decode_repeats", "warmup_runs_per_length", "batch_size", "cuda_graph", "gpu_memory_utilization"):
                assert run["metadata"][key] == first["metadata"][key], key
            for row, ref in zip(run["results"], first["results"]):
                assert row["input_tokens"] == ref["input_tokens"]
                assert row["prefill_prompt_sha256"] == ref["prefill_prompt_sha256"]
                assert [r["prompt_sha256"] for r in row["requests"]] == [r["prompt_sha256"] for r in ref["requests"]]
    rows = []
    for index, original in enumerate(first["results"]):
        row = {"input_tokens": original["input_tokens"], "backends": {}}
        for backend, runs in reports.items():
            source = [run["results"][index] for run in runs]
            requests = [req for src in source for req in src["requests"]]
            steps = [ms for req in requests for ms in req["decode_step_ms"]]
            prefill = [ms for src in source for ms in src["prefill_samples_ms"]]
            row["backends"][backend] = {
                "requests": len(requests), "decode_steps": len(steps),
                "decode_tok_s": 1000 * len(steps) / sum(steps),
                "decode_tok_s_per_round": [src["decode_tok_s_aggregate"] for src in source],
                "prefill_tok_s": 1000 * original["input_tokens"] / statistics.median(prefill),
                "ttft_ms": statistics.median(req["ttft_ms"] for req in requests),
                "tpot_ms_p50": statistics.median(steps), "tpot_ms_p90": percentile(steps, 0.9),
                "e2e_output_tok_s": len(requests) * first["metadata"]["decode_tokens"] * 1000 / sum(req["e2e_ms"] for req in requests),
            }
        pairs = {}
        for label, before, after in (
            ("eager_to_saved_cuda", "torch-eager", "cuda"),
            ("compiled_to_cuda_same_compiled_residual", "torch-compile", "cuda-compiled-residual"),
            ("upstream_norm_settings_to_saved_cuda_including_residual_change", "torch-compile", "cuda"),
        ):
            pairs[label] = {
                "before": before, "after": after,
                "decode_throughput_change_pct": 100 * (row["backends"][after]["decode_tok_s"] / row["backends"][before]["decode_tok_s"] - 1),
                "prefill_throughput_change_pct": 100 * (row["backends"][after]["prefill_tok_s"] / row["backends"][before]["prefill_tok_s"] - 1),
                "e2e_output_throughput_change_pct": 100 * (row["backends"][after]["e2e_output_tok_s"] / row["backends"][before]["e2e_output_tok_s"] - 1),
            }
        row["comparisons"] = pairs
        rows.append(row)
    return {"method": "all runs pooled; 6 full requests per backend/length; FP16 KV cache; only RMSNorm methods selected", "results": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    (args.directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
