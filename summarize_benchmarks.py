"""Aggregate all four predeclared runs; never select the best run."""
import argparse
import json
import statistics
from pathlib import Path


def percentile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def summarize(folder):
    reports = {mode: [json.loads((folder / f"{mode}_round{i}.json").read_text()) for i in (1, 2)]
               for mode in ("fp16", "int8")}
    baseline = reports["fp16"][0]
    all_reports = [r for group in reports.values() for r in group]
    for report in all_reports:
        for key in ("model", "model_config_sha256", "benchmark_sha256", "seed", "decode_tokens", "prefill_repeats", "decode_repeats", "warmup_runs_per_length", "batch_size", "cuda_graph", "gpu_memory_utilization"):
            assert report["metadata"][key] == baseline["metadata"][key], key
        assert [r["input_tokens"] for r in report["results"]] == [r["input_tokens"] for r in baseline["results"]]
        for row, ref in zip(report["results"], baseline["results"]):
            assert row["prefill_prompt_sha256"] == ref["prefill_prompt_sha256"]
            assert [r["prompt_sha256"] for r in row["requests"]] == [r["prompt_sha256"] for r in ref["requests"]]

    rows = []
    for index, original in enumerate(baseline["results"]):
        row = {"input_tokens": original["input_tokens"]}
        for mode in reports:
            source = [r["results"][index] for r in reports[mode]]
            requests = [request for result in source for request in result["requests"]]
            prefill = [ms for result in source for ms in result["prefill_samples_ms"]]
            steps = [ms for req in requests for ms in req["decode_step_ms"]]
            row[mode] = {
                "requests": len(requests), "decode_steps": len(steps),
                "prefill_ms_median": statistics.median(prefill),
                "prefill_tok_s": original["input_tokens"] * 1000 / statistics.median(prefill),
                "ttft_ms_median": statistics.median(r["ttft_ms"] for r in requests),
                "tpot_ms_median": statistics.median(steps), "tpot_ms_p90": percentile(steps, 0.9),
                "decode_tok_s_aggregate": len(steps) * 1000 / sum(steps),
                "decode_tok_s_inverse_median": 1000 / statistics.median(steps),
                "decode_tok_s_per_round": [result["decode_tok_s_aggregate"] for result in source],
                "e2e_output_tok_s": len(requests) * baseline["metadata"]["decode_tokens"] * 1000 / sum(r["e2e_ms"] for r in requests),
                "peak_allocated_mib_per_round": [result["memory"]["peak_allocated_bytes"] / 2**20 for result in source],
            }
        rows.append(row)
    cache = {}
    for mode in reports:
        sizes = {r["metadata"]["cache"]["bytes_per_token_all_layers"] for r in reports[mode]}
        assert len(sizes) == 1
        cache[mode] = {"bytes_per_token_all_layers": sizes.pop(),
                       "capacity_tokens_per_round": [r["metadata"]["cache"]["capacity_tokens"] for r in reports[mode]]}
    cache["per_token_storage_reduction_pct"] = 100 * (1 - cache["int8"]["bytes_per_token_all_layers"] / cache["fp16"]["bytes_per_token_all_layers"])
    return {"method": "pooled all requests across both independent processes per mode; aggregate decode tokens / total measured decode time",
            "cache": cache, "results": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    (args.directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
