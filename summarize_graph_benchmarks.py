"""Aggregate six independent Graph/staging ablation processes without selection."""
import argparse
import json
import statistics
from pathlib import Path
from summarize_benchmarks import percentile

MODES = {"graph_off": (False, False), "graph_temp_metadata": (True, False), "graph_staging": (True, True)}


def summarize(folder, selected=None):
    selected = tuple(MODES) if selected is None else tuple(selected)
    reports = {name: [json.loads((folder / f"{name}_round{i}.json").read_text()) for i in (1, 2)] for name in selected}
    first = reports[selected[0]][0]
    for name, runs in reports.items():
        for run in runs:
            meta = run["metadata"]
            assert (meta["cuda_graph"], meta["decode_staging"]) == MODES[name]
            for key in ("model", "model_config_sha256", "benchmark_sha256", "operator_backends_sha256", "operator_backend", "git_commit", "seed", "decode_tokens", "prefill_repeats", "decode_repeats", "warmup_runs_per_length", "batch_size", "gpu_memory_utilization", "kv_quant"):
                assert meta[key] == first["metadata"][key], key
            assert len(run["results"]) == len(first["results"])
            for row, ref in zip(run["results"], first["results"]):
                assert row["input_tokens"] == ref["input_tokens"]
                assert row["prefill_prompt_sha256"] == ref["prefill_prompt_sha256"]
                assert [r["prompt_sha256"] for r in row["requests"]] == [r["prompt_sha256"] for r in ref["requests"]]
    rows = []
    for index, original in enumerate(first["results"]):
        row = {"input_tokens": original["input_tokens"], "modes": {}}
        for name, runs in reports.items():
            source = [run["results"][index] for run in runs]
            requests = [r for src in source for r in src["requests"]]
            steps = [ms for r in requests for ms in r["decode_step_ms"]]
            prefill = [ms for src in source for ms in src["prefill_samples_ms"]]
            row["modes"][name] = {
                "requests": len(requests), "decode_steps": len(steps),
                "decode_tok_s": 1000 * len(steps) / sum(steps),
                "decode_tok_s_per_round": [src["decode_tok_s_aggregate"] for src in source],
                "prefill_tok_s": 1000 * original["input_tokens"] / statistics.median(prefill),
                "ttft_ms": statistics.median(r["ttft_ms"] for r in requests),
                "tpot_ms_p50": statistics.median(steps), "tpot_ms_p90": percentile(steps, .9),
                "e2e_output_tok_s": len(requests) * first["metadata"]["decode_tokens"] * 1000 / sum(r["e2e_ms"] for r in requests),
                "peak_allocated_mib_per_round": [src["memory"]["peak_allocated_bytes"] / 2**20 for src in source],
            }
        pairs = {}
        for name, before, after in (
            ("graph_with_temp_metadata_vs_off", "graph_off", "graph_temp_metadata"),
            ("graph_with_staging_vs_off", "graph_off", "graph_staging"),
            ("staging_on_existing_graph", "graph_temp_metadata", "graph_staging"),
        ):
            if before not in row["modes"] or after not in row["modes"]:
                continue
            pairs[name] = {"before": before, "after": after,
                "decode_throughput_change_pct": 100 * (row["modes"][after]["decode_tok_s"] / row["modes"][before]["decode_tok_s"] - 1),
                "e2e_output_throughput_change_pct": 100 * (row["modes"][after]["e2e_output_tok_s"] / row["modes"][before]["e2e_output_tok_s"] - 1)}
        row["comparisons"] = pairs
        rows.append(row)
    return {"method": "all selected modes, two processes each, fixed FP16 KV cache and the same RMSNorm backend; pooled Decode tokens / pooled time", "results": rows}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    args = p.parse_args()
    result = summarize(args.directory, args.modes)
    (args.directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
