"""Run the fixed single-request experiment sequentially on one GPU.

python run_resume_benchmarks.py --model /path/to/Qwen3-0.6B --output-dir benchmark_results/my_run
"""
import argparse
import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, mode in (("fp16_round1", "off"), ("int8_round1", "on"), ("int8_round2", "on"), ("fp16_round2", "off")):
        output = args.output_dir / f"{name}.json"
        if output.exists():
            raise SystemExit(f"Refusing to overwrite {output}; choose a fresh output directory.")
        print(f"START {name}", flush=True)
        subprocess.run([
            sys.executable, str(Path(__file__).with_name("benchmark_fa2_e2e.py")),
            "--model", args.model, "--kv-quant", mode,
            "--lengths", "512", "1024", "2048", "--decode-tokens", "128",
            "--prefill-repeats", "5", "--decode-repeats", "3", "--warmup-runs", "1",
            "--seed", "20260907", "--gpu-memory-utilization", "0.9", "--output", str(output),
        ], check=True)
        print(f"DONE {name}", flush=True)
