"""Independent graph ablations: off, graph+temporary metadata, graph+CPU reuse."""
import argparse
import subprocess
import sys
from pathlib import Path

MODES = (("graph_off", "off", "off"), ("graph_temp_metadata", "on", "off"), ("graph_staging", "on", "on"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for round_id, modes in ((1, MODES), (2, tuple(reversed(MODES)))):
        for name, graph, staging in modes:
            output = args.output_dir / f"{name}_round{round_id}.json"
            if output.exists():
                raise SystemExit(f"Refusing to overwrite {output}")
            print(f"START {name} round {round_id}", flush=True)
            subprocess.run([
                sys.executable, str(Path(__file__).with_name("benchmark_fa2_e2e.py")),
                "--model", args.model, "--kv-quant", "off", "--rmsnorm-backend", "cuda",
                "--cuda-graph", graph, "--decode-staging", staging,
                "--lengths", "512", "1024", "2048", "--decode-tokens", "128",
                "--prefill-repeats", "5", "--decode-repeats", "3", "--warmup-runs", "1",
                "--seed", "20260907", "--gpu-memory-utilization", "0.9", "--output", str(output),
            ], check=True)
            print(f"DONE {name} round {round_id}", flush=True)
