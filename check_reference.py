"""Small teacher-forced logits comparison, in separate GPU processes.

Generate a temporary reference with --backend transformers, then compare
--backend nano --kv-quant off/on. This is a smoke check, not a quality eval.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

PROMPTS = [
    "What is the capital of France? Answer with the city name only.",
    "What is 2 + 3? Answer with the number only.",
    "Explain what a GPU does in one short sentence.",
]


def reference(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation="eager", local_files_only=True,
    ).cuda().eval()
    cases = []
    with torch.inference_mode():
        for prompt in PROMPTS:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            ids = tokenizer.encode(text)
            current = torch.tensor([ids], device="cuda")
            cache = None
            logits, generated = [], []
            for _ in range(16):
                result = model(input_ids=current, past_key_values=cache, use_cache=True)
                cache = result.past_key_values
                last = result.logits[0, -1].float()
                assert torch.isfinite(last).all()
                token = last.argmax().item()
                logits.append(last.cpu())
                generated.append(token)
                if token == tokenizer.eos_token_id:
                    break
                current = torch.tensor([[token]], device="cuda")
            cases.append({"prompt": prompt, "input_ids": ids, "generated_ids": generated,
                          "text": tokenizer.decode(generated, skip_special_tokens=True),
                          "logits": torch.stack(logits)})
    args.reference.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cases, args.reference)
    summary = {"backend": "transformers", "dtype": "float16", "attention": "eager", "cases": [
        {k: v for k, v in c.items() if k != "logits"} for c in cases]}
    return summary


def compare(args):
    from nanovllm import LLM, SamplingParams
    from operator_backends import configure_backend
    backend = configure_backend(args.rmsnorm_backend)
    cases = torch.load(args.reference, weights_only=True)
    llm = LLM(args.model, kv_quant=args.kv_quant == "on", enforce_eager=False,
              max_num_seqs=1, max_model_len=512, max_num_batched_tokens=512,
              gpu_memory_utilization=0.9)
    results = []

    class TeacherForcedSampler(torch.nn.Module):
        def __init__(self, case):
            super().__init__()
            self.case = case
            self.steps = []

        def forward(self, logits, temperatures):
            step = len(self.steps)
            got = logits[0].detach().float().cpu()
            expected = self.case["logits"][step]
            finite = bool(torch.isfinite(got).all())
            assert finite, "non-finite inference logits"
            delta = got - expected
            token = self.case["generated_ids"][step]
            self.steps.append({
                "step": step, "phase": "prefill" if step == 0 else "decode",
                "top1_matches_reference": got.argmax().item() == expected.argmax().item(),
                "mean_abs_logit_error": delta.abs().mean().item(),
                "max_abs_logit_error": delta.abs().max().item(),
                "reference_token_logprob_error": abs((got.log_softmax(0)[token] - expected.log_softmax(0)[token]).item()),
                "finite": finite,
            })
            # Both models see exactly the same continuation; errors cannot be
            # attributed to differing sampled text. This is outside timing.
            return torch.tensor([token], device=logits.device)

    with torch.inference_mode():
        for case in cases:
            sampler = TeacherForcedSampler(case)
            llm.model_runner.sampler = sampler
            llm.generate([case["input_ids"]], SamplingParams(max_tokens=len(case["generated_ids"]), ignore_eos=True), use_tqdm=False)
            assert len(sampler.steps) == len(case["generated_ids"])
            results.append({"prompt": case["prompt"], "reference_text": case["text"], "steps": sampler.steps})
    steps = [s for c in results for s in c["steps"]]
    return {"backend": "nano", "operator_backend": backend, "kv_quant": args.kv_quant == "on", "cuda_graph": True,
            "method": "teacher-forced reference tokens, same inputs, single request; not perplexity or free-generation accuracy",
            "compared_steps": len(steps),
            "top1_agreement": sum(s["top1_matches_reference"] for s in steps) / len(steps),
            "mean_abs_logit_error": sum(s["mean_abs_logit_error"] for s in steps) / len(steps),
            "max_abs_logit_error": max(s["max_abs_logit_error"] for s in steps),
            "cases": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("transformers", "nano"), required=True)
    parser.add_argument("--kv-quant", choices=("on", "off"), default="off")
    parser.add_argument("--rmsnorm-backend", choices=("torch-eager", "cuda", "torch-compile", "cuda-compiled-residual"), default="cuda")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20260907)
    result = reference(args) if args.backend == "transformers" else compare(args)
    result["model"] = args.model
    result["torch"] = torch.__version__
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))
