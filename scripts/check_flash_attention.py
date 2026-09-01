"""Exercise FlashAttention2 and load the exact merged F01 policy parent."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_VERSION = "2.8.3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch
    from flash_attn import flash_attn_func
    from transformers import AutoModelForCausalLM, AutoTokenizer

    installed_version = importlib.metadata.version("flash_attn")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        raise RuntimeError(f"Expected Pro 6000 Blackwell capability (12, 0), got {capability}")

    # Exercise both FA2 forward and backward kernels independently of Transformers.
    qkv = [
        torch.randn(2, 128, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    ]
    output = flash_attn_func(*qkv, causal=True)
    output.float().square().mean().backward()
    kernel_finite = bool(
        torch.isfinite(output).all().item()
        and all(tensor.grad is not None and torch.isfinite(tensor.grad).all().item() for tensor in qkv)
    )
    del output, qkv
    torch.cuda.empty_cache()

    model_path = Path(args.model).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to("cuda")
    encoded = tokenizer("FlashAttention policy load smoke", return_tensors="pt").to("cuda")
    with torch.inference_mode():
        generated = model.generate(**encoded, max_new_tokens=1, do_sample=False)
    generated_tokens = int(generated.shape[-1] - encoded["input_ids"].shape[-1])

    report = {
        "schema_version": 1,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "expected_version": EXPECTED_VERSION,
        "installed_version": installed_version,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(),
        "device_capability": list(capability),
        "kernel_forward_backward_finite": kernel_finite,
        "model_path": str(model_path),
        "model_attn_implementation": model.config._attn_implementation,
        "generated_tokens": generated_tokens,
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    passed = (
        installed_version == EXPECTED_VERSION
        and kernel_finite
        and model.config._attn_implementation == "flash_attention_2"
        and generated_tokens == 1
    )
    report["status"] = "PASS" if passed else "FAIL"
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
