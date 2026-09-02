"""Validate the exact F10 parent on veRL's packed, chunked-entropy path."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-size", type=int, default=2048)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from verl.utils import torch_functional as verl_F
    from verl.workers.engine.fsdp import transformer_impl

    if args.chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    capability = torch.cuda.get_device_capability()
    model_path = Path(args.model).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to("cuda")
    base_model.config.use_cache = False
    model = get_peft_model(
        base_model,
        LoraConfig(
            r=32,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    model.train()

    prompts = [
        "Set the cabin temperature to 22 degrees.",
        "The passenger says the rear window is fogging up; call the appropriate cabin tool.",
    ]
    encoded = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")
    outputs = model(**encoded, use_cache=False)
    shifted_logits = outputs.logits[:, :-1, :]
    shifted_labels = encoded["input_ids"][:, 1:]
    valid_mask = encoded["attention_mask"][:, 1:].bool()
    packed_logits = shifted_logits[valid_mask]
    packed_labels = shifted_labels[valid_mask]
    valid_tokens = int(packed_labels.numel())
    if valid_tokens < 2:
        raise RuntimeError("Smoke prompts produced too few valid tokens")

    # This is the same helper selected by FSDPEngine when remove-padding is on.
    entropy = verl_F.entropy_from_logits_with_chunking(
        packed_logits, chunk_size=args.chunk_size
    )
    token_log_probs = torch.log_softmax(packed_logits.float(), dim=-1).gather(
        dim=-1, index=packed_labels.unsqueeze(-1)
    ).squeeze(-1)
    loss = -token_log_probs.mean()
    loss.backward()

    grad_sq = 0.0
    trainable_parameters = 0
    finite_gradients = True
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable_parameters += parameter.numel()
        if "lora_" in name and parameter.grad is not None:
            finite_gradients = finite_gradients and bool(torch.isfinite(parameter.grad).all().item())
            grad_sq += float(parameter.grad.float().square().sum().item())
    lora_grad_norm = math.sqrt(grad_sq)

    engine_source = inspect.getsource(transformer_impl.FSDPEngine.prepare_model_outputs)
    packed_branch_contract = all(
        marker in engine_source
        for marker in (
            "use_remove_padding",
            "entropy_from_logits_with_chunking",
            "entropy_from_logits_chunk_size",
        )
    )
    entropy_finite = bool(torch.isfinite(entropy).all().item())
    log_probs_finite = bool(torch.isfinite(token_log_probs).all().item())
    loss_finite = bool(torch.isfinite(loss).item())
    passed = all(
        (
            capability == (12, 0),
            model.config._attn_implementation == "flash_attention_2",
            packed_branch_contract,
            entropy_finite,
            log_probs_finite,
            loss_finite,
            finite_gradients,
            math.isfinite(lora_grad_norm),
            lora_grad_norm > 0.0,
        )
    )
    report = {
        "schema_version": 1,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "status": "PASS" if passed else "FAIL",
        "model_path": str(model_path),
        "device_name": torch.cuda.get_device_name(),
        "device_capability": list(capability),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "model_attn_implementation": model.config._attn_implementation,
        "lora_rank": 32,
        "lora_alpha": 32,
        "trainable_parameters": trainable_parameters,
        "sequence_lengths": encoded["attention_mask"].sum(dim=-1).tolist(),
        "packed_valid_tokens": valid_tokens,
        "entropy_chunk_size": args.chunk_size,
        "entropy_finite": entropy_finite,
        "log_probs_finite": log_probs_finite,
        "loss_finite": loss_finite,
        "finite_lora_gradients": finite_gradients,
        "lora_gradient_norm": lora_grad_norm,
        "verl_packed_branch_contract": packed_branch_contract,
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
