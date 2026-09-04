"""Export and validate a PEFT LoRA adapter from a veRL FSDP checkpoint."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(path: Path) -> list[dict[str, Any]]:
    rows = []
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        if item.name == "adapter_manifest.json":
            continue
        rows.append(
            {
                "path": item.relative_to(path).as_posix(),
                "size_bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    return rows


def validate_adapter_files(adapter: Path, expected_rank: int, expected_alpha: int) -> dict[str, Any]:
    from safetensors import safe_open

    config_path = adapter / "adapter_config.json"
    weights_path = adapter / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError("Exported adapter is missing config or safetensors weights")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if str(config.get("peft_type")) != "LORA":
        raise ValueError(f"Expected LORA adapter, got {config.get('peft_type')!r}")
    if int(config.get("r", -1)) != expected_rank:
        raise ValueError(f"Unexpected adapter rank: {config.get('r')}")
    if int(config.get("lora_alpha", -1)) != expected_alpha:
        raise ValueError(f"Unexpected adapter alpha: {config.get('lora_alpha')}")
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    if not keys or any("lora_" not in key for key in keys):
        raise ValueError("Exported safetensors do not contain a pure LoRA state dict")
    return {"tensor_count": len(keys), "config": config}


def validate_parent_adapter_generation(parent: Path, adapter: Path) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(parent, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        parent,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        attn_implementation="flash_attention_2",
    ).to("cuda")
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    generated = int(output.shape[-1] - inputs["input_ids"].shape[-1])
    peak_memory = int(torch.cuda.max_memory_allocated())
    del output, inputs, model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    if generated != 1:
        raise ValueError(f"Expected one generated token, got {generated}")
    return {"generated_tokens": generated, "peak_cuda_memory_bytes": peak_memory}


def export_adapter(args: argparse.Namespace) -> dict[str, Any]:
    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    actor_dir = args.checkpoint.resolve() / "actor"
    parent = args.parent.resolve()
    output = args.output.resolve()
    if not actor_dir.is_dir() or not parent.is_dir():
        raise FileNotFoundError("Checkpoint actor directory or parent model is missing")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite adapter output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    incoming = output.parent / f".{output.name}.incoming-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    if incoming.exists():
        raise FileExistsError(f"Incoming export path already exists: {incoming}")
    incoming.mkdir()

    try:
        config = ModelMergerConfig(
            operation="merge",
            backend="fsdp",
            target_dir=str(incoming),
            local_dir=str(actor_dir),
            hf_model_config_path=str(actor_dir / "huggingface"),
        )
        merger = FSDPModelMerger(config)
        world_size = merger._get_world_size()
        rank_zero = merger._load_rank_zero_state_dict(world_size)
        mesh, mesh_names = merger._extract_device_mesh_info(rank_zero, world_size)
        del rank_zero
        gc.collect()
        total_shards, mesh_shape = merger._calculate_shard_configuration(mesh, mesh_names)
        state_dict = merger._load_and_merge_state_dicts(
            world_size, total_shards, mesh_shape, mesh_names
        )
        adapter_value = merger.save_lora_adapter(state_dict)
        if not adapter_value:
            raise ValueError("veRL checkpoint did not contain LoRA parameters")
        adapter = Path(adapter_value)
        if not adapter.is_dir():
            raise FileNotFoundError(f"veRL did not create the adapter directory: {adapter}")
        del state_dict, merger
        gc.collect()

        file_check = validate_adapter_files(adapter, args.expected_rank, args.expected_alpha)
        generation = validate_parent_adapter_generation(parent, adapter)
        files = inventory(adapter)
        manifest = {
            "schema_version": 1,
            "status": "PASS",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_checkpoint": str(args.checkpoint.resolve()),
            "source_step": args.step,
            "selection_metric": args.metric_key,
            "selection_score": args.metric_value,
            "tie_break": "earlier",
            "parent_model": str(parent),
            "source_slurm_job": args.source_slurm_job,
            "export_slurm_job": os.environ.get("SLURM_JOB_ID"),
            "adapter_rank": args.expected_rank,
            "adapter_alpha": args.expected_alpha,
            "tensor_count": file_check["tensor_count"],
            "generation_validation": generation,
            "files": files,
            "total_bytes": sum(int(row["size_bytes"]) for row in files),
        }
        (adapter / "adapter_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(adapter, output)
        incoming.rmdir()
        return {**manifest, "output": str(output)}
    except Exception:
        if incoming.exists():
            shutil.rmtree(incoming)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--metric-key", default="val-core/car_bench/reward/mean@1")
    parser.add_argument("--metric-value", type=float, required=True)
    parser.add_argument("--source-slurm-job", required=True)
    parser.add_argument("--expected-rank", type=int, default=32)
    parser.add_argument("--expected-alpha", type=int, default=32)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = export_adapter(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
