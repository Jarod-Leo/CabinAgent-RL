"""Merge a corrected PEFT LoRA into an immutable local policy parent."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_adapter_config(adapter_path: Path) -> dict[str, Any]:
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing PEFT adapter config: {config_path}")
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON mapping: {config_path}")
    if value.get("peft_type") != "LORA":
        raise ValueError(f"Expected a LoRA adapter, got {value.get('peft_type')!r}")
    return value


def build_inventory(directory: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == "parent_manifest.json":
            continue
        inventory.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-rank", type=int, default=16)
    args = parser.parse_args()

    base_model = Path(args.base_model).resolve()
    adapter = Path(args.adapter).resolve()
    output = Path(args.output).resolve()
    if not base_model.is_dir():
        raise FileNotFoundError(f"Base model directory does not exist: {base_model}")
    if not adapter.is_dir():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter}")
    if output.exists():
        raise FileExistsError(f"Immutable parent target already exists: {output}")

    adapter_config = load_adapter_config(adapter)
    actual_rank = int(adapter_config.get("r", -1))
    if actual_rank != args.expected_rank:
        raise ValueError(
            f"Unexpected F01 adapter rank: expected {args.expected_rank}, got {actual_rank}"
        )

    job_suffix = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    staging = output.with_name(f".{output.name}.staging-{job_suffix}")
    if staging.exists():
        raise FileExistsError(f"Staging target already exists: {staging}")
    staging.parent.mkdir(parents=True, exist_ok=True)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    merged = peft_model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(
        staging,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    tokenizer.save_pretrained(staging)

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "base_model": str(base_model),
        "adapter": str(adapter),
        "adapter_rank": actual_rank,
        "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
        "merge": "peft.merge_and_unload(safe_merge=True)",
        "dtype": "bfloat16",
        "inventory": build_inventory(staging),
    }
    (staging / "parent_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    staging.rename(output)
    print(output)


if __name__ == "__main__":
    main()
