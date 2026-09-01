"""Verify an immutable merged parent and run a one-token GPU load smoke."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.merge_lora_parent import sha256_file


def validate_inventory(model_dir: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    inventory = manifest.get("inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("Parent manifest must contain a non-empty inventory")
    expected_paths = {str(item["path"]) for item in inventory}
    actual_paths = {
        path.relative_to(model_dir).as_posix()
        for path in model_dir.rglob("*")
        if path.is_file() and path.name != "parent_manifest.json"
    }
    if actual_paths != expected_paths:
        raise ValueError(
            f"Parent inventory mismatch: missing={sorted(expected_paths - actual_paths)}, "
            f"extra={sorted(actual_paths - expected_paths)}"
        )

    verified: list[dict[str, Any]] = []
    for item in inventory:
        path = model_dir / str(item["path"])
        size_bytes = path.stat().st_size
        if size_bytes != int(item["size_bytes"]):
            raise ValueError(f"Size mismatch for {path}: {size_bytes} != {item['size_bytes']}")
        digest = sha256_file(path)
        if digest != str(item["sha256"]):
            raise ValueError(f"SHA-256 mismatch for {path}: {digest} != {item['sha256']}")
        verified.append({"path": item["path"], "size_bytes": size_bytes, "sha256": digest})
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model_dir = Path(args.model).resolve()
    manifest_path = model_dir / "parent_manifest.json"
    output = Path(args.output).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing parent manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected a JSON mapping: {manifest_path}")
    verified = validate_inventory(model_dir, manifest)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    inputs = tokenizer("Reply with OK.", return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=1, do_sample=False)

    report = {
        "schema_version": 1,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "model": str(model_dir),
        "source_merge_job_id": manifest.get("slurm_job_id"),
        "inventory_files": len(verified),
        "inventory_bytes": sum(int(item["size_bytes"]) for item in verified),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_dtype": str(next(model.parameters()).dtype),
        "tokenizer_size": len(tokenizer),
        "input_tokens": int(inputs["input_ids"].shape[-1]),
        "generated_tokens": int(generated.shape[-1] - inputs["input_ids"].shape[-1]),
        "status": "PASS",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
