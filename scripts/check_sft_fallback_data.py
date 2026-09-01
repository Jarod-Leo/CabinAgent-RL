"""Tokenize fallback records before allocating a training GPU."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_sft_fallback import encode_sft_examples
from src.data.trajectory_schema import read_jsonl

TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _expected_tool_calls(record: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message_index, message in enumerate(record.get("messages", [])):
        for call_index, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") if isinstance(call, Mapping) else None
            if not isinstance(function, Mapping):
                raise RuntimeError(
                    f"Non-canonical tool call at messages[{message_index}].tool_calls[{call_index}]"
                )
            arguments = function.get("arguments")
            if not isinstance(arguments, Mapping):
                raise RuntimeError(
                    "Tool arguments must be an object before Qwen templating at "
                    f"messages[{message_index}].tool_calls[{call_index}]"
                )
            calls.append({"name": str(function.get("name") or ""), "arguments": dict(arguments)})
    return calls


def validate_tool_call_round_trip(tokenizer: Any, record: dict[str, Any]) -> int:
    expected = _expected_tool_calls(record)
    if not expected:
        return 0
    rendered = tokenizer.apply_chat_template(
        record.get("messages", []),
        tools=record.get("tools") or None,
        tokenize=False,
        add_generation_prompt=False,
    )
    actual: list[dict[str, Any]] = []
    for block_index, block in enumerate(TOOL_CALL_BLOCK.findall(str(rendered))):
        try:
            payload = json.loads(block)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Rendered tool call {block_index} is not JSON: {block!r}") from exc
        arguments = payload.get("arguments") if isinstance(payload, Mapping) else None
        if not isinstance(arguments, Mapping):
            raise RuntimeError(
                f"Rendered tool call {block_index} arguments are {type(arguments).__name__}, not object"
            )
        actual.append({"name": str(payload.get("name") or ""), "arguments": dict(arguments)})
    if actual != expected:
        raise RuntimeError(f"Qwen tool-call round trip mismatch: expected={expected!r} actual={actual!r}")
    return len(expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/sft_fallback_lora.yaml")
    parser.add_argument("--output", default="reports/sft_fallback_tokenization.json")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / config["model_name_or_path"],
        trust_remote_code=True,
    )
    result = {}
    for split, key in (("train", "train_data"), ("val", "val_data")):
        records = read_jsonl(ROOT / config[key])
        round_trip_calls = sum(validate_tool_call_round_trip(tokenizer, row) for row in records)
        examples, stats = encode_sft_examples(tokenizer, records, int(config["max_length"]))
        lengths = [len(row["input_ids"]) for row in examples]
        result[split] = {
            **stats,
            "min_tokens": min(lengths) if lengths else 0,
            "max_tokens": max(lengths) if lengths else 0,
            "mean_tokens": round(sum(lengths) / len(lengths), 3) if lengths else 0.0,
            "round_trip_tool_calls": round_trip_calls,
        }
    if result["train"]["records"] < int(config["min_train_records"]):
        raise RuntimeError(f"Insufficient fallback records: {result['train']}")
    if result["train"]["examples"] == 0:
        raise RuntimeError(f"No tokenized fallback examples: {result['train']}")
    if result["train"]["round_trip_tool_calls"] == 0:
        raise RuntimeError(f"No train tool calls were round-trip validated: {result['train']}")
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"SFT_DATA_OK output={output} train_examples={result['train']['examples']}")


if __name__ == "__main__":
    main()
