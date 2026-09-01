"""Build simple SFT records from successful trajectories."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any


def _arguments_object(value: Any, *, location: str) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON tool arguments at {location}: {value!r}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(
            f"Tool arguments at {location} must decode to an object, got {type(value).__name__}"
        )
    return copy.deepcopy(dict(value))


def normalize_sft_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return Qwen-template messages with object-valued function arguments."""

    normalized = copy.deepcopy(messages)
    for message_index, message in enumerate(normalized):
        calls = message.get("tool_calls") or []
        if not calls:
            continue
        canonical_calls: list[dict[str, Any]] = []
        for call_index, raw_call in enumerate(calls):
            if not isinstance(raw_call, Mapping):
                raise ValueError(
                    f"Tool call at messages[{message_index}].tool_calls[{call_index}] must be an object"
                )
            call = dict(raw_call)
            location = f"messages[{message_index}].tool_calls[{call_index}]"
            raw_function = call.get("function")
            if isinstance(raw_function, Mapping):
                function = dict(raw_function)
                function["arguments"] = _arguments_object(
                    function.get("arguments"), location=f"{location}.function.arguments"
                )
                call["function"] = function
                call.setdefault("type", "function")
            else:
                function = {
                    "name": str(call.get("name") or ""),
                    "arguments": _arguments_object(
                        call.get("arguments"), location=f"{location}.arguments"
                    ),
                }
                call = {key: value for key, value in call.items() if key in {"id"}}
                call.update({"type": "function", "function": function})
            if not str(call["function"].get("name") or ""):
                raise ValueError(f"Tool call at {location} has no function name")
            canonical_calls.append(call)
        message["tool_calls"] = canonical_calls
    return normalized


def build_sft_records(trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for trajectory in trajectories:
        if float(trajectory.get("metrics", {}).get("success", 0.0)) < 1.0:
            continue
        messages = list(trajectory.get("messages", []))
        response = trajectory.get("model_response", {})
        if response and (not messages or messages[-1] != response):
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": response.get("content", ""),
            }
            tool_calls = response.get("tool_calls", [])
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            messages.append(assistant_message)
        messages = normalize_sft_messages(messages)
        tools = list(trajectory.get("tools", []))
        digest = hashlib.sha256(
            json.dumps(
                {"messages": messages, "tools": tools},
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        trajectory_metadata = trajectory.get("metadata", {})
        records.append(
            {
                "id": trajectory.get("id") or digest[:16],
                "benchmark": trajectory.get("benchmark"),
                "messages": messages,
                "tools": tools,
                "metadata": {
                    "source": "successful_environment_trajectory",
                    "source_file": trajectory_metadata.get("source_file"),
                    "task_id": trajectory_metadata.get("task_id", trajectory.get("id")),
                },
            }
        )
    return records


def split_train_val(
    records: list[dict[str, Any]], val_ratio: float = 0.2, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not records:
        return [], []
    group_keys = sorted(
        {str(row.get("metadata", {}).get("task_id") or row.get("id")) for row in records},
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest(),
    )
    val_group_count = max(1, round(len(group_keys) * val_ratio)) if len(group_keys) > 1 else 0
    val_groups = set(group_keys[:val_group_count])
    train = [
        row
        for row in records
        if str(row.get("metadata", {}).get("task_id") or row.get("id")) not in val_groups
    ]
    val = [
        row
        for row in records
        if str(row.get("metadata", {}).get("task_id") or row.get("id")) in val_groups
    ]
    return train, val
