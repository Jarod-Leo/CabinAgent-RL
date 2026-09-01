"""Shared task, response, and trajectory structures for baseline runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


JsonDict = dict[str, Any]


@dataclass
class BenchmarkTask:
    """A normalized benchmark item used by CAR-bench and BFCL adapters."""

    id: str
    benchmark: str
    messages: list[JsonDict]
    tools: list[JsonDict]
    expected_tool_calls: list[JsonDict] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)
    split: str = "dev"

    @classmethod
    def from_dict(cls, item: JsonDict, default_benchmark: str) -> "BenchmarkTask":
        return cls(
            id=str(item["id"]),
            benchmark=str(item.get("benchmark", default_benchmark)),
            messages=list(item.get("messages", [])),
            tools=list(item.get("tools", [])),
            expected_tool_calls=[
                normalize_tool_call(call) for call in item.get("expected_tool_calls", [])
            ],
            metadata=dict(item.get("metadata", {})),
            split=str(item.get("split", "dev")),
        )

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass
class ModelResponse:
    """Normalized model output used by evaluation code."""

    content: str
    tool_calls: list[JsonDict] = field(default_factory=list)
    raw: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "content": self.content,
            "tool_calls": [normalize_tool_call(call) for call in self.tool_calls],
            "raw": self.raw,
        }


@dataclass
class Trajectory:
    """A reproducible record for one benchmark interaction."""

    id: str
    benchmark: str
    split: str
    messages: list[JsonDict]
    tools: list[JsonDict]
    expected_tool_calls: list[JsonDict]
    model_response: JsonDict
    predicted_tool_calls: list[JsonDict]
    tool_results: list[JsonDict]
    metrics: JsonDict
    failures: list[JsonDict] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)


def normalize_tool_call(call: JsonDict) -> JsonDict:
    """Keep a small, stable representation for function/tool calls."""

    if not call:
        return {"name": "", "arguments": {}}
    name = call.get("name") or call.get("function", {}).get("name") or ""
    arguments = call.get("arguments")
    if arguments is None and isinstance(call.get("function"), dict):
        arguments = call["function"].get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"_raw": arguments}
    if arguments is None:
        arguments = {}
    return {"name": str(name), "arguments": arguments}


def read_jsonl(path: str | Path) -> list[JsonDict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[JsonDict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSONL: {exc}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[JsonDict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_json(path: str | Path, row: JsonDict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_tasks(path: str | Path, benchmark: str, limit: int | None = None) -> list[BenchmarkTask]:
    rows = read_jsonl(path)
    tasks = [BenchmarkTask.from_dict(row, default_benchmark=benchmark) for row in rows]
    if limit is not None:
        return tasks[:limit]
    return tasks

