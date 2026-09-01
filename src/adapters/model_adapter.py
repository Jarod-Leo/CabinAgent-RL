"""Model adapter interfaces and the deterministic local smoke adapter."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from src.data.trajectory_schema import JsonDict, ModelResponse


class ModelAdapter(ABC):
    """Common model adapter contract for local, HF, API, and vLLM backends."""

    name: str

    @abstractmethod
    def generate(
        self,
        messages: list[JsonDict],
        tools: list[JsonDict],
        metadata: JsonDict | None = None,
    ) -> ModelResponse:
        """Generate an assistant response for a benchmark task."""


class LocalRuleBasedAdapter(ModelAdapter):
    """A deterministic adapter for local baseline plumbing tests.

    It is intentionally simple: enough to validate trajectory capture, metrics,
    and reports without requiring a GPU model.
    """

    name = "local_rules"

    def generate(
        self,
        messages: list[JsonDict],
        tools: list[JsonDict],
        metadata: JsonDict | None = None,
    ) -> ModelResponse:
        text = " ".join(str(m.get("content", "")) for m in messages if m.get("role") == "user")
        lower = text.lower()
        tool_names = {tool.get("name", "") for tool in tools}

        if "温度" in text or "冷" in text or "热" in text:
            if "cabin.set_temperature" in tool_names:
                temp = _extract_number(text, default=24)
                seat = "passenger" if "副驾" in text or "乘客" in text else "driver"
                return _call(
                    "已为你调整座舱温度。",
                    "cabin.set_temperature",
                    {"seat": seat, "temperature_c": temp},
                    self.name,
                )

        if "快充" in text or "充电" in text:
            if "navigation.search_poi" in tool_names:
                return _call(
                    "我来查找最近的快充站。",
                    "navigation.search_poi",
                    {"category": "fast_charging", "sort_by": "nearest"},
                    self.name,
                )

        if "天窗" in text:
            if "vehicle.set_sunroof" in tool_names:
                position = "vent" if "一点" in text or "透气" in text else "open"
                return _call(
                    "已调整天窗。",
                    "vehicle.set_sunroof",
                    {"position": position},
                    self.name,
                )

        if "咖啡" in text or "舒服" in text:
            return ModelResponse(
                content="我需要再确认一下你的偏好和目的地范围，再帮你继续处理。",
                tool_calls=[],
                raw={"adapter": self.name, "rule": "clarify_ambiguous_request"},
            )

        if "weather" in lower or "天气" in text:
            if "weather.get_current" in tool_names:
                location = _extract_location(text, default="current_location")
                return _call(
                    "我来查询当前天气。",
                    "weather.get_current",
                    {"location": location},
                    self.name,
                )

        if "add" in lower or "sum" in lower or "加" in text:
            if "math.add" in tool_names:
                numbers = _extract_numbers(text)
                if len(numbers) >= 2:
                    return _call(
                        "我来计算两数之和。",
                        "math.add",
                        {"a": numbers[0], "b": numbers[1]},
                        self.name,
                    )

        if "calendar" in lower or "meeting" in lower or "日程" in text:
            if "calendar.create_event" in tool_names:
                return _call(
                    "我来创建日程。",
                    "calendar.create_event",
                    {"title": "team meeting", "date": "2026-06-21", "time": "09:00"},
                    self.name,
                )

        return ModelResponse(
            content="当前本地基线没有足够信息安全执行这个请求，我会先说明限制并请求补充信息。",
            tool_calls=[],
            raw={"adapter": self.name, "rule": "fallback"},
        )


def create_model_adapter(name: str) -> ModelAdapter:
    normalized = name.strip().lower()
    if normalized in {"local", "local_rules", "rules"}:
        return LocalRuleBasedAdapter()
    raise ValueError(f"Unsupported adapter '{name}'. Available: local_rules")


def _call(content: str, name: str, arguments: JsonDict, adapter_name: str) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=[{"name": name, "arguments": arguments}],
        raw={"adapter": adapter_name, "rule": name},
    )


def _extract_number(text: str, default: int) -> int:
    numbers = _extract_numbers(text)
    return numbers[0] if numbers else default


def _extract_numbers(text: str) -> list[int]:
    return [int(match) for match in re.findall(r"-?\d+", text)]


def _extract_location(text: str, default: str) -> str:
    patterns = [
        r"in ([A-Za-z\s]+)",
        r"for ([A-Za-z\s]+)",
        r"在([\u4e00-\u9fffA-Za-z\s]+?)(?:的)?天气",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return default

