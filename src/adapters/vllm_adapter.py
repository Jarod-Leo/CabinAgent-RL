"""OpenAI-compatible vLLM adapter skeleton."""

from __future__ import annotations

import json
import urllib.request

from src.adapters.model_adapter import ModelAdapter
from src.data.trajectory_schema import JsonDict, ModelResponse


class VLLMOpenAIAdapter(ModelAdapter):
    """Minimal adapter for a vLLM OpenAI-compatible chat endpoint."""

    name = "vllm_openai"

    def __init__(self, endpoint: str, model: str, timeout: float = 60.0) -> None:
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        messages: list[JsonDict],
        tools: list[JsonDict],
        metadata: JsonDict | None = None,
    ) -> ModelResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": [{"type": "function", "function": tool} for tool in tools],
            "temperature": 0,
            "max_tokens": 1024,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))

        message = data["choices"][0]["message"]
        tool_calls = []
        for call in message.get("tool_calls", []) or []:
            function = call.get("function", {})
            tool_calls.append({"name": function.get("name", ""), "arguments": function.get("arguments", {})})
        return ModelResponse(content=message.get("content") or "", tool_calls=tool_calls, raw=data)

