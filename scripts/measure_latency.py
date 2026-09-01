"""Measure simple OpenAI-compatible chat completion latency."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request


def main() -> None:
    args = parse_args()
    latencies: list[float] = []
    for idx in range(args.requests):
        start = time.perf_counter()
        call_chat_completion(args.endpoint, args.model, args.prompt, args.timeout)
        latencies.append(time.perf_counter() - start)
        print(f"request={idx + 1} latency_s={latencies[-1]:.4f}")

    latencies_sorted = sorted(latencies)
    p50 = percentile(latencies_sorted, 50)
    p95 = percentile(latencies_sorted, 95)
    report = (
        "# Latency Report\n\n"
        f"- Endpoint: `{args.endpoint}`\n"
        f"- Model: `{args.model}`\n"
        f"- Requests: {len(latencies)}\n"
        f"- Mean latency: {statistics.mean(latencies):.4f}s\n"
        f"- P50 latency: {p50:.4f}s\n"
        f"- P95 latency: {p95:.4f}s\n"
    )
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(f"Wrote {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default="cabinagent-qwen2.5-7b")
    parser.add_argument("--prompt", default="帮我导航到最近的快充站。")
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", default="reports/latency_report.md")
    return parser.parse_args()


def call_chat_completion(endpoint: str, model: str, prompt: str, timeout: float) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "temperature": 0,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()


def percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    idx = min(len(values) - 1, round((p / 100) * (len(values) - 1)))
    return values[idx]


if __name__ == "__main__":
    main()

