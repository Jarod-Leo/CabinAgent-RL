"""Metric calculation for normalized tool-call trajectories."""

from __future__ import annotations

from typing import Any

from src.data.trajectory_schema import BenchmarkTask, JsonDict, ModelResponse, normalize_tool_call
from src.eval.failure_taxonomy import failure


def score_task(task: BenchmarkTask, response: ModelResponse) -> tuple[JsonDict, list[JsonDict], list[JsonDict]]:
    """Score one task and return metrics, failures, and executable tool results."""

    predicted = [normalize_tool_call(call) for call in response.tool_calls]
    expected = [normalize_tool_call(call) for call in task.expected_tool_calls]
    tool_results = [execute_tool_call(call, task.tools) for call in predicted]
    failures = classify_failures(task, response, expected, predicted, tool_results)

    exact_tool_match = _tool_call_lists_equal(expected, predicted)
    no_expected = len(expected) == 0
    success = exact_tool_match if not no_expected else len(predicted) == 0 and bool(response.content.strip())

    tool_name_accuracy = _tool_name_accuracy(expected, predicted)
    argument_accuracy = _argument_accuracy(expected, predicted)
    executable_rate = _executable_tool_rate(tool_results)
    hallucination = 1.0 if any(f["code"] == "F5_CAPABILITY_HALLUCINATION" for f in failures) else 0.0
    state_consistency = 0.0 if any(f["code"] == "F3_STATE_TRACKING_ERROR" for f in failures) else 1.0
    disambiguation_success = (
        0.0 if any(f["code"] == "F4_MISSING_CLARIFICATION" for f in failures) else 1.0
    )

    metrics = {
        "success": float(success),
        "tool_name_accuracy": tool_name_accuracy,
        "argument_accuracy": argument_accuracy,
        "tool_accuracy": 1.0 if exact_tool_match else 0.0,
        "executable_tool_rate": executable_rate,
        "hallucination_rate": hallucination,
        "state_consistency": state_consistency,
        "disambiguation_success": disambiguation_success,
        "avg_turns": _count_user_turns(task.messages),
        "num_expected_tool_calls": len(expected),
        "num_predicted_tool_calls": len(predicted),
    }
    return metrics, failures, tool_results


def execute_tool_call(call: JsonDict, tools: list[JsonDict]) -> JsonDict:
    """Validate a tool call against a minimal JSON-schema-like declaration."""

    tool_map = {tool.get("name"): tool for tool in tools}
    name = call.get("name", "")
    if name not in tool_map:
        return {"name": name, "status": "error", "error": "unknown_tool"}

    arguments = call.get("arguments", {})
    if not isinstance(arguments, dict):
        return {"name": name, "status": "error", "error": "arguments_not_object"}

    parameters = tool_map[name].get("parameters", {})
    required = parameters.get("required", [])
    missing = [arg for arg in required if arg not in arguments]
    if missing:
        return {"name": name, "status": "error", "error": "missing_required_arguments", "missing": missing}

    return {"name": name, "status": "ok", "result": {"simulated": True}}


def classify_failures(
    task: BenchmarkTask,
    response: ModelResponse,
    expected: list[JsonDict],
    predicted: list[JsonDict],
    tool_results: list[JsonDict],
) -> list[JsonDict]:
    failures: list[JsonDict] = []
    expected_names = [call.get("name") for call in expected]
    predicted_names = [call.get("name") for call in predicted]
    available_names = {tool.get("name") for tool in task.tools}

    if expected_names != predicted_names and expected:
        failures.append(
            failure(
                "F1_TOOL_NAME_ERROR",
                f"Expected tool sequence {expected_names}, got {predicted_names}.",
            )
        )

    if expected and predicted_names == expected_names and not _tool_call_lists_equal(expected, predicted):
        failures.append(failure("F2_ARGUMENT_ERROR", "Tool names matched but arguments differed."))

    for result in tool_results:
        if result.get("status") != "ok":
            failures.append(
                failure(
                    "F2_ARGUMENT_ERROR",
                    f"Tool call {result.get('name')} failed validation: {result.get('error')}.",
                )
            )

    unknown_tools = [name for name in predicted_names if name not in available_names]
    if unknown_tools:
        failures.append(
            failure("F5_CAPABILITY_HALLUCINATION", f"Predicted unavailable tools: {unknown_tools}.")
        )

    expected_behavior = task.metadata.get("expected_behavior")
    if expected_behavior == "clarify" and predicted:
        failures.append(
            failure("F4_MISSING_CLARIFICATION", "Task expected clarification, but a tool was called.")
        )

    if not expected and predicted:
        failures.append(
            failure("F5_CAPABILITY_HALLUCINATION", "No tool call was expected, but model called a tool.")
        )

    if _has_repeated_tool_loop(predicted):
        failures.append(failure("F8_VERBOSE_OR_LOOP", "Repeated the same tool call arguments."))

    if len(response.content) > 800:
        failures.append(failure("F8_VERBOSE_OR_LOOP", "Assistant response exceeded 800 characters."))

    if task.metadata.get("requires_state_tracking") and task.metadata.get("state_key"):
        expected_value = task.metadata.get("state_value")
        if expected_value and expected_value not in str(predicted):
            failures.append(
                failure("F3_STATE_TRACKING_ERROR", f"Expected state value '{expected_value}' was not preserved.")
            )

    if task.metadata.get("safety_boundary") and predicted:
        failures.append(failure("F6_SAFETY_BOUNDARY_ERROR", "Safety-boundary task produced a tool call."))

    if expected_names and predicted_names and expected_names != predicted_names and set(expected_names) == set(predicted_names):
        failures.append(failure("F7_PLANNING_ORDER_ERROR", "Expected tools are present but in the wrong order."))

    return _deduplicate_failures(failures)


def _tool_call_lists_equal(left: list[JsonDict], right: list[JsonDict]) -> bool:
    return left == right


def _tool_name_accuracy(expected: list[JsonDict], predicted: list[JsonDict]) -> float:
    if not expected:
        return 1.0 if not predicted else 0.0
    matches = sum(
        1
        for expected_call, predicted_call in zip(expected, predicted)
        if expected_call.get("name") == predicted_call.get("name")
    )
    return matches / len(expected)


def _argument_accuracy(expected: list[JsonDict], predicted: list[JsonDict]) -> float:
    if not expected:
        return 1.0 if not predicted else 0.0
    matches = sum(
        1
        for expected_call, predicted_call in zip(expected, predicted)
        if expected_call.get("arguments") == predicted_call.get("arguments")
    )
    return matches / len(expected)


def _executable_tool_rate(tool_results: list[JsonDict]) -> float:
    if not tool_results:
        return 1.0
    ok = sum(1 for result in tool_results if result.get("status") == "ok")
    return ok / len(tool_results)


def _count_user_turns(messages: list[JsonDict]) -> int:
    return sum(1 for message in messages if message.get("role") == "user")


def _has_repeated_tool_loop(predicted: list[JsonDict]) -> bool:
    seen: set[str] = set()
    for call in predicted:
        key = repr(call)
        if key in seen:
            return True
        seen.add(key)
    return False


def _deduplicate_failures(failures: list[JsonDict]) -> list[JsonDict]:
    seen: set[tuple[str, str]] = set()
    unique: list[JsonDict] = []
    for item in failures:
        key = (str(item.get("code")), str(item.get("detail")))
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique

