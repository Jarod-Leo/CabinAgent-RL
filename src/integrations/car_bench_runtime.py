"""Runtime bridge for the official CAR-bench environment.

This module intentionally imports CAR-bench lazily. Local CPU tests can inspect
the data helpers without installing the GPU/benchmark environment.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.rewards.prm_lite import score_trajectory


TASK_FAMILIES = ("base", "hallucination", "disambiguation")
_DATA_MANAGER_LOCK = threading.Lock()
_INITIAL_USER_CACHE_GUARD = threading.Lock()
_INITIAL_USER_RESPONSES: dict[str, str] = {}
_INITIAL_USER_LOCKS: dict[str, threading.Lock] = {}


def _json_value(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


def read_task_rows(dataset_root: str | Path, family: str, split: str) -> list[dict[str, Any]]:
    if family not in TASK_FAMILIES:
        raise ValueError(f"Unsupported CAR task family: {family}")
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported CAR split: {split}")

    path = Path(dataset_root) / "tasks" / f"{family}_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def infer_task_family(task_id: str) -> str:
    family = task_id.split("_", 1)[0]
    if family not in TASK_FAMILIES:
        raise ValueError(f"Cannot infer CAR task family from task_id={task_id!r}")
    return family


def load_task_row(dataset_root: str | Path, task_id: str, split: str) -> dict[str, Any]:
    family = infer_task_family(task_id)
    matches = [row for row in read_task_rows(dataset_root, family, split) if row["task_id"] == task_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one {split} row for {task_id}, found {len(matches)}")
    return matches[0]


def task_row_to_official(row: dict[str, Any]) -> Any:
    """Convert a downloaded JSONL row to the official ``car_bench.types.Task``."""

    from car_bench.types import Action, Task

    value = dict(row)
    value["actions"] = [Action(**item) for item in _json_value(value.get("actions"), [])]
    value["context_init_config"] = _json_value(value.get("context_init_config"), {})
    value["removed_part"] = _json_value(value.get("removed_part"), None)
    return Task(**value)


def stable_seed(task_id: str) -> int:
    digest = hashlib.sha256(task_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def is_initial_user_turn(messages: list[dict[str, Any]]) -> bool:
    return len(messages) == 2


def require_initial_continue(keyword: Any) -> None:
    value = getattr(keyword, "value", keyword)
    if str(value).strip().upper() != "CONTINUE":
        raise ValueError(
            "The initial simulator turn must use conversation_control_keyword=CONTINUE"
        )


def initial_user_cache_key(model: str, messages: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _initial_user_lock(cache_key: str) -> threading.Lock:
    with _INITIAL_USER_CACHE_GUARD:
        return _INITIAL_USER_LOCKS.setdefault(cache_key, threading.Lock())


class VLLMUserSimulator:
    """Official CAR user prompt/state machine backed by a local vLLM endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        task_id: str,
        timeout_seconds: float = 120.0,
        max_tokens: int = 512,
        initial_temperature: float = 0.0,
        temperature: float = 0.2,
    ) -> None:
        from car_bench.envs.user.user import LLMUserSimulationEnv
        from openai import OpenAI

        class _Simulator(LLMUserSimulationEnv):
            def generate_next_message(inner_self, messages: list[dict[str, Any]]) -> str:
                return self._generate(inner_self, messages)

        self.client = OpenAI(base_url=base_url.rstrip("/") + "/", api_key="local-vllm", timeout=timeout_seconds)
        self.model = model
        self.max_tokens = max_tokens
        self.initial_temperature = float(initial_temperature)
        self.temperature = float(temperature)
        self.seed = stable_seed(task_id)
        self.delegate = _Simulator(model=model, provider="openai", user_thinking=False)

    def _generate(self, delegate: Any, messages: list[dict[str, Any]]) -> str:
        from car_bench.envs.user.user_end_conversation import (
            check_end_conversation,
            end_conversation_failure,
        )

        initial_turn = is_initial_user_turn(messages)
        if not initial_turn:
            try:
                if end_conversation_failure.get():
                    return "###STOP###"
            except LookupError:
                pass

        cache_key = initial_user_cache_key(self.model, messages) if initial_turn else ""
        cache_lock = _initial_user_lock(cache_key) if initial_turn else nullcontext()
        with cache_lock:
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    cached_content = _INITIAL_USER_RESPONSES.get(cache_key) if initial_turn else None
                    if cached_content is not None:
                        content = cached_content
                    else:
                        request_messages = list(messages)
                        if initial_turn and attempt > 0:
                            request_messages.append(
                                {
                                    "role": "system",
                                    "content": (
                                        "This is the first user turn, before the assistant has acted. "
                                        "Set conversation_control_keyword exactly to CONTINUE and "
                                        "provide the task's initial user request. Do not stop or grade "
                                        "the conversation."
                                    ),
                                }
                            )
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=request_messages,
                            max_tokens=self.max_tokens,
                            temperature=(
                                self.initial_temperature if initial_turn else self.temperature
                            ),
                            seed=stable_seed(f"{self.seed}:{attempt}"),
                            response_format={
                                "type": "json_schema",
                                "json_schema": {
                                    "name": delegate.response_format.__name__,
                                    "schema": delegate.response_format.model_json_schema(),
                                    "strict": True,
                                },
                            },
                        )
                        content = response.choices[0].message.content or "{}"

                    parsed = delegate.response_format.model_validate_json(content)
                    if initial_turn:
                        require_initial_continue(parsed.conversation_control_keyword)
                        _INITIAL_USER_RESPONSES[cache_key] = content
                    delegate.messages.append({"role": "assistant", "content": content})
                    return check_end_conversation(
                        parsed.conversation_control_keyword,
                        parsed.user_message,
                    )
                except Exception as exc:  # network and structured-output failures share a bounded retry
                    last_error = exc
                    if attempt < 2:
                        time.sleep(1.0 + attempt)
        raise RuntimeError(f"User simulator failed after 3 attempts: {last_error}") from last_error

    def reset(self, **kwargs: Any) -> str:
        return self.delegate.reset(**kwargs)

    def step(self, content: str) -> str:
        return self.delegate.step(content)

    def get_total_cost(self) -> float:
        return 0.0


class AutomaticOnlyPolicyEvaluator:
    """Run CAR's deterministic policy checks while disabling LLM judging."""

    def __init__(self) -> None:
        from car_bench.envs.policy_evaluator import LLMPolicyEvaluatorEnv

        class _Evaluator(LLMPolicyEvaluatorEnv):
            def __init__(inner_self) -> None:
                inner_self.total_cost = 0.0

            def evaluate_llm(inner_self, policy: str, trajectory: str) -> str:
                del policy, trajectory
                return json.dumps({"reasoning": "LLM policy judging disabled", "policy_followed": True})

        self.delegate = _Evaluator()

    def evaluate_llm(self, policy: str, trajectory: str) -> str:
        return self.delegate.evaluate_llm(policy, trajectory)

    def evaluate_aut(self, trajectory: list[dict[str, Any]]) -> str:
        return self.delegate.evaluate_aut(trajectory)

    def get_total_cost(self) -> float:
        return 0.0


def _ensure_data_manager(dataset_root: Path) -> None:
    from car_bench.envs.car_voice_assistant.mock_data import car_va_data_manager
    from car_bench.envs.car_voice_assistant.mock_data.data_manager import DataManager

    expected = str((dataset_root / "mock_data").resolve())
    with _DATA_MANAGER_LOCK:
        current = getattr(car_va_data_manager, "_instance", None)
        current_path = getattr(current, "nav_folder_path", "") if current is not None else ""
        if current is None or not str(current_path).startswith(expected):
            car_va_data_manager._instance = DataManager(expected, preload=False)


def _initialize_context(env: Any, task: Any) -> dict[str, Any]:
    from car_bench.envs.car_voice_assistant.context.dynamic_context_state import ContextState, context_state
    from car_bench.envs.car_voice_assistant.context.fixed_context import FixedContext, fixed_context
    from car_bench.envs.car_voice_assistant.tasks.task_config import TaskConfig, task_config
    from car_bench.envs.policy_evaluator import policy_errors_during_runtime
    from car_bench.envs.tool_execution_error_evaluator import tool_execution_errors_during_runtime
    from car_bench.envs.user.user_end_conversation import end_conversation_failure

    tokens = {
        "task": (task_config, task_config.set(TaskConfig())),
        "state": (context_state, context_state.set(ContextState())),
        "fixed": (fixed_context, fixed_context.set(FixedContext())),
        "policy": (policy_errors_during_runtime, policy_errors_during_runtime.set([])),
        "tool": (tool_execution_errors_during_runtime, tool_execution_errors_during_runtime.set([])),
        "end": (end_conversation_failure, end_conversation_failure.set([])),
    }
    task_config.get().update_state(calendar_id=task.calendar_id)
    fixed_context.get().update_state(**task.context_init_config)
    context_state.get().update_state(**task.context_init_config)
    env.wiki = env.wiki.replace(
        "{{placeholder_location_based_on_task_context_init_config}}",
        fixed_context.get().current_location.model_dump_json(),
    )
    env.wiki = env.wiki.replace(
        "{{placeholder_datetime_based_on_task_context_init_config}}",
        fixed_context.get().current_datetime.model_dump_json(),
    )
    return tokens


def _reset_context(tokens: dict[str, Any]) -> None:
    for key in ("end", "tool", "policy", "fixed", "state", "task"):
        variable, token = tokens[key]
        variable.reset(token)


def _tool_call_message(name: str, arguments: Any, call_id: str | None = None) -> dict[str, Any]:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return {
        "id": call_id or f"call_{uuid.uuid4().hex}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def actions_from_assistant(message: dict[str, Any]) -> list[Any]:
    from car_bench.types import Action, RESPOND_ACTION_NAME, USER_AS_A_TOOL_ACTION_NAMES

    raw_calls = message.get("tool_calls") or []
    if not raw_calls:
        return [Action(name=RESPOND_ACTION_NAME, kwargs={"content": message.get("content") or ""})]

    actions = []
    for raw in raw_calls:
        function = raw.get("function") or {}
        name = str(function.get("name") or "")
        arguments = _json_value(function.get("arguments"), {})
        if name in USER_AS_A_TOOL_ACTION_NAMES:
            content = arguments.get("message_to_user", arguments.get("content", ""))
            return [Action(name=name, kwargs={"content": content})]
        actions.append(Action(name=name, kwargs=arguments if isinstance(arguments, dict) else {}))
    return actions


def _removed_result(observation: str, removed_part: list[str] | None, tool_name: str) -> str:
    if not removed_part:
        return observation
    matching = [item for item in removed_part if item.startswith(f"result.{tool_name}.")]
    if not matching:
        return observation
    from car_bench.envs.tool_manipulation import remove_result_element

    try:
        return json.dumps(remove_result_element(json.loads(observation), matching), ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return observation


@dataclass
class StepResult:
    done: bool
    observations: list[str]
    added_messages: list[dict[str, Any]]
    reward: float
    info: dict[str, Any]
    executable_calls: int
    attempted_calls: int


class CarBenchSession:
    """One isolated CAR trajectory with explicit context-var lifecycle."""

    def __init__(
        self,
        dataset_root: str | Path,
        task_id: str,
        split: str,
        simulator_base_url: str,
        simulator_model: str,
        simulator_initial_temperature: float = 0.0,
        simulator_temperature: float = 0.2,
    ) -> None:
        from car_bench.envs.base import Env
        from car_bench.envs.car_voice_assistant.mock_data import load_data
        from car_bench.envs.car_voice_assistant.tools import ALL_TOOLS
        from car_bench.envs.car_voice_assistant.wiki import WIKI

        self.dataset_root = Path(dataset_root).resolve()
        _ensure_data_manager(self.dataset_root)
        self.task = task_row_to_official(load_task_row(self.dataset_root, task_id, split))
        self.env = Env(
            data_load_func=load_data,
            tools=ALL_TOOLS,
            tasks=[self.task],
            wiki=str(WIKI),
            user_strategy="human",
            user_model="unused",
            policy_evaluator_strategy="human",
            policy_evaluator_model="unused",
            evaluate_policy=True,
            score_tool_execution_errors=True,
            score_policy_errors=True,
        )
        self.env.terminate_tools = ["call_phone_by_number"]
        self.env.user = VLLMUserSimulator(
            simulator_base_url,
            simulator_model,
            task_id,
            initial_temperature=simulator_initial_temperature,
            temperature=simulator_temperature,
        )
        self.env.policy_evaluator = AutomaticOnlyPolicyEvaluator()
        self.tokens = _initialize_context(self.env, self.task)
        self.messages: list[dict[str, Any]] = []
        self.tools_info: list[dict[str, Any]] = []
        self.last_reward = 0.0
        self.last_info: dict[str, Any] = {}
        self.attempted_calls = 0
        self.executable_calls = 0
        self.tool_call_parse_values: list[float] = []
        self.closed = False

    def start(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        from car_bench.envs.tool_manipulation import remove_tool_elements

        reset = self.env.reset(task_index=0)
        removals = ["planning_tool", "think", *(self.task.removed_part or [])]
        self.tools_info = remove_tool_elements(self.env.tools_info, self.env.tools_info, removals)
        self.messages = [
            {"role": "system", "content": self.env.wiki or ""},
            {"role": "user", "content": reset.observation},
        ]
        self.last_info = reset.info.model_dump()
        return list(self.messages), list(self.tools_info)

    def record_parse(self, valid: bool) -> None:
        self.tool_call_parse_values.append(1.0 if valid else 0.0)

    def step(self, assistant_message: dict[str, Any]) -> StepResult:
        from car_bench.envs.tool_manipulation import check_hallucinated_removed_part
        from car_bench.types import USER_AS_A_TOOL_ACTION_NAMES

        actions = actions_from_assistant(assistant_message)
        if self.task.removed_part and assistant_message.get("tool_calls"):
            check_hallucinated_removed_part(
                self.task.removed_part,
                assistant_message["tool_calls"],
                self.task.task_type,
            )

        self.messages.append(assistant_message)
        response = self.env.run_steps(actions, self.messages)
        observations = [str(item) for item in response.observation]
        added: list[dict[str, Any]] = []
        is_user_turn = actions[0].name in USER_AS_A_TOOL_ACTION_NAMES

        attempted = 0 if is_user_turn else len(actions)
        executable = 0
        if not is_user_turn:
            for index, action in enumerate(actions):
                observation = observations[index]
                if not observation.startswith(("Error:", "Unknown action")):
                    executable += 1
                if not response.done:
                    call = (assistant_message.get("tool_calls") or [])[index]
                    content = _removed_result(observation, self.task.removed_part, action.name)
                    added.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": action.name,
                            "content": content,
                        }
                    )
        elif not response.done:
            added.append({"role": "user", "content": observations[0]})

        self.messages.extend(added)
        self.attempted_calls += attempted
        self.executable_calls += executable
        self.last_reward = float(response.reward)
        self.last_info = response.info.model_dump()
        return StepResult(
            done=bool(response.done),
            observations=observations,
            added_messages=added,
            reward=self.last_reward,
            info=self.last_info,
            executable_calls=executable,
            attempted_calls=attempted,
        )

    def trajectory(
        self,
        *,
        group_id: str,
        trial: int,
        termination_reason: str,
        assistant_messages: Iterable[dict[str, Any]],
        reward_mode: str = "outcome",
        process_reward_weight: float = 0.3,
    ) -> dict[str, Any]:
        reward_info = (self.last_info.get("reward_info") or {}).get("info") or {}
        aut_errors = reward_info.get("policy_aut_errors") or []
        tool_errors = reward_info.get("tool_execution_errors") or []
        end_keyword = str(reward_info.get("end_conversation_keyword") or "")
        success = float(abs(self.last_reward - 1.0) <= 1e-6)
        executable_rate = self.executable_calls / self.attempted_calls if self.attempted_calls else 1.0
        parse_rate = (
            sum(self.tool_call_parse_values) / len(self.tool_call_parse_values)
            if self.tool_call_parse_values
            else 1.0
        )
        failures: list[dict[str, str]] = []
        if parse_rate < 1.0:
            failures.append({"code": "F1_TOOL_NAME_ERROR", "detail": "Malformed tool-call markup."})
        if tool_errors or executable_rate < 1.0:
            failures.append({"code": "F2_ARGUMENT_ERROR", "detail": "; ".join(tool_errors) or "Tool error."})
        if aut_errors:
            failures.append({"code": "F6_SAFETY_BOUNDARY_ERROR", "detail": "; ".join(aut_errors)})
        if "HALLUCINATION" in end_keyword:
            failures.append({"code": "F5_CAPABILITY_HALLUCINATION", "detail": end_keyword})
        if "DISAMBIGUATION" in end_keyword:
            failures.append({"code": "F4_MISSING_CLARIFICATION", "detail": end_keyword})
        if termination_reason == "max_turns":
            failures.append({"code": "F8_VERBOSE_OR_LOOP", "detail": "Reached the CAR turn limit."})

        assistant_list = list(assistant_messages)
        predicted = []
        for message in assistant_list:
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                predicted.append(
                    {
                        "name": function.get("name", ""),
                        "arguments": _json_value(function.get("arguments"), {}),
                    }
                )
        normalized = {
            "id": f"{self.task.task_id}-trial-{trial}",
            "benchmark": "carbench",
            "split": "train",
            "messages": list(self.messages),
            "tools": [],
            "expected_tool_calls": [],
            "model_response": assistant_list[-1] if assistant_list else {},
            "predicted_tool_calls": predicted,
            "tool_results": [message for message in self.messages if message.get("role") == "tool"],
            "metrics": {
                "success": success,
                "reward": self.last_reward,
                "tool_call_parse_rate": round(parse_rate, 6),
                "executable_tool_rate": round(executable_rate, 6),
                "state_consistency": float(reward_info.get("r_actions_final") or success),
            },
            "failures": failures,
            "metadata": {
                "task_id": self.task.task_id,
                "task_type": self.task.task_type.value,
                "group_id": group_id,
                "trial": trial,
                "termination_reason": termination_reason,
                "tool_call_count": len(predicted),
                "policy_violation": bool(aut_errors),
            },
        }
        breakdown = score_trajectory(normalized, process_reward_weight=process_reward_weight)
        selected = breakdown.outcome if reward_mode == "outcome" else breakdown.total
        normalized["reward"] = {**breakdown.to_dict(), "selected": selected}
        return normalized

    def close(self) -> None:
        if not self.closed:
            _reset_context(self.tokens)
            self.closed = True

    def __enter__(self) -> "CarBenchSession":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()


def normalize_openai_message(message: Any) -> dict[str, Any]:
    raw = message.model_dump(exclude_none=True) if hasattr(message, "model_dump") else dict(message)
    calls = []
    for call in raw.get("tool_calls") or []:
        function = call.get("function") or {}
        calls.append(_tool_call_message(function.get("name", ""), function.get("arguments", "{}"), call.get("id")))
    result = {"role": "assistant", "content": raw.get("content") or ""}
    if calls:
        result["tool_calls"] = calls
    return result
