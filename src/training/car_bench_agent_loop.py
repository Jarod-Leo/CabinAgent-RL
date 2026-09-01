"""veRL v0.9 agent loop for online CAR-bench trajectories."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from typing import Any

from src.integrations.car_bench_runtime import CarBenchSession, _tool_call_message


def _assistant_message(tokenizer: Any, token_ids: list[int], calls: list[Any]) -> dict[str, Any]:
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    content = re.sub(r"<tool_call>.*?</tool_call>", "", decoded, flags=re.DOTALL).strip()
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [
            _tool_call_message(call.name, call.arguments, call.tool_call_id)
            for call in calls
        ]
    return message


def _parse_is_valid(tokenizer: Any, token_ids: list[int], calls: list[Any]) -> bool:
    decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    starts = decoded.count("<tool_call>")
    ends = decoded.count("</tool_call>")
    if starts == 0 and ends == 0:
        return True
    return starts == ends == len(calls) and starts > 0


def _task_id(extra_info: Any) -> str:
    if hasattr(extra_info, "item"):
        extra_info = extra_info.item()
    if not isinstance(extra_info, dict) or not extra_info.get("task_id"):
        raise ValueError("CAR agent loop requires extra_info.task_id")
    return str(extra_info["task_id"])


def _register_class() -> type:
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput
    from verl.experimental.agent_loop.tool_parser import ToolParser

    class CarBenchAgentLoop(AgentLoopBase):
        def __init__(
            self,
            *args: Any,
            dataset_root: str,
            simulator_base_url: str,
            simulator_model: str = "cabinagent-user-simulator",
            split: str = "train",
            max_turns: int = 20,
            tool_parser: str = "hermes",
            simulator_initial_temperature: float = 0.0,
            simulator_temperature: float = 0.2,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.dataset_root = dataset_root
            self.simulator_base_url = simulator_base_url
            self.simulator_model = simulator_model
            self.split = split
            self.max_turns = int(max_turns)
            self.simulator_initial_temperature = float(simulator_initial_temperature)
            self.simulator_temperature = float(simulator_temperature)
            self.response_length = int(self.rollout_config.response_length)
            self.tool_parser = ToolParser.get_tool_parser(tool_parser, self.tokenizer)

        async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> Any:
            task_id = _task_id(kwargs.get("extra_info"))
            group_id = str(kwargs.get("index", task_id))
            session = CarBenchSession(
                dataset_root=self.dataset_root,
                task_id=task_id,
                split=self.split,
                simulator_base_url=self.simulator_base_url,
                simulator_model=self.simulator_model,
                simulator_initial_temperature=self.simulator_initial_temperature,
                simulator_temperature=self.simulator_temperature,
            )
            assistant_messages: list[dict[str, Any]] = []
            response_mask: list[int] = []
            response_logprobs: list[float] | None = None
            all_ids: list[int] = []
            routed_experts = None
            generate_seconds = 0.0
            environment_seconds = 0.0
            num_preempted = 0
            termination_reason = "max_turns"
            extra_fields: dict[str, Any] = {}

            try:
                messages, tools = await asyncio.to_thread(session.start)
                prompt_ids = await self.apply_chat_template(messages, tools=tools)
                all_ids = list(prompt_ids)
                request_id = uuid.uuid4().hex

                params = dict(sampling_params)
                if self.tool_parser.stop_token_ids:
                    params["stop_token_ids"] = list(
                        set((params.get("stop_token_ids") or []) + self.tool_parser.stop_token_ids)
                    )

                for _turn in range(self.max_turns):
                    started = time.perf_counter()
                    generated = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=all_ids,
                        sampling_params=params,
                    )
                    generate_seconds += time.perf_counter() - started
                    token_ids = list(generated.token_ids)
                    all_ids.extend(token_ids)
                    response_mask.extend([1] * len(token_ids))
                    num_preempted += int(generated.num_preempted or 0)
                    if generated.routed_experts is not None:
                        routed_experts = generated.routed_experts
                    if not extra_fields:
                        extra_fields.update(generated.extra_fields or {})

                    if generated.log_probs is not None:
                        if response_logprobs is None:
                            if len(response_mask) == len(token_ids):
                                response_logprobs = []
                            else:
                                response_logprobs = None
                        if response_logprobs is not None:
                            response_logprobs.extend(generated.log_probs)
                    elif response_logprobs is not None:
                        response_logprobs.extend([0.0] * len(token_ids))

                    _content, calls = await self.tool_parser.extract_tool_calls(token_ids)
                    session.record_parse(_parse_is_valid(self.tokenizer, token_ids, calls))
                    message = _assistant_message(self.tokenizer, token_ids, calls)
                    assistant_messages.append(message)

                    started = time.perf_counter()
                    step = await asyncio.to_thread(session.step, message)
                    environment_seconds += time.perf_counter() - started
                    if step.done:
                        termination_reason = "environment_done"
                        break
                    if len(response_mask) >= self.response_length:
                        termination_reason = "max_tokens"
                        break

                    if step.added_messages:
                        observation_ids = await self.apply_chat_template(
                            step.added_messages,
                            remove_system_prompt=True,
                        )
                        observation_ids = list(self.turn_separator) + list(observation_ids)
                        if len(response_mask) + len(observation_ids) >= self.response_length:
                            termination_reason = "max_tokens"
                            break
                        all_ids.extend(observation_ids)
                        response_mask.extend([0] * len(observation_ids))
                        if response_logprobs is not None:
                            response_logprobs.extend([0.0] * len(observation_ids))

                trajectory = session.trajectory(
                    group_id=group_id,
                    trial=0,
                    termination_reason=termination_reason,
                    assistant_messages=assistant_messages,
                    reward_mode=os.environ.get("CABIN_REWARD_MODE", "outcome"),
                    process_reward_weight=float(os.environ.get("CABIN_PROCESS_REWARD_WEIGHT", "0.3")),
                )
                extra_fields.update(
                    {
                        "trajectory": trajectory,
                        "turn_scores": [],
                        "tool_rewards": [],
                    }
                )
                response_ids = all_ids[len(prompt_ids) :]
                return AgentLoopOutput(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids[: self.response_length],
                    response_mask=response_mask[: self.response_length],
                    response_logprobs=(
                        response_logprobs[: self.response_length]
                        if response_logprobs is not None
                        else None
                    ),
                    routed_experts=(
                        routed_experts[: len(prompt_ids) + self.response_length]
                        if routed_experts is not None
                        else None
                    ),
                    reward_score=float(trajectory["reward"]["selected"]),
                    num_turns=len(session.messages),
                    metrics=AgentLoopMetrics(
                        generate_sequences=generate_seconds,
                        tool_calls=environment_seconds,
                        compute_score=0.0,
                        num_preempted=num_preempted,
                    ),
                    extra_fields=extra_fields,
                )
            finally:
                session.close()

    CarBenchAgentLoop.__name__ = "CarBenchAgentLoop"
    CarBenchAgentLoop.__qualname__ = "CarBenchAgentLoop"
    CarBenchAgentLoop.__module__ = __name__
    return CarBenchAgentLoop


# Hydra resolves this symbol from configs/agent_loop/carbench.yaml. Building the
# class lazily keeps local CPU utilities importable without veRL installed.
try:
    CarBenchAgentLoop = _register_class()
except ModuleNotFoundError as exc:
    if exc.name and (exc.name == "verl" or exc.name.startswith("verl.")):
        CarBenchAgentLoop = None  # type: ignore[assignment]
    else:
        raise
