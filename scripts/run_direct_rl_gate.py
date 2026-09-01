"""Collect grouped CAR trajectories from two external vLLM servers."""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
import threading
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.trajectory_schema import write_jsonl
from src.integrations.car_bench_runtime import CarBenchSession, normalize_openai_message, stable_seed
_CLIENTS = threading.local()


def _client(base_url: str) -> Any:
    from openai import OpenAI

    key = base_url.rstrip("/")
    clients = getattr(_CLIENTS, "values", None)
    if clients is None:
        clients = {}
        _CLIENTS.values = clients
    if key not in clients:
        clients[key] = OpenAI(base_url=key + "/", api_key="local-vllm", timeout=180.0)
    return clients[key]


def _rollout(args: argparse.Namespace, task_id: str, trial: int) -> dict[str, Any]:
    session = CarBenchSession(
        dataset_root=args.dataset_root,
        task_id=task_id,
        split="train",
        simulator_base_url=args.simulator_base_url,
        simulator_model=args.simulator_model,
        simulator_initial_temperature=args.simulator_initial_temperature,
        simulator_temperature=args.simulator_temperature,
    )
    assistant_messages: list[dict[str, Any]] = []
    termination = "max_turns"
    try:
        session.start()
        for _turn in range(args.max_turns):
            response = _client(args.policy_base_url).chat.completions.create(
                model=args.policy_model,
                messages=session.messages,
                tools=session.tools_info,
                tool_choice="auto",
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                seed=stable_seed(f"policy:{args.seed}:{task_id}:{trial}"),
            )
            message = normalize_openai_message(response.choices[0].message)
            malformed = "<tool_call>" in message.get("content", "") and not message.get("tool_calls")
            session.record_parse(not malformed)
            assistant_messages.append(message)
            step = session.step(message)
            if step.done:
                termination = "environment_done"
                break
        trajectory = session.trajectory(
            group_id=task_id,
            trial=trial,
            termination_reason=termination,
            assistant_messages=assistant_messages,
            reward_mode="outcome",
        )
        trajectory["metadata"]["first_user_message"] = session.messages[1]["content"]
        trajectory["metadata"]["sampling"] = {
            "policy_temperature": args.temperature,
            "policy_top_p": args.top_p,
            "policy_seed": stable_seed(f"policy:{args.seed}:{task_id}:{trial}"),
            "simulator_initial_temperature": args.simulator_initial_temperature,
            "simulator_followup_temperature": args.simulator_temperature,
        }
        trajectory["metadata"]["policy_adapter"] = args.policy_adapter or None
        return trajectory
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/direct_rl_gate.yaml")
    parser.add_argument("--gate-parquet", default="data/processed/carbench/gate.parquet")
    parser.add_argument("--dataset-root", default="data/official/car-bench-dataset")
    parser.add_argument("--policy-base-url", required=True)
    parser.add_argument("--simulator-base-url", required=True)
    parser.add_argument("--policy-model", default="cabinagent-policy")
    parser.add_argument("--policy-adapter", default="")
    parser.add_argument("--simulator-model", default="cabinagent-user-simulator")
    parser.add_argument("--output", default="experiments/G00/trajectories.jsonl")
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--task-count", type=int)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--simulator-initial-temperature", type=float)
    parser.add_argument("--simulator-temperature", type=float)
    args = parser.parse_args()

    with (ROOT / args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    policy_sampling = config["policy_sampling"]
    simulator_sampling = config["simulator_sampling"]
    args.group_size = args.group_size if args.group_size is not None else int(config["group_size"])
    args.task_count = args.task_count if args.task_count is not None else int(config["task_count"])
    args.max_turns = args.max_turns if args.max_turns is not None else int(policy_sampling["max_turns"])
    args.max_tokens = args.max_tokens if args.max_tokens is not None else int(policy_sampling["max_tokens"])
    args.temperature = args.temperature if args.temperature is not None else float(policy_sampling["temperature"])
    args.top_p = args.top_p if args.top_p is not None else float(policy_sampling["top_p"])
    args.seed = args.seed if args.seed is not None else int(policy_sampling["seed"])
    args.concurrency = args.concurrency if args.concurrency is not None else int(policy_sampling["concurrency"])
    args.simulator_initial_temperature = (
        args.simulator_initial_temperature
        if args.simulator_initial_temperature is not None
        else float(simulator_sampling["initial_temperature"])
    )
    args.simulator_temperature = (
        args.simulator_temperature
        if args.simulator_temperature is not None
        else float(simulator_sampling["followup_temperature"])
    )

    from datasets import load_dataset

    gate_path = (ROOT / args.gate_parquet).resolve()
    args.dataset_root = str((ROOT / args.dataset_root).resolve())
    rows = load_dataset("parquet", data_files=str(gate_path), split="train")
    task_ids = [str(row["extra_info"]["task_id"]) for row in rows.select(range(min(args.task_count, len(rows))))]
    work = [(task_id, trial) for task_id in task_ids for trial in range(args.group_size)]
    trajectories: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(_rollout, args, task_id, trial) for task_id, trial in work]
        for future in concurrent.futures.as_completed(futures):
            trajectories.append(future.result())
    trajectories.sort(key=lambda row: (row["metadata"]["task_id"], row["metadata"]["trial"]))
    output = (ROOT / args.output).resolve()
    write_jsonl(output, trajectories)
    print(f"GATE_ROLLOUTS_OK trajectories={len(trajectories)} output={output}")


if __name__ == "__main__":
    main()
