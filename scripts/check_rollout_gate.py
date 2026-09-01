"""Check a real CAR rollout JSONL before direct GRPO training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.trajectory_schema import read_jsonl
from src.eval.rollout_gate import RolloutGateThresholds, evaluate_rollout_gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="reports/direct_rl_gate.json")
    parser.add_argument("--config", default="configs/train/direct_rl_gate.yaml")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    thresholds = RolloutGateThresholds(
        min_trajectories=int(config["min_trajectories"]),
        min_tool_call_parse_rate=float(config["min_tool_call_parse_rate"]),
        min_executable_tool_rate=float(config["min_executable_tool_rate"]),
        min_mixed_reward_group_ratio=float(config["min_mixed_reward_group_ratio"]),
        min_consistent_initial_user_group_ratio=float(
            config["min_consistent_initial_user_group_ratio"]
        ),
        max_loop_or_max_turn_rate=float(config["max_loop_or_max_turn_rate"]),
        min_successful_trajectories=int(config["min_successful_trajectories"]),
        expected_group_size=int(config["group_size"]),
    )
    result = evaluate_rollout_gate(read_jsonl(args.input), thresholds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"DIRECT_RL_GATE={'PASS' if result.passed else 'FAIL'} output={output}")
    if not result.passed:
        for reason in result.reasons:
            print(f"- {reason}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
