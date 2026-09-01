from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from scripts.build_carbench_parquet import build_splits, select_gate_rows
from src.integrations.car_bench_runtime import (
    infer_task_family,
    initial_user_cache_key,
    is_initial_user_turn,
    require_initial_continue,
    stable_seed,
)


class CarBenchDataTests(unittest.TestCase):
    def test_initial_simulator_turn_contract(self) -> None:
        messages = [{"role": "system"}, {"role": "user"}]
        self.assertTrue(is_initial_user_turn(messages))
        require_initial_continue("CONTINUE")
        with self.assertRaises(ValueError):
            require_initial_continue("STOP")

    def test_initial_user_cache_key_covers_model_and_full_prompt(self) -> None:
        messages = [{"role": "system", "content": "task"}, {"role": "user", "content": "hi"}]
        key = initial_user_cache_key("simulator", messages)
        self.assertEqual(key, initial_user_cache_key("simulator", list(messages)))
        self.assertNotEqual(key, initial_user_cache_key("other-model", messages))
        self.assertNotEqual(
            key,
            initial_user_cache_key(
                "simulator",
                [{"role": "system", "content": "other"}, {"role": "user", "content": "hi"}],
            ),
        )

    def _rows(self, family: str) -> list[dict[str, str]]:
        counts = {"base": 50, "hallucination": 48, "disambiguation": 31}
        return [
            {
                "task_id": f"{family}_{index}",
                "task_type": family,
                "persona": "hidden persona",
                "instruction": "hidden instruction",
                "actions": "[]",
                "context_init_config": "{}",
            }
            for index in range(counts[family])
        ]

    def test_stratified_split_and_gate_exclude_hidden_fields(self) -> None:
        with patch(
            "scripts.build_carbench_parquet.read_task_rows",
            side_effect=lambda _root, family, _split: self._rows(family),
        ):
            train, dev = build_splits(None, seed=42)
            gate = select_gate_rows(train, count=20)
            self.assertEqual((len(train), len(dev), len(gate)), (103, 26, 20))
            for row in train + dev + gate:
                serialized = json.dumps(row)
                self.assertNotIn("hidden persona", serialized)
                self.assertNotIn("hidden instruction", serialized)
                self.assertEqual(row["agent_name"], "car_bench")

    def test_task_helpers_are_stable(self) -> None:
        self.assertEqual(infer_task_family("hallucination_12"), "hallucination")
        self.assertEqual(stable_seed("base_0"), stable_seed("base_0"))


if __name__ == "__main__":
    unittest.main()
