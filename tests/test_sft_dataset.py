from __future__ import annotations

import unittest
from collections import UserDict
import copy

from src.data.sft_dataset import build_sft_records, normalize_sft_messages, split_train_val
from scripts.check_sft_fallback_data import validate_tool_call_round_trip
from scripts.build_corrective_sft_data import build_corrective_records
from scripts.train_sft_fallback import encode_sft_examples


def trajectory(task_id: str, content: str, success: float = 1.0) -> dict:
    return {
        "id": f"{task_id}-trial",
        "benchmark": "carbench",
        "messages": [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "request"},
            {"role": "assistant", "content": content},
        ],
        "model_response": {"role": "assistant", "content": content},
        "tools": [{"type": "function", "function": {"name": "tool"}}],
        "metrics": {"success": success},
        "metadata": {"task_id": task_id},
    }


class SftDatasetTests(unittest.TestCase):
    def test_corrective_records_refuse_without_failed_tool_calls(self) -> None:
        row = trajectory("hallucination-1", "failed", success=0.0)
        row["metadata"]["task_type"] = "hallucination_missing_tool_parameter"
        row["messages"] = [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "Set the unavailable parameter."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "bad", "arguments": {}}}],
            },
            {"role": "tool", "content": "missing required positional argument"},
        ]
        records = build_corrective_records([row])
        self.assertEqual(len(records), 2)
        self.assertTrue(all(record["metadata"]["task_id"] == "hallucination-1" for record in records))
        for record in records:
            self.assertTrue(all(not message.get("tool_calls") for message in record["messages"]))
            self.assertNotIn("missing required positional argument", str(record["messages"]))

    def test_corrective_records_ignore_non_target_tasks(self) -> None:
        self.assertEqual(build_corrective_records([trajectory("base-1", "failed", 0.0)]), [])

    def test_only_assistant_target_tokens_are_labeled(self) -> None:
        class FakeTokenizer:
            def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
                del value, add_special_tokens
                return [1, 2, 3]

            def convert_tokens_to_ids(self, value: str) -> int:
                del value
                return 6

            def apply_chat_template(
                self, messages: list[dict], *, tools: list | None, tokenize: bool, add_generation_prompt: bool
            ) -> UserDict:
                del messages, tools, tokenize, add_generation_prompt
                return UserDict({"input_ids": [7, 1, 2, 3, 4, 5, 6]})

        records = [{"messages": [{"role": "user"}, {"role": "assistant"}], "tools": []}]
        examples, stats = encode_sft_examples(FakeTokenizer(), records, max_length=8)
        self.assertEqual(stats["examples"], 1)
        self.assertEqual(examples[0]["labels"], [-100, -100, -100, -100, 4, 5, 6])

    def test_successful_complete_trajectories_are_deduplicated_without_duplicate_response(self) -> None:
        row = trajectory("task-a", "done")
        records = build_sft_records([row, row, trajectory("task-b", "failed", 0.0)])
        self.assertEqual(len(records), 1)
        self.assertEqual(len(records[0]["messages"]), 3)
        self.assertEqual(records[0]["tools"], row["tools"])

    def test_tool_call_arguments_are_objects_without_mutating_source(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "set_light", "arguments": '{"on": true}'},
                    }
                ],
            }
        ]
        original = copy.deepcopy(messages)
        normalized = normalize_sft_messages(messages)
        arguments = normalized[0]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(arguments, {"on": True})
        self.assertEqual(messages, original)

    def test_double_encoded_tool_arguments_are_rejected(self) -> None:
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "set_light", "arguments": '"{\\"on\\": true}"'}}
                ],
            }
        ]
        with self.assertRaisesRegex(ValueError, "must decode to an object"):
            normalize_sft_messages(messages)

    def test_qwen_template_tool_calls_round_trip_as_objects(self) -> None:
        class FakeRoundTripTokenizer:
            def apply_chat_template(
                self, messages: list[dict], *, tools: list | None, tokenize: bool, add_generation_prompt: bool
            ) -> str:
                del messages, tools, tokenize, add_generation_prompt
                return '<tool_call>\n{"name":"set_light","arguments":{"on":true}}\n</tool_call>'

        record = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "set_light", "arguments": {"on": True}},
                        }
                    ],
                }
            ],
            "tools": [],
        }
        self.assertEqual(validate_tool_call_round_trip(FakeRoundTripTokenizer(), record), 1)

    def test_train_val_split_keeps_tasks_disjoint(self) -> None:
        records = build_sft_records(
            [trajectory(f"task-{index}", f"answer-{index}") for index in range(5)]
        )
        train, val = split_train_val(records, val_ratio=0.4, seed=42)
        train_tasks = {row["metadata"]["task_id"] for row in train}
        val_tasks = {row["metadata"]["task_id"] for row in val}
        self.assertEqual(len(train) + len(val), 5)
        self.assertFalse(train_tasks & val_tasks)


if __name__ == "__main__":
    unittest.main()
