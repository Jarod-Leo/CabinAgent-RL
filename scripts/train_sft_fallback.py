"""Train the guarded Qwen2.5-7B LoRA minimal-SFT fallback."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.trajectory_schema import read_jsonl


def _token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)


def encode_sft_examples(
    tokenizer: Any,
    records: list[dict[str, Any]],
    max_length: int,
) -> tuple[list[dict[str, list[int]]], dict[str, int]]:
    examples: list[dict[str, list[int]]] = []
    skipped_too_long = 0
    skipped_no_target = 0
    assistant_prefix = _token_ids(
        tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    )
    end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not assistant_prefix or end_token_id is None:
        raise RuntimeError("Qwen assistant boundary tokens are unavailable")
    for record in records:
        messages = list(record.get("messages", []))
        tools = record.get("tools") or None
        full_ids = _token_ids(
            tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=False,
            )
        )
        labels = [-100] * len(full_ids)
        index = 0
        while index <= len(full_ids) - len(assistant_prefix):
            if full_ids[index : index + len(assistant_prefix)] != assistant_prefix:
                index += 1
                continue
            target_start = index + len(assistant_prefix)
            try:
                target_end = full_ids.index(end_token_id, target_start) + 1
            except ValueError:
                break
            labels[target_start:target_end] = full_ids[target_start:target_end]
            index = target_end
        if len(full_ids) > max_length:
            skipped_too_long += 1
            continue
        if not any(label != -100 for label in labels):
            skipped_no_target += 1
            continue
        examples.append(
            {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            }
        )
    return examples, {
        "records": len(records),
        "examples": len(examples),
        "skipped_too_long": skipped_too_long,
        "skipped_no_target": skipped_no_target,
    }


class TokenizedDataset:
    def __init__(self, examples: list[dict[str, list[int]]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


class CausalLmCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_length = max(len(row["input_ids"]) for row in features)
        input_ids = []
        attention_mask = []
        labels = []
        for row in features:
            padding = max_length - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(row["attention_mask"] + [0] * padding)
            labels.append(row["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/sft_fallback_lora.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gate-report", required=True)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--train-max-records", type=int, default=-1)
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    gate_path = (ROOT / args.gate_report).resolve()
    gate_report = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate_report.get("passed") is not False:
        raise RuntimeError("Minimal SFT fallback requires a persisted failed gate report")

    train_records = read_jsonl(ROOT / config["train_data"])
    val_records = read_jsonl(ROOT / config["val_data"])
    if args.train_max_records > 0:
        train_records = train_records[: args.train_max_records]
    required_records = min(
        int(config["min_train_records"]),
        args.train_max_records if args.train_max_records > 0 else int(config["min_train_records"]),
    )
    if len(train_records) < required_records:
        raise RuntimeError(
            f"Fallback train data has {len(train_records)} records; need {required_records}"
        )

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    model_path = (ROOT / config["model_name_or_path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_examples, train_stats = encode_sft_examples(
        tokenizer, train_records, int(config["max_length"])
    )
    val_examples, val_stats = encode_sft_examples(
        tokenizer, val_records, int(config["max_length"])
    )
    if not train_examples:
        raise RuntimeError(f"No train examples after tokenization: {train_stats}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(config["lora_rank"]),
            lora_alpha=int(config["lora_alpha"]),
            lora_dropout=float(config["lora_dropout"]),
            target_modules=config["target_modules"],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    run_dir = (ROOT / "experiments" / args.run_id).resolve()
    checkpoint_dir = run_dir / "checkpoints"
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        do_train=True,
        do_eval=False,
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        num_train_epochs=float(config["epochs"]),
        max_steps=args.max_steps,
        learning_rate=float(config["learning_rate"]),
        warmup_ratio=float(config["warmup_ratio"]),
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        logging_steps=int(config["logging_steps"]),
        logging_first_step=True,
        save_strategy="steps",
        save_steps=int(config["save_steps"]),
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=CausalLmCollator(tokenizer.pad_token_id),
        train_dataset=TokenizedDataset(train_examples),
        eval_dataset=TokenizedDataset(val_examples) if val_examples else None,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate() if val_examples else {}
    final_adapter = checkpoint_dir / "final_adapter"
    model.save_pretrained(final_adapter)
    tokenizer.save_pretrained(final_adapter)
    metrics = {
        "gate_report": args.gate_report,
        "train": train_result.metrics,
        "eval": eval_metrics,
        "train_data": train_stats,
        "val_data": val_stats,
        "final_adapter": str(final_adapter),
        "max_cuda_memory_gib": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
    }
    metrics_path = run_dir / "metrics" / "sft_train.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"SFT_FALLBACK_OK run_id={args.run_id} adapter={final_adapter}")


if __name__ == "__main__":
    main()
