from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_config(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".json":
        raise ValueError("training config must be JSON so cluster runs need no YAML dependency")
    return json.loads(path.read_text(encoding="utf-8"))


def _gate(config: dict[str, Any]) -> list[str]:
    failures = []
    audit_path = Path(config["audit_report"])
    if not audit_path.exists() or not json.loads(audit_path.read_text())["ok"]:
        failures.append("data audit is missing or failed")
    trajectory_path = Path(config["trajectories"])
    if not trajectory_path.exists():
        failures.append("trajectory file is missing")
    else:
        count = sum(1 for line in trajectory_path.open(encoding="utf-8") if line.strip())
        if count < int(config.get("minimum_trajectories", 200)):
            failures.append(f"only {count} trajectories are available")
    return failures


def _sft(config: dict[str, Any]) -> None:
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("install the 'ml,train' extras before training") from exc

    model_name = config.get("model", "Qwen/Qwen2.5-1.5B-Instruct")
    rows = [json.loads(line) for line in Path(config["trajectories"]).open(encoding="utf-8")]
    texts = [
        {"text": row["prompt"] + "\n" + json.dumps(row["action"], sort_keys=True)}
        for row in rows
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quantization, device_map="auto"
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list(texts),
        peft_config=LoraConfig(
            r=int(config.get("lora_rank", 16)),
            lora_alpha=int(config.get("lora_alpha", 32)),
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
        args=SFTConfig(
            output_dir=config["output_dir"],
            dataset_text_field="text",
            max_length=int(config.get("max_length", 2048)),
            num_train_epochs=float(config.get("epochs", 1)),
            learning_rate=float(config.get("learning_rate", 2e-4)),
            per_device_train_batch_size=int(config.get("batch_size", 1)),
            gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 16)),
            bf16=True,
            logging_steps=5,
            save_strategy="epoch",
            report_to="none",
        ),
    )
    trainer.train()
    trainer.save_model(config["output_dir"])


def run_training(kind: str, config_path: Path) -> int:
    config = _load_config(config_path)
    failures = _gate(config)
    report = {"kind": kind, "gate_passed": not failures, "failures": failures}
    gate_output = Path(config.get("gate_output", f"{kind}_gate.json"))
    gate_output.parent.mkdir(parents=True, exist_ok=True)
    gate_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if failures:
        print(json.dumps(report, indent=2))
        return 2
    if config.get("dry_run", False):
        print(json.dumps(report, indent=2))
        return 0
    if kind == "sft":
        _sft(config)
        return 0
    raise RuntimeError(
        "GRPO is intentionally gated until the interactive rollout adapter is validated; "
        "set dry_run=true to validate readiness without launching training"
    )

