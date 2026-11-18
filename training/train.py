"""Phase 4 — QLoRA fine-tuning on Spider.

Fine-tunes Qwen2.5-3B-Instruct (or any causal LM) on the preprocessed Spider
training set using 4-bit QLoRA via HuggingFace PEFT + TRL's SFTTrainer.

Usage:
    # Single GPU (RTX 4090 / A10 — 12-16 GB VRAM is sufficient for a 3B model)
    python training/train.py

    # With custom config
    python training/train.py --config configs/config.yaml

    # Sweep: override specific hparams from CLI (for Phase 5)
    python training/train.py --lora-r 8 --lr 1e-4 --run-name sweep_r8_lr1e4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from utils import get_logger, load_config  # noqa

log = get_logger("train")


def build_training_args(cfg: dict, output_dir: str, run_name: str):
    """Build TRL SFTConfig (superset of TrainingArguments) from config."""
    from trl import SFTConfig

    import torch

    tc = cfg["training"]
    # bf16 needs Ampere or newer; fall back to fp16 on older cluster GPUs.
    bf16 = tc["bf16"] and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return SFTConfig(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=tc["num_epochs"],
        per_device_train_batch_size=tc["per_device_train_batch_size"],
        per_device_eval_batch_size=tc["per_device_train_batch_size"] * 2,
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        learning_rate=tc["learning_rate"],
        lr_scheduler_type=tc["lr_scheduler"],
        warmup_steps=tc["warmup_ratio"],  # float in [0, 1) = ratio of total steps
        fp16=tc["fp16"] and not bf16,
        bf16=bf16,
        logging_steps=tc["logging_steps"],
        eval_strategy="steps",
        eval_steps=tc["eval_steps"],
        save_strategy="steps",
        save_steps=tc["save_steps"],
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        dataloader_num_workers=4,
        max_length=tc["max_seq_length"],
        # Prompt/completion dataset → loss only on the SQL completion tokens.
        completion_only_loss=True,
        gradient_checkpointing=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--lora-r", type=int, default=None, help="Override LoRA rank")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Override num epochs")
    parser.add_argument("--run-name", default="qwen-text2sql-qlora", help="W&B run name")
    parser.add_argument("--resume-from", default=None,
                        help="Checkpoint dir to resume from (used by the Slurm requeue loop)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.lora_r:
        cfg["training"]["lora_r"] = args.lora_r
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr
    if args.epochs:
        cfg["training"]["num_epochs"] = args.epochs

    tc = cfg["training"]

    # ── Imports (deferred so the script can be imported without GPU) ──────────
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTTrainer

    # ── W&B initialisation ────────────────────────────────────────────────────
    if os.getenv("WANDB_API_KEY"):
        import wandb
        wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"]["entity"],
            name=args.run_name,
            config={
                "model": cfg["model"]["base_id"],
                **tc,
            },
        )
        log.info("W&B run: %s", wandb.run.url)
    else:
        log.warning("WANDB_API_KEY not set — training metrics logged locally only.")

    # ── Load dataset ──────────────────────────────────────────────────────────
    train_file = cfg["data"]["train_file"]
    dev_file = cfg["data"]["dev_file"]

    if not Path(train_file).exists():
        raise FileNotFoundError(f"{train_file} not found — run prepare_data.py first.")

    log.info("Loading dataset …")
    dataset = load_dataset(
        "json",
        data_files={"train": train_file, "validation": dev_file},
    )

    # TRL prompt/completion format: the prompt already ends with "[SQL]", the
    # completion is the gold query (TRL appends EOS, so the model learns to stop).
    dataset = dataset.map(
        lambda ex: {"prompt": ex["prompt"], "completion": "\n" + ex["sql"]},
        remove_columns=dataset["train"].column_names,
    )

    log.info("Train: %d  |  Validation: %d", len(dataset["train"]), len(dataset["validation"]))

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    model_id = cfg["model"]["base_id"]
    log.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Model in 4-bit ────────────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    log.info("Loading base model in 4-bit …")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # ── LoRA adapters ─────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=tc["lora_r"],
        lora_alpha=tc["lora_alpha"],
        lora_dropout=tc["lora_dropout"],
        target_modules=tc["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training arguments ────────────────────────────────────────────────────
    output_dir = tc["output_dir"]
    training_args = build_training_args(cfg, output_dir, args.run_name)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    log.info("Starting training …")
    if args.resume_from:
        log.info("Resuming from %s", args.resume_from)
    trainer.train(resume_from_checkpoint=args.resume_from)

    # ── Save best checkpoint ──────────────────────────────────────────────────
    best_dir = Path(output_dir) / "best_checkpoint"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    log.info("Best checkpoint saved → %s", best_dir)

    if os.getenv("WANDB_API_KEY"):
        wandb.finish()


if __name__ == "__main__":
    main()
