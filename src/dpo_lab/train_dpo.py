# train_dpo.py
#
# Stage 2 of the DPO virtual lab: fine-tune the same base model on one arm of
# preference data at a given data fraction (dose-response design).
#
# LoRA + DPO (trl's DPOTrainer) instead of full RLHF/PPO: one supervised pass,
# no separate reward model, feasible on a single A100 for a 7B model.
#
# Run once per (arm, fraction) cell on a machine with an NVIDIA GPU
# (Shusei's desktop PC — do NOT submit to Discovery unless explicitly decided):
#   python train_dpo.py --arm human --fraction 0.25
#   python train_dpo.py --arm ai    --fraction 1.0
# Debug the pipeline first with a small model:
#   python train_dpo.py --arm human --fraction 0.1 --base-model Qwen/Qwen2.5-0.5B-Instruct
#
# Output: LoRA adapter + tokenizer in models/dpo_lab/{arm}-{pct}pct/

import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import PATH_TO_DPO_DATA, PATH_TO_DPO_CHECKPOINTS, DPO_BASE_MODEL


def load_pairs(arm, fraction):
    path = os.path.join(PATH_TO_DPO_DATA, f"{arm}_prefs.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — run prepare_data.py first")

    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("skip"):
                continue
            pairs.append(
                {"prompt": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"]}
            )

    n = max(1, int(len(pairs) * fraction))
    print(f"{arm} arm: {len(pairs)} pairs available, training on first {n} ({fraction:.0%})")
    return pairs[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["human", "ai"], required=True)
    parser.add_argument("--fraction", type=float, required=True, help="e.g. 0.1, 0.25, 0.5, 1.0")
    parser.add_argument("--base-model", default=DPO_BASE_MODEL)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tag = f"{args.arm}-{int(args.fraction * 100)}pct"
    output_dir = os.path.join(PATH_TO_DPO_CHECKPOINTS, tag)
    os.makedirs(output_dir, exist_ok=True)

    train_dataset = Dataset.from_list(load_pairs(args.arm, args.fraction))

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    training_args = DPOConfig(
        output_dir=output_dir,
        beta=args.beta,
        max_length=1024,
        max_prompt_length=512,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="no",
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        report_to="none",
        seed=42,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(output_dir)  # saves the LoRA adapter
    tokenizer.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump({**vars(args), "n_pairs": len(train_dataset), "tag": tag}, f, indent=2)

    print(f"Saved adapter to {output_dir}")


if __name__ == "__main__":
    main()
