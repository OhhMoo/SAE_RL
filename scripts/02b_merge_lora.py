"""
Step 2b: Merge LoRA adapter weights back into the base model.

After SFT with LoRA, verl saves adapter weights separately. This script
merges them into a full HuggingFace model that can be used as the PPO
actor initialization and for SAE activation collection.

Usage:
    python scripts/02b_merge_lora.py \
        --base_model Qwen/Qwen2.5-0.5B-Instruct \
        --lora_path checkpoints/sft/sae_rl_gsm8k/sft_qwen2.5_0.5b/global_step_XXX/actor \
        --output_path checkpoints/sft_merged
"""

import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading LoRA adapter: {args.lora_path}")
    model = PeftModel.from_pretrained(model, args.lora_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {args.output_path}")
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print("Done.")


if __name__ == "__main__":
    main()
