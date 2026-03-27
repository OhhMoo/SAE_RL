"""
Step 4: Collect residual stream activations from model checkpoints.

Runs the model on GSM8k prompts and caches activations from specified layers.
These cached activations are used to train SAEs in the next step.

Usage:
    python scripts/04_collect_activations.py \
        --model_path Qwen/Qwen2.5-0.5B-Instruct \
        --checkpoint_name pretrained \
        --layers 6 12 18 23 \
        --save_dir data/activations
"""

import argparse
import os

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def collect_activations(model, tokenizer, prompts, layers, max_length=512, batch_size=8):
    """Run forward passes and collect residual stream activations at specified layers."""
    device = model.device
    hooks = []
    activations = {layer: [] for layer in layers}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is a tuple; first element is the hidden state
            hidden = output[0] if isinstance(output, tuple) else output
            # Take the mean over the sequence dimension to get a per-example vector
            activations[layer_idx].append(hidden.detach().mean(dim=1).cpu())
        return hook_fn

    # Register hooks on the specified transformer layers
    for layer_idx in layers:
        hook = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)

    try:
        for i in tqdm(range(0, len(prompts), batch_size), desc="Collecting activations"):
            batch = prompts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            with torch.no_grad():
                model(**inputs)
    finally:
        for hook in hooks:
            hook.remove()

    # Concatenate all batches
    result = {}
    for layer_idx in layers:
        result[layer_idx] = torch.cat(activations[layer_idx], dim=0)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--checkpoint_name", type=str, required=True,
                        help="Label for this checkpoint: pretrained, sft, or ppo")
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 12, 18, 23],
                        help="Which transformer layers to collect from")
    parser.add_argument("--save_dir", type=str, default="data/activations")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of prompts (for quick testing)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Loading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load GSM8k prompts
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    prompts = [ex["question"] for ex in dataset]
    if args.max_samples:
        prompts = prompts[: args.max_samples]

    print(f"Collecting activations from {len(prompts)} prompts, layers {args.layers}")
    acts = collect_activations(
        model, tokenizer, prompts, args.layers, args.max_length, args.batch_size
    )

    for layer_idx, tensor in acts.items():
        save_path = os.path.join(
            args.save_dir, f"{args.checkpoint_name}_layer{layer_idx}.pt"
        )
        torch.save(tensor, save_path)
        print(f"Saved layer {layer_idx}: {tensor.shape} -> {save_path}")


if __name__ == "__main__":
    main()
