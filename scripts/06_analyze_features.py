"""
Step 6: Compare SAE features across training stages.

Loads trained SAEs from each checkpoint (pretrained, sft, ppo) and analyzes:
- Feature activation frequency and magnitude distributions
- Feature cosine similarity between stages (feature drift)
- Features that appear or disappear between stages

Usage:
    python scripts/06_analyze_features.py \
        --sae_dir checkpoints/saes \
        --activations_dir data/activations \
        --output_dir results/feature_analysis
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# Import our SAE architecture
import sys
sys.path.insert(0, os.path.dirname(__file__))

# Inline the SAE class import to avoid module naming issues
# The TopKSAE class is defined in 05_train_sae.py
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "train_sae", os.path.join(os.path.dirname(__file__), "05_train_sae.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TopKSAE = _mod.TopKSAE


def load_sae(sae_path, device="cpu"):
    checkpoint = torch.load(sae_path, weights_only=False, map_location=device)
    config = checkpoint["config"]
    sae = TopKSAE(config["d_model"], config["d_sae"], config["k"])
    sae.load_state_dict(checkpoint["state_dict"])
    sae.eval()
    return sae, config


def compute_feature_stats(sae, activations, device="cpu"):
    """Compute per-feature activation statistics."""
    sae = sae.to(device)
    activations = activations.to(device).float()

    with torch.no_grad():
        z_sparse = sae.encode(activations)

    # Feature activation frequency: how often each feature fires
    active = (z_sparse != 0).float()
    freq = active.mean(dim=0).cpu().numpy()

    # Mean activation magnitude when active
    z_sum = z_sparse.sum(dim=0).cpu().numpy()
    count = active.sum(dim=0).cpu().numpy()
    mean_mag = np.divide(z_sum, count, out=np.zeros_like(z_sum), where=count > 0)

    return {"freq": freq, "mean_magnitude": mean_mag, "z_sparse": z_sparse.cpu()}


def plot_frequency_comparison(stats_by_stage, layer, output_dir):
    """Compare feature activation frequencies across stages."""
    fig, axes = plt.subplots(1, len(stats_by_stage), figsize=(5 * len(stats_by_stage), 4))
    if len(stats_by_stage) == 1:
        axes = [axes]

    for ax, (stage, stats) in zip(axes, stats_by_stage.items()):
        freq = stats["freq"]
        ax.hist(freq[freq > 0], bins=50, alpha=0.7)
        ax.set_title(f"{stage} (layer {layer})")
        ax.set_xlabel("Activation frequency")
        ax.set_ylabel("Number of features")
        ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"freq_comparison_layer{layer}.png"), dpi=150)
    plt.close()


def plot_feature_drift(stats_by_stage, layer, output_dir):
    """Compute decoder weight cosine similarity between stages."""
    stages = list(stats_by_stage.keys())
    if len(stages) < 2:
        return

    # Compare consecutive stages
    for i in range(len(stages) - 1):
        s1, s2 = stages[i], stages[i + 1]
        # Use decoder weights as feature directions
        # We'd need the SAEs loaded here - this is a placeholder
        # In practice, compare the decoder weight columns
        print(f"  Feature drift {s1} -> {s2}: (requires decoder weight comparison)")


def compute_dead_alive_features(stats_by_stage, threshold=0.01):
    """Find features that are born or die between stages."""
    stages = list(stats_by_stage.keys())
    results = {}

    for i in range(len(stages) - 1):
        s1, s2 = stages[i], stages[i + 1]
        alive1 = stats_by_stage[s1]["freq"] > threshold
        alive2 = stats_by_stage[s2]["freq"] > threshold

        born = (~alive1) & alive2  # dead in s1, alive in s2
        died = alive1 & (~alive2)  # alive in s1, dead in s2

        results[f"{s1}->{s2}"] = {
            "born": born.sum(),
            "died": died.sum(),
            "stable": (alive1 & alive2).sum(),
            "always_dead": (~alive1 & ~alive2).sum(),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae_dir", type=str, default="checkpoints/saes")
    parser.add_argument("--activations_dir", type=str, default="data/activations")
    parser.add_argument("--output_dir", type=str, default="results/feature_analysis")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Group SAEs and activations by layer
    sae_files = sorted(Path(args.sae_dir).glob("sae_*.pt"))
    act_files = sorted(Path(args.activations_dir).glob("*.pt"))

    # Parse filenames: sae_{stage}_layer{N}.pt, {stage}_layer{N}.pt
    by_layer = defaultdict(dict)
    sae_by_key = {}
    for f in sae_files:
        # e.g., sae_pretrained_layer12.pt -> stage="pretrained", layer=12
        parts = f.stem.replace("sae_", "").rsplit("_layer", 1)
        if len(parts) == 2:
            stage, layer = parts[0], int(parts[1])
            by_layer[layer][stage] = None  # placeholder
            sae_by_key[(stage, layer)] = f

    act_by_key = {}
    for f in act_files:
        parts = f.stem.rsplit("_layer", 1)
        if len(parts) == 2:
            stage, layer = parts[0], int(parts[1])
            act_by_key[(stage, layer)] = f

    # Analyze each layer
    for layer in sorted(by_layer.keys()):
        print(f"\n{'='*60}")
        print(f"Layer {layer}")
        print(f"{'='*60}")

        stats_by_stage = {}
        stage_order = ["pretrained", "sft", "ppo"]

        for stage in stage_order:
            key = (stage, layer)
            if key not in sae_by_key or key not in act_by_key:
                continue

            print(f"\n  Stage: {stage}")
            sae, config = load_sae(sae_by_key[key], args.device)
            acts = torch.load(act_by_key[key], weights_only=True)

            stats = compute_feature_stats(sae, acts, args.device)
            stats_by_stage[stage] = stats

            n_active = (stats["freq"] > 0.01).sum()
            n_dead = (stats["freq"] <= 0.01).sum()
            print(f"    Active features (>1%): {n_active}")
            print(f"    Dead features (<=1%): {n_dead}")
            print(f"    Mean freq (active): {stats['freq'][stats['freq'] > 0.01].mean():.4f}")

        if stats_by_stage:
            plot_frequency_comparison(stats_by_stage, layer, args.output_dir)

            lifecycle = compute_dead_alive_features(stats_by_stage)
            for transition, counts in lifecycle.items():
                print(f"\n  Feature lifecycle {transition}:")
                for k, v in counts.items():
                    print(f"    {k}: {v}")

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
