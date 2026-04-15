#!/usr/bin/env python3
"""
10_classify_features.py
Classify SAE features into "reasoning", "hacking", or "stable" by comparing
activation frequency across the clean learning phase vs the hacking phase.

Inputs:
  - checkpoints/saes/          (original 7 checkpoints, K=32 or matching K)
  - checkpoints/saes_dense/    (dense checkpoints around onset, steps 110-190)
  - data/activations/          (cached activation tensors)

Outputs (to results/precedence_analysis/):
  - feature_frequencies_layer{N}.csv   rows=checkpoints, cols=feature indices
  - feature_classification_layer{N}.csv  feature_idx, hacking_score, class
  - classification_summary.csv          per-layer counts of each class

Usage:
    python scripts/10_classify_features.py
    python scripts/10_classify_features.py --sae_dir checkpoints/saes_k64_10ep --output_dir results/precedence_analysis_k64
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ── SAE model (matches 05_train_sae.py) ───────────────────────────────────────

class TopKSAE(nn.Module):
    def __init__(self, d_model, d_sae, k):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(d_model, d_sae)
        self.decoder = nn.Linear(d_sae, d_model)

    def encode(self, x):
        z = self.encoder(x)
        topk_vals, topk_idx = torch.topk(z, self.k, dim=-1)
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(-1, topk_idx, topk_vals)
        return z_sparse

    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z), z


def load_sae(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    sae = TopKSAE(cfg["d_model"], cfg["d_sae"], cfg["k"])
    sae.load_state_dict(ckpt["state_dict"])
    return sae.to(device).eval(), cfg


# ── Ordered checkpoint sequence ───────────────────────────────────────────────

# All checkpoints in chronological order: original 7 + dense 9
CHECKPOINT_ORDER = [
    "sft",
    "ppo_step10",
    "ppo_step50",
    "ppo_step100",
    # dense onset window
    "ppo_step110",
    "ppo_step120",
    "ppo_step130",
    "ppo_step140",
    "ppo_step150",
    "ppo_step160",
    "ppo_step170",
    "ppo_step180",
    "ppo_step190",
    # hacking phase
    "ppo_step200",
    "ppo_step300",
    "ppo_step435",
]

# Approximate PPO step for each checkpoint (for plotting)
CHECKPOINT_TO_STEP = {
    "sft": 0,
    "ppo_step10": 10,
    "ppo_step50": 50,
    "ppo_step100": 100,
    "ppo_step110": 110,
    "ppo_step120": 120,
    "ppo_step130": 130,
    "ppo_step140": 140,
    "ppo_step150": 150,
    "ppo_step160": 160,
    "ppo_step170": 170,
    "ppo_step180": 180,
    "ppo_step190": 190,
    "ppo_step200": 200,
    "ppo_step300": 300,
    "ppo_step435": 435,
}

# Phase definitions
CLEAN_PHASE  = {"sft", "ppo_step10", "ppo_step50", "ppo_step100"}
HACKING_PHASE = {"ppo_step200", "ppo_step300", "ppo_step435"}


# ── Feature frequency computation ─────────────────────────────────────────────

def compute_feature_freq(sae, acts: torch.Tensor, device: str, batch_size: int = 512):
    """Return activation frequency vector (d_sae,) in [0, 1]."""
    acts = acts.float()
    freq = torch.zeros(sae.encoder.out_features)
    total = 0
    for i in range(0, len(acts), batch_size):
        batch = acts[i : i + batch_size].to(device)
        with torch.no_grad():
            z = sae.encode(batch)
        freq += (z != 0).float().sum(dim=0).cpu()
        total += len(batch)
    return (freq / total).numpy()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae_dir", type=str, default="checkpoints/saes",
                        help="Primary SAE directory (original 7 checkpoints)")
    parser.add_argument("--dense_sae_dir", type=str, default="checkpoints/saes_dense",
                        help="Dense SAE directory (onset window checkpoints)")
    parser.add_argument("--activations_dir", type=str, default="data/activations")
    parser.add_argument("--output_dir", type=str, default="results/precedence_analysis")
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 12, 18, 23])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hacking_threshold", type=float, default=0.005,
                        help="Min |freq_hack - freq_clean| to classify as hacking/reasoning")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover all available SAE files across both dirs
    sae_paths: dict[tuple, Path] = {}
    for sae_dir in [args.sae_dir, args.dense_sae_dir]:
        if not os.path.isdir(sae_dir):
            continue
        for f in Path(sae_dir).glob("sae_*.pt"):
            stem = f.stem[len("sae_"):]
            parts = stem.rsplit("_layer", 1)
            if len(parts) == 2:
                stage, layer_str = parts
                try:
                    layer = int(layer_str)
                    sae_paths[(stage, layer)] = f
                except ValueError:
                    pass

    # Discover available activation files
    act_paths: dict[tuple, Path] = {}
    for f in Path(args.activations_dir).glob("*.pt"):
        stem = f.stem
        parts = stem.rsplit("_layer", 1)
        if len(parts) == 2:
            stage, layer_str = parts
            try:
                layer = int(layer_str)
                act_paths[(stage, layer)] = f
            except ValueError:
                pass

    summary_rows = []

    for layer in args.layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer}")
        print(f"{'='*60}")

        # Collect frequency vectors for all available checkpoints
        freq_records = []  # list of (stage, ppo_step, freq_vector)

        for stage in CHECKPOINT_ORDER:
            sae_key = (stage, layer)
            act_key = (stage, layer)

            if sae_key not in sae_paths:
                print(f"  [skip] {stage} — SAE not found")
                continue
            if act_key not in act_paths:
                print(f"  [skip] {stage} — activations not found")
                continue

            print(f"  Computing freq: {stage}", end="", flush=True)
            sae, cfg = load_sae(str(sae_paths[sae_key]), args.device)
            acts = torch.load(str(act_paths[act_key]), weights_only=True)
            freq = compute_feature_freq(sae, acts, args.device)
            ppo_step = CHECKPOINT_TO_STEP.get(stage, -1)
            freq_records.append((stage, ppo_step, freq))
            n_active = (freq > 0.01).sum()
            print(f"  →  {n_active}/{len(freq)} active (>{1}%)")

        if len(freq_records) < 2:
            print(f"  [warn] Not enough checkpoints for layer {layer}, skipping")
            continue

        d_sae = len(freq_records[0][2])

        # ── Build frequency matrix ─────────────────────────────────────────
        stages_present   = [r[0] for r in freq_records]
        steps_present    = [r[1] for r in freq_records]
        freq_matrix      = np.stack([r[2] for r in freq_records])  # (n_ckpts, d_sae)

        freq_df = pd.DataFrame(
            freq_matrix,
            index=pd.MultiIndex.from_arrays([stages_present, steps_present],
                                            names=["stage", "ppo_step"]),
            columns=[f"feat_{i}" for i in range(d_sae)],
        )
        freq_csv = os.path.join(args.output_dir, f"feature_frequencies_layer{layer}.csv")
        freq_df.to_csv(freq_csv)
        print(f"  Saved frequency matrix → {freq_csv}")

        # ── Classify features ──────────────────────────────────────────────
        clean_mask  = np.array([s in CLEAN_PHASE  for s in stages_present])
        hacking_mask = np.array([s in HACKING_PHASE for s in stages_present])

        if clean_mask.sum() == 0 or hacking_mask.sum() == 0:
            print("  [warn] Missing clean or hacking phase checkpoints — cannot classify")
            continue

        freq_clean  = freq_matrix[clean_mask].mean(axis=0)   # (d_sae,)
        freq_hack   = freq_matrix[hacking_mask].mean(axis=0) # (d_sae,)
        hacking_score = freq_hack - freq_clean                # positive → more active in hacking

        # Classification
        labels = np.full(d_sae, "stable", dtype=object)
        labels[hacking_score >  args.hacking_threshold] = "hacking"
        labels[hacking_score < -args.hacking_threshold] = "reasoning"

        n_hacking   = (labels == "hacking").sum()
        n_reasoning = (labels == "reasoning").sum()
        n_stable    = (labels == "stable").sum()
        print(f"  Classification:  hacking={n_hacking}  reasoning={n_reasoning}  stable={n_stable}")

        clf_df = pd.DataFrame({
            "feature_idx":   np.arange(d_sae),
            "freq_clean":    freq_clean,
            "freq_hack":     freq_hack,
            "hacking_score": hacking_score,
            "class":         labels,
        })
        clf_csv = os.path.join(args.output_dir, f"feature_classification_layer{layer}.csv")
        clf_df.to_csv(clf_csv, index=False)
        print(f"  Saved classification → {clf_csv}")

        summary_rows.append({
            "layer":      layer,
            "n_total":    d_sae,
            "n_hacking":  int(n_hacking),
            "n_reasoning": int(n_reasoning),
            "n_stable":   int(n_stable),
            "n_checkpoints": len(freq_records),
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(args.output_dir, "classification_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"\nSummary saved → {summary_csv}")
        print(summary_df.to_string(index=False))

    print(f"\nNext: python scripts/11_temporal_precedence.py --analysis_dir {args.output_dir}")


if __name__ == "__main__":
    main()
