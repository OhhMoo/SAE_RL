"""
Step 5: Train Sparse Autoencoders on cached activations.

Trains a TopK SAE on the residual stream activations collected in step 4.
Trains one SAE per (checkpoint, layer) pair for comparison.

Usage:
    python scripts/05_train_sae.py \
        --activations_dir data/activations \
        --save_dir checkpoints/saes \
        --expansion_factor 8 \
        --k 32
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class TopKSAE(nn.Module):
    """Sparse Autoencoder with TopK activation function."""

    def __init__(self, d_model, d_sae, k):
        super().__init__()
        self.k = k
        self.d_model = d_model
        self.d_sae = d_sae

        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_model, bias=True)

        # Initialize decoder columns to unit norm
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=0
            )

    def encode(self, x):
        z = self.encoder(x)
        # TopK: zero out all but top-k activations
        topk_values, topk_indices = torch.topk(z, self.k, dim=-1)
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(-1, topk_indices, topk_values)
        return z_sparse

    def forward(self, x):
        z_sparse = self.encode(x)
        x_hat = self.decoder(z_sparse)
        return x_hat, z_sparse


def train_sae(activations, d_sae, k, epochs=10, lr=3e-4, batch_size=256, device="cuda"):
    d_model = activations.shape[-1]
    sae = TopKSAE(d_model, d_sae, k).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    dataset = TensorDataset(activations)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        total_l0 = 0
        n_batches = 0

        for (batch,) in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch = batch.to(device).float()
            x_hat, z_sparse = sae(batch)

            # Reconstruction loss
            recon_loss = (batch - x_hat).pow(2).mean()

            loss = recon_loss
            optimizer.zero_grad()
            loss.backward()

            # Normalize decoder columns after gradient step
            optimizer.step()
            with torch.no_grad():
                sae.decoder.weight.data = nn.functional.normalize(
                    sae.decoder.weight.data, dim=0
                )

            total_loss += recon_loss.item()
            total_l0 += (z_sparse != 0).float().sum(dim=-1).mean().item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_l0 = total_l0 / n_batches
        print(f"  Epoch {epoch+1}: recon_loss={avg_loss:.6f}, avg_L0={avg_l0:.1f}")

    return sae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations_dir", type=str, default="data/activations")
    parser.add_argument("--save_dir", type=str, default="checkpoints/saes")
    parser.add_argument("--expansion_factor", type=int, default=8,
                        help="SAE hidden dim = expansion_factor * d_model")
    parser.add_argument("--k", type=int, default=32, help="TopK sparsity")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    activation_files = sorted(Path(args.activations_dir).glob("*.pt"))
    if not activation_files:
        print(f"No activation files found in {args.activations_dir}")
        return

    for act_file in activation_files:
        name = act_file.stem  # e.g., "pretrained_layer12"
        print(f"\n{'='*60}")
        print(f"Training SAE for: {name}")
        print(f"{'='*60}")

        activations = torch.load(act_file, weights_only=True)
        d_model = activations.shape[-1]
        d_sae = d_model * args.expansion_factor

        print(f"  Activations shape: {activations.shape}")
        print(f"  SAE: d_model={d_model}, d_sae={d_sae}, k={args.k}")

        sae = train_sae(
            activations, d_sae, args.k,
            epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, device=args.device,
        )

        save_path = os.path.join(args.save_dir, f"sae_{name}.pt")
        torch.save({
            "state_dict": sae.state_dict(),
            "config": {
                "d_model": d_model,
                "d_sae": d_sae,
                "k": args.k,
                "source": name,
            },
        }, save_path)
        print(f"  Saved SAE -> {save_path}")


if __name__ == "__main__":
    main()
