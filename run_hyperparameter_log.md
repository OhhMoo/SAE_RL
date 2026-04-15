# SAE-RL Experiment Run Hyperparameter Log

**Project:** `sae_rl_gsm8k`
**Task:** GSM8k chain-of-thought math reasoning (SFT → PPO → SAE interpretability)
**Model backbone:** Qwen/Qwen2.5-0.5B-Instruct (d_model=896, 24 layers)
**WandB project:** `sae_rl_gsm8k`

---

## Hardware & Environment

| Item | Value |
|------|-------|
| Host | `entropie-chem-hmc-edu` |
| GPUs | 2× NVIDIA RTX A6000 (48 GB VRAM each, Ampere, 10752 CUDA cores) |
| CPU | 64 physical / 128 logical cores |
| RAM | 540 GB |
| CUDA | 12.8 |
| Python | CPython 3.11.15 |
| Conda env | `sae_rl` |
| Framework | verl (FSDP backend), Ray + vLLM (PPO rollout) |
| Config system | Hydra (outputs saved to `outputs/<date>/<time>/`) |

---

## Pipeline Overview

```
01_prepare_data.py        →  data/gsm8k/          (PPO parquet: prompt + reward_model)
01b_prepare_sft_data.py   →  data/gsm8k_sft/      (SFT parquet: messages column)
02_sft_qwen.sh            →  checkpoints/sft*      (LoRA SFT via verl sft_trainer)
02b_merge_lora.py         →  checkpoints/sft*_merged
03b_eval_sft.py           →  format/accuracy gate before PPO
03_ppo_qwen*.sh           →  checkpoints/sae_rl_gsm8k/ppo_*/  (verl main_ppo)
run_stage3_pipeline.sh    →  merge PPO ckpts → activations → SAEs
04_collect_activations.py →  data/activations/
05_train_sae.py           →  checkpoints/saes/
06_analyze_features.py    →  results/feature_analysis/
07_eval_sae_metrics.py    →  results/sae_eval_metrics.csv
```

---

## Stage 1 — Data Preparation

Both datasets derived from `openai/gsm8k` (main config, 7473 train / 1319 test).

**System prompt (shared):**
> "You are a math problem solver. Think step by step. You MUST end your response with '#### \<number\>' where \<number\> is the final numerical answer (digits only, no units or markdown)."

**Instruction suffix appended to each question:**
> "Let's think step by step and output the final answer after '####'."

| Output | Format | Used by |
|--------|--------|---------|
| `data/gsm8k/train.parquet` | `prompt` (chat list) + `reward_model.ground_truth` (exact match) | PPO |
| `data/gsm8k_sft/train.parquet` | `messages` (system + user + full CoT assistant) | SFT |

---

## Stage 2 — Supervised Fine-Tuning (SFT)

**Runner:** `verl.trainer.sft_trainer` via `torchrun --standalone --nnodes=1`

All runs used: **FSDP engine, bfloat16, AdamW optimizer, LoRA on all linear layers.**

### Run History

| Timestamp | Experiment name | Key difference | Outcome |
|-----------|----------------|----------------|---------|
| 2026-03-27 02:01 | `sft_qwen2.5_0.5b` | Wrong data (`gsm8k/` not `gsm8k_sft/`), no `messages_key`, Windows paths | **Failed** — wrong column format |
| 2026-03-27 02:06 | `sft_qwen2.5_0.5b` | Fixed to `gsm8k_sft/`, added `messages_key=messages` | **Failed** — no SDPA, attention error |
| 2026-03-27 02:07 | `sft_qwen2.5_0.5b` | Identical re-run | **Failed** |
| 2026-03-27 02:12 | `sft_qwen2.5_0.5b` | Added `+model.override_config.attn_implementation=sdpa` | **Failed** — still on Windows |
| 2026-03-27 02:18 | `sft_qwen2.5_0.5b` | Identical re-run on Windows | **Failed** |
| **2026-03-31 00:03** | `sft_qwen2.5_0.5b` | Moved to Linux; added `engine.use_torch_compile=false` | **SUCCESS** → `checkpoints/sft/` |
| **2026-03-31 17:14** | `sft_qwen2.5_0.5b` | SFT v2: increased `total_epochs=5`, `lora_alpha=32` | **SUCCESS** → `checkpoints/sft_v2/` |

### SFT v1 Hyperparameters (2026-03-31 00:03 → `checkpoints/sft/`)

| Hyperparameter | Value |
|----------------|-------|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| LoRA rank | 32 |
| LoRA alpha | 16 |
| LoRA target modules | all-linear |
| Learning rate | 1e-4 (constant schedule) |
| Optimizer | AdamW (`β₁=0.9, β₂=0.999, weight_decay=0.01`) |
| Gradient clip | 1.0 |
| Total epochs | 3 |
| Train batch size | 256 |
| Micro batch per GPU | 4 |
| Max sequence length | 1024 |
| Dynamic batching | enabled |
| Attention | SDPA (PyTorch native) |
| `use_torch_compile` | false |
| Gradient checkpointing | enabled |
| Checkpoint path | `checkpoints/sft/` |

**Final training metrics (WandB run `dcy4yfc7`):**
- Train loss: **0.330** | Val loss: **0.499**
- Total tokens processed: ~65k (0.009B)
- Wall time: ~1444 s (24 min)
- Global steps: 145

### SFT v2 Hyperparameters (2026-03-31 17:14 → `checkpoints/sft_v2/`)

Same as v1 except:

| Hyperparameter | v1 | v2 |
|----------------|----|----|
| Total epochs | 3 | **5** |
| LoRA alpha | 16 | **32** |
| Checkpoint path | `checkpoints/sft/` | `checkpoints/sft_v2/` |

> Note: Setting `lora_alpha = lora_rank = 32` means the LoRA scaling factor `alpha/rank = 1.0` (no down-scaling of LoRA updates).

### LoRA Merge (02b_merge_lora.py)

FSDP checkpoints converted to HuggingFace PEFT format with `verl.model_merger`, then merged to a flat model:

```
checkpoints/sft/global_step_87   → checkpoints/sft_merged/
checkpoints/sft_v2/global_step_* → checkpoints/sft_v2_merged/
```

### SFT Evaluation Gate (03b_eval_sft.py)

Greedy decoding on 200 test samples before proceeding to PPO.

| Parameter | Value |
|-----------|-------|
| n_samples | 200 |
| batch_size | 8 |
| max_new_tokens | 512 |
| min_format_rate | 0.80 (exit code 1 if below) |
| min_accuracy | 0.25 (exit code 1 if below) |
| Baseline SFT accuracy | ~33% |

---

## Stage 3 — PPO Training

**Runner:** `verl.trainer.main_ppo` via Ray + vLLM
**Algorithm:** PPO with GAE advantage estimation, KL-penalized reward
**Infrastructure:** 2 GPUs, FSDP, async vLLM rollout engine

### Run History

| Timestamp | Experiment name | Actor init | Critic init | Rollout n | Epochs | Outcome |
|-----------|----------------|------------|-------------|-----------|--------|---------|
| 2026-03-31 00:36–01:11 (×8) | `ppo_qwen2.5_0.5b` | `sft_merged` | `sft_merged` | 1 (default) | 5† | **Crashed** — repeated CUDA/vLLM init failures |
| **2026-03-31 13:22** | `ppo_qwen2.5_0.5b` | `sft_merged` | `sft_merged` | 1 (default) | 15 | Partial run |
| **2026-03-31 13:25** | `ppo_qwen2.5_0.5b` | `sft_merged` | `sft_merged` | 1 (default) | 15 | Partial run |
| **2026-03-31 13:31** | `ppo_qwen2.5_0.5b` | `sft_merged` | `sft_merged` | 1 (default) | 15 | **SUCCESS** → `checkpoints/sae_rl_gsm8k/ppo_qwen2.5_0.5b/` |
| 2026-03-31 17:52 | `ppo_qwen2.5_0.5b` | `sft_v2_merged` | base Qwen | **8** | 15 | Crashed |
| 2026-03-31 18:00 | `ppo_qwen2.5_0.5b` | `sft_v2_merged` | base Qwen | **8** | 15 | Crashed |
| 2026-03-31 18:03 | `ppo_qwen2.5_0.5b_v2` | `sft_v2_merged` | base Qwen | **8** | 15 | Crashed |
| **2026-04-01 11:28** | **`ppo_qwen2.5_0.5b_v3`** | `sft_v2_merged` | base Qwen | **8** | **5** | **SUCCESS** |

† Early batch had `trainer.total_epochs=15` overridden by `trainer.total_epochs=5` (duplicate override key), giving 5 effective epochs. All 8 crashed before producing checkpoints.

### PPO Hyperparameters (main run: 2026-03-31 13:31, `ppo_qwen2.5_0.5b`)

**Data:**

| Parameter | Value |
|-----------|-------|
| Train files | `data/gsm8k/train.parquet` |
| Val files | `data/gsm8k/test.parquet` |
| Train batch size | 256 |
| Max prompt length | 512 |
| Max response length | 512 |
| Filter overlong prompts | True |
| Reward | Rule-based exact match on `#### <answer>` |

**Actor:**

| Parameter | Value |
|-----------|-------|
| Model | `checkpoints/sft_merged` |
| Attention | SDPA |
| LoRA | disabled (rank=0, full fine-tune from SFT weights) |
| Optimizer | AdamW |
| Learning rate | 1e-6 (constant) |
| Weight decay | 0.01 |
| Grad clip | 1.0 |
| PPO mini-batch size | 64 |
| PPO micro-batch per GPU | 8 |
| PPO epochs per rollout | 1 |
| Clip ratio (ε) | 0.2 |
| Gradient checkpointing | enabled |
| `use_remove_padding` | False |
| `use_kl_loss` | False (KL applied in reward instead) |

**Critic:**

| Parameter | Value |
|-----------|-------|
| Model | `checkpoints/sft_merged` |
| Optimizer | AdamW |
| Learning rate | 1e-5 (constant) |
| Clip range value | 0.5 |
| Gradient checkpointing | disabled |
| PPO micro-batch per GPU | 8 |

**Rollout (vLLM):**

| Parameter | Value |
|-----------|-------|
| Engine | vLLM (async mode) |
| Samples per prompt (n) | 1 |
| Temperature | 1.0 |
| Top-p | 1.0 |
| Top-k | -1 (disabled) |
| GPU memory utilization | 0.4 |
| Tensor model parallel size | 1 |
| Max batched tokens | 8192 |
| Max num seqs | 1024 |
| Chunked prefill | enabled |
| Prefix caching | enabled |

**Algorithm:**

| Parameter | Value |
|-----------|-------|
| Advantage estimator | GAE |
| γ (gamma) | 1.0 |
| λ (lambda) | 1.0 |
| KL penalty | `kl` (direct KL divergence added to reward) |
| KL coefficient | 0.001 (fixed) |
| KL target | 0.1 |
| KL horizon | 10000 |
| `use_kl_in_reward` | True |

**Trainer:**

| Parameter | Value |
|-----------|-------|
| Total epochs | 15 |
| Steps per epoch | ~29 (7473 samples / batch 256) |
| Total steps | ~435 |
| n GPUs | 2 |
| Save frequency | every 5 steps |
| Test/val frequency | every 5 steps |
| Checkpoint path | `checkpoints/sae_rl_gsm8k/ppo_qwen2.5_0.5b/` |
| Critic warmup | 0 |
| Seed | 42 (FSDP), 1 (trainer) |

**Saved checkpoints used for SAE analysis:**
`global_step_10, 50, 100, 200, 300, 435`

**Observed training behavior:**

| Steps | GSM8k solve rate | Notes |
|-------|-----------------|-------|
| 0 (SFT baseline) | ~33% | Starting point |
| 10 | ~36% | First RL signal applied |
| 50 | ~56% | Rapid improvement phase |
| 100 | ~67% | Strong task learning |
| 200 | ~79%* | Possibly reward hacking begins |
| 300 | ~12%* | Reward hacking confirmed (response length collapsed to ~5–6 tokens) |
| 435 | ~13%* | Continued degenerate mode |

\* Steps 200–435: model learned to output short strings matching `#### <number>` format without genuine reasoning. Affects SAE interpretation for these checkpoints.

---

## Stage 4 — Activation Collection

**Script:** `04_collect_activations.py`
**Method:** Forward hooks on `model.model.layers[N]` residual stream outputs; padding tokens excluded via attention mask.

| Parameter | Value |
|-----------|-------|
| Source dataset | GSM8k train split (question-only, no answers) |
| Layers tapped | 6, 12, 18, 23 |
| Max tokens per (checkpoint × layer) | 500,000 |
| Actual tokens collected | ~222,000 per file |
| Batch size | 16 (stage3 pipeline) |
| Max sequence length | 512 |
| Model dtype | float16 |
| Output shape | `(n_tokens, 896)` per `.pt` file |
| Output file size | ~762 MB per file |
| Total files | 28 (7 stages × 4 layers) |

**Checkpoints processed:**

| Stage label | Model path |
|-------------|-----------|
| `sft` | `checkpoints/sft_merged` |
| `ppo_step10` | `checkpoints/ppo_merged/step_10` |
| `ppo_step50` | `checkpoints/ppo_merged/step_50` |
| `ppo_step100` | `checkpoints/ppo_merged/step_100` |
| `ppo_step200` | `checkpoints/ppo_merged/step_200` |
| `ppo_step300` | `checkpoints/ppo_merged/step_300` |
| `ppo_step435` | `checkpoints/ppo_merged/step_435` |

---

## Stage 5 — SAE Training

**Script:** `05_train_sae.py` (hand-rolled TopK SAE)
**Run via:** `run_stage3_pipeline.sh`

One SAE trained per `(stage, layer)` pair → 28 total SAEs.

### Architecture

| Parameter | Value |
|-----------|-------|
| Type | TopK Sparse Autoencoder |
| d_model (input/output) | 896 |
| Expansion factor | 8 |
| d_sae (hidden dim) | 7168 |
| k (active features per token) | 32 |
| Encoder | Linear(896 → 7168) + bias |
| Decoder | Linear(7168 → 896) + bias, unit-norm columns |
| Decoder norm | Enforced after every gradient step |

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | **10** (default is 50 — undertrained; see caveats) |
| Learning rate | 3e-4 (Adam) |
| Batch size | 256 |
| Loss | Reconstruction MSE only (no L1/sparsity penalty) |
| Device | CUDA |
| Dead feature resampling interval | every 5 epochs |
| Dead feature threshold | freq < 1e-4 across epoch |
| Resample strategy | Encoder row ← normalized high-reconstruction-loss token; decoder col ← same; biases reset to 0 |
| Resampling candidate pool | Top-25% highest-loss tokens from 8192-sample subset |

### Output Checkpoints

Saved to `checkpoints/saes/sae_{stage}_layer{N}.pt` as:
```python
{
    "state_dict": sae.state_dict(),
    "config": {"d_model": 896, "d_sae": 7168, "k": 32, "source": "<stage>_layer<N>"}
}
```

### SAE Files Produced

| File | Stage | Layer |
|------|-------|-------|
| `sae_sft_layer6.pt` | sft | 6 |
| `sae_sft_layer12.pt` | sft | 12 |
| `sae_sft_layer18.pt` | sft | 18 |
| `sae_sft_layer23.pt` | sft | 23 |
| `sae_ppo_step10_layer{6,12,18,23}.pt` | ppo_step10 | 6, 12, 18, 23 |
| `sae_ppo_step50_layer{6,12,18,23}.pt` | ppo_step50 | 6, 12, 18, 23 |
| `sae_ppo_step100_layer{6,12,18,23}.pt` | ppo_step100 | 6, 12, 18, 23 |
| `sae_ppo_step200_layer{6,12,18,23}.pt` | ppo_step200 | 6, 12, 18, 23 |
| `sae_ppo_step300_layer{6,12,18,23}.pt` | ppo_step300 | 6, 12, 18, 23 |
| `sae_ppo_step435_layer{6,12,18,23}.pt` | ppo_step435 | 6, 12, 18, 23 |

**Total: 28 SAEs**

---

## Stage 6 — Feature Analysis

**Script:** `06_analyze_features.py`

| Parameter | Value |
|-----------|-------|
| SAE dir | `checkpoints/saes/` |
| Activations dir | `data/activations/` |
| Output dir | `results/feature_analysis/` |
| Device | CPU |
| Dead feature threshold | freq < 0.01 |
| Stable feature threshold | best-match cosine similarity > 0.9 |
| Drifted feature threshold | best-match cosine similarity < 0.5 |

**Outputs generated:**
- `freq_layer{6,12,18,23}.png` — activation frequency histograms per stage
- `drift_{s1}_{s2}_layer{N}.png` — feature drift plots for all 6 consecutive transitions × 4 layers = 24 plots

---

## Stage 7 — SAE Evaluation Metrics

**Script:** `07_eval_sae_metrics.py`
**Output:** `results/sae_eval_metrics.csv`

| Parameter | Value |
|-----------|-------|
| Held-out split | Last 20% of cached activation tokens |
| Delta loss prompts | 200 GSM8k test questions |
| Delta loss batch size | 4 |
| Delta loss max length | 256 tokens |

**Metrics computed:**

| Metric | Description |
|--------|-------------|
| `recon_mse` | Mean squared reconstruction error on held-out tokens |
| `mean_l0` | Average active features per token (always 32 by TopK construction) |
| `model_delta_loss` | Change in cross-entropy when SAE reconstruction is spliced into residual stream at inference |

### Results (`results/sae_eval_metrics.csv`)

| stage | layer | recon_mse | mean_l0 | model_delta_loss |
|-------|-------|-----------|---------|-----------------|
| sft | 6 | 0.136993 | 32.00 | −1.333909 |
| sft | 12 | 0.407043 | 32.00 | −1.282245 |
| sft | 18 | 0.231628 | 32.00 | −0.070511 |
| sft | 23 | 0.804597 | 32.00 | −0.633108 |
| ppo_step10 | 6 | 0.975520 | 32.00 | −1.254492 |
| ppo_step10 | 12 | 0.274211 | 32.00 | −0.859535 |
| ppo_step10 | 18 | 0.319532 | 32.00 | +0.095954 |
| ppo_step10 | 23 | 0.801987 | 32.00 | −0.228977 |
| ppo_step50 | 6 | 1.723535 | 32.00 | −1.671710 |
| ppo_step50 | 12 | 0.081252 | 32.00 | −0.346445 |
| ppo_step50 | 18 | 0.380938 | 32.00 | −0.701579 |
| ppo_step50 | 23 | 0.835443 | 32.00 | −0.145933 |
| ppo_step100 | 6 | 0.050313 | 32.00 | −1.223047 |
| ppo_step100 | 12 | 1.356936 | 32.00 | −0.694438 |
| ppo_step100 | 18 | 0.531712 | 32.00 | −0.052751 |
| ppo_step100 | 23 | 0.883354 | 32.00 | −0.338555 |
| ppo_step200 | 6 | 1.399631 | 32.00 | −2.101737 |
| ppo_step200 | 12 | 0.825739 | 32.00 | −0.472696 |
| ppo_step200 | 18 | 0.327060 | 32.00 | +0.490148 |
| ppo_step200 | 23 | 0.892349 | 32.00 | +0.050010 |
| ppo_step300 | 6 | 0.068272 | 32.00 | −1.357092 |
| ppo_step300 | 12 | 1.686285 | 32.00 | −0.368631 |
| ppo_step300 | 18 | 0.437860 | 32.00 | +0.570341 |
| ppo_step300 | 23 | 0.925784 | 32.00 | −0.448021 |
| ppo_step435 | 6 | 0.056221 | 32.00 | −1.407711 |
| ppo_step435 | 12 | 0.167075 | 32.00 | −0.589400 |
| ppo_step435 | 18 | 0.427913 | 32.00 | +0.225214 |
| ppo_step435 | 23 | 0.913364 | 32.00 | −0.332837 |

> **Note on delta loss sign:** Negative values mean the SAE reconstruction *improves* model cross-entropy relative to baseline (likely a side-effect of the splicing mechanism and the undertrained SAEs). Positive values (layer 18 at steps 200–435) indicate meaningful information loss.

---

## Known Issues & Caveats

| Issue | Impact | Fix |
|-------|--------|-----|
| SAEs trained for 10 epochs (default 50) | High dead-feature rate (~89% at layer 6), inflated recon loss, noisy drift estimates | Rerun `05_train_sae.py --epochs 50` |
| PPO reward hacking at steps 200–435 | Response length collapsed to ~5–6 tokens; solve rate dropped from ~79% to ~12–13% | Use only SFT→step_100 window for clean interpretation |
| Duplicate `trainer.total_epochs` override in early PPO batch | 8 Mar 31 AM runs had effective epochs=5 instead of 15; all crashed anyway | Fixed in later runs |
| SFT v1 used `lora_alpha=16` (scaling=0.5) | Weaker LoRA updates vs v2 | SFT v2 used `lora_alpha=32` (scaling=1.0) |
| Delta loss metric signs anomalous | Undertrained SAEs produce atypical splice behavior | Recompute after 50-epoch SAE retraining |

---

## Checkpoint Inventory

```
checkpoints/
├── sft/                           # SFT v1 LoRA checkpoint (3 epochs)
│   └── sae_rl_gsm8k/sft_qwen2.5_0.5b/global_step_87/
├── sft_merged/                    # SFT v1 merged flat model
├── sft_v2/                        # SFT v2 LoRA checkpoint (5 epochs, alpha=32)
├── sft_v2_merged/                 # SFT v2 merged flat model
├── sae_rl_gsm8k/
│   └── ppo_qwen2.5_0.5b/         # PPO run (15 epochs, n=1 rollout)
│       ├── global_step_5/
│       ├── global_step_10/
│       ├── global_step_15/
│       ├── global_step_20/
│       ├── global_step_25/
│       ├── global_step_30/
│       └── ...  (every 5 steps up to ~435)
├── ppo_merged/                    # Merged HF models for SAE collection
│   ├── step_10/
│   ├── step_50/
│   ├── step_100/
│   ├── step_200/
│   ├── step_300/
│   └── step_435/
└── saes/                          # 28 trained SAEs
    ├── sae_sft_layer{6,12,18,23}.pt
    ├── sae_ppo_step10_layer{6,12,18,23}.pt
    ├── ...
    └── sae_ppo_step435_layer{6,12,18,23}.pt
```
