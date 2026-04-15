#!/usr/bin/env bash
# 09_dense_pipeline.sh
# Merge PPO checkpoints around the reward-hacking onset window (steps 100–200),
# collect activations, and train SAEs — all to a separate "dense" directory.
#
# This feeds into 10_classify_features.py for temporal precedence analysis.
#
# Usage:
#   conda activate sae_rl
#   bash scripts/09_dense_pipeline.sh
#   bash scripts/09_dense_pipeline.sh --k 64 --epochs 10
#
# Outputs:
#   checkpoints/ppo_merged/step_{110..195}/     (merged models)
#   data/activations/ppo_step{110..195}_layer*.pt
#   checkpoints/saes_dense/sae_ppo_step*_layer*.pt

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PPO_CKPT_BASE="$ROOT/checkpoints/sae_rl_gsm8k/ppo_qwen2.5_0.5b"
MERGED_DIR="$ROOT/checkpoints/ppo_merged"
ACTIVATIONS_DIR="$ROOT/data/activations"
DENSE_ACT_DIR="$ROOT/data/activations_dense_tmp"
SAES_DIR="$ROOT/checkpoints/saes_dense"
LAYERS="6 12 18 23"
K=64
EPOCHS=10

# Dense steps every 10 steps between 100 and 200 (exclusive)
DENSE_STEPS=(110 120 130 140 150 160 170 180 190)

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --k)      K="$2";      shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        *) echo "[error] Unknown argument: $1"; exit 1 ;;
    esac
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Dense Checkpoint Pipeline"
echo "  Steps : ${DENSE_STEPS[*]}"
echo "  k     : $K"
echo "  Epochs: $EPOCHS"
echo "  Output: $SAES_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mkdir -p "$MERGED_DIR" "$ACTIVATIONS_DIR" "$SAES_DIR" "$DENSE_ACT_DIR"

# ── Step 1: Merge PPO checkpoints ─────────────────────────────────────────────
echo ""
echo "▶  Step 1/3: Merging PPO checkpoints"

for STEP in "${DENSE_STEPS[@]}"; do
    MERGED_PATH="$MERGED_DIR/step_$STEP"

    if [ -d "$MERGED_PATH" ] && [ "$(ls -A "$MERGED_PATH" 2>/dev/null)" ]; then
        echo "  [skip] step_$STEP — already merged"
        continue
    fi

    ACTOR_PATH="$PPO_CKPT_BASE/global_step_${STEP}/actor"
    if [ ! -d "$ACTOR_PATH" ]; then
        echo "  [warn] Checkpoint not found: $ACTOR_PATH — skipping"
        continue
    fi

    echo "  Merging step_$STEP → $MERGED_PATH"
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir  "$ACTOR_PATH" \
        --target_dir "$MERGED_PATH"
done

# ── Step 2: Collect activations ───────────────────────────────────────────────
echo ""
echo "▶  Step 2/3: Collecting activations"

for STEP in "${DENSE_STEPS[@]}"; do
    MERGED_PATH="$MERGED_DIR/step_$STEP"
    NAME="ppo_step${STEP}"

    if [ ! -d "$MERGED_PATH" ]; then
        echo "  [skip] $NAME — merged model missing"
        continue
    fi

    # Check if all layer files exist
    MISSING=false
    for L in $LAYERS; do
        [ ! -f "$ACTIVATIONS_DIR/${NAME}_layer${L}.pt" ] && MISSING=true && break
    done

    if [ "$MISSING" = false ]; then
        echo "  [skip] $NAME — activations already exist"
        continue
    fi

    echo "  Collecting: $NAME"
    python scripts/04_collect_activations.py \
        --model_path      "$MERGED_PATH" \
        --checkpoint_name "$NAME" \
        --layers          $LAYERS \
        --save_dir        "$ACTIVATIONS_DIR" \
        --batch_size      16 \
        --max_tokens      500000
done

# ── Step 3: Symlink dense activations → temp dir, train SAEs ──────────────────
echo ""
echo "▶  Step 3/3: Training SAEs on dense checkpoints"

# Clear and repopulate temp symlink dir (only dense steps)
rm -f "$DENSE_ACT_DIR"/*.pt
LINKED=0
for STEP in "${DENSE_STEPS[@]}"; do
    NAME="ppo_step${STEP}"
    for L in $LAYERS; do
        SRC="$ACTIVATIONS_DIR/${NAME}_layer${L}.pt"
        DST="$DENSE_ACT_DIR/${NAME}_layer${L}.pt"
        if [ -f "$SRC" ] && [ ! -L "$DST" ]; then
            ln -s "$SRC" "$DST"
            LINKED=$((LINKED + 1))
        fi
    done
done
echo "  Linked $LINKED activation files into $DENSE_ACT_DIR"

python scripts/05_train_sae.py \
    --activations_dir "$DENSE_ACT_DIR" \
    --save_dir        "$SAES_DIR" \
    --expansion_factor 8 \
    --k               "$K" \
    --epochs          "$EPOCHS" \
    --lr              3e-4 \
    --batch_size      256 \
    --device          cuda

# Clean up temp dir
rm -rf "$DENSE_ACT_DIR"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Done."
echo "  SAEs → $SAES_DIR"
echo "  Next: python scripts/10_classify_features.py"
