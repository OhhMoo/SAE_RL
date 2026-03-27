#!/bin/bash
# Step 3: PPO Training on top of the SFT checkpoint
#
# Uses verl's PPO trainer with GSM8k rule-based reward (exact match on #### answer).
# The actor is initialized from the SFT checkpoint; the critic from the base model.
#
# Usage:
#   bash scripts/03_ppo_qwen.sh [extra_configs...]
#
# Important: Set ACTOR_MODEL_PATH below to your SFT checkpoint before running.
# After SFT with LoRA, you need the merged model path. See 02b_merge_lora.py.

set -x

# ----- Configure these paths -----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_DIR}/data/gsm8k"

# Point this to your merged SFT model (or base model if skipping SFT)
ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
CRITIC_MODEL_PATH="${CRITIC_MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
NUM_GPUS="${NUM_GPUS:-2}"
# ----------------------------------

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=False \
    critic.model.path=$CRITIC_MODEL_PATH \
    +critic.model.override_config.attn_implementation=sdpa \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=8 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.kl_coef=0.02 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='sae_rl_gsm8k' \
    trainer.experiment_name='ppo_qwen2.5_0.5b' \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    "$@"
