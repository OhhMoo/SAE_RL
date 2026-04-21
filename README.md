# Studying RL with SAEs
RL fine-tuning demonstrably changes model behavior, but how it changes internal representations is poorly understood. We designed a controlled experiment — fixed dataset (GSM8k), fixed model (Qwen2.5-0.5B), fixed SAE architecture (BatchTopK) — so that any differences in learned features across checkpoints reflect genuine representational change, not artifacts of the setup.


## References

- [verl documentation](https://verl.readthedocs.io)
- [SAELens (decoderesearch)](https://github.com/decoderesearch/SAELens)
- [Qwen2.5 model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- See `research_plan.md` for full design rationale, hypotheses, and references to the SAE regularization paper.
