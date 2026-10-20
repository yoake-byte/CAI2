"""
Generate before/after transcripts between SFT and DPO checkpoints
for the final report.
"""
import torch
from configs.paths_config import SFT_CHECKPOINT_DIR, DPO_CHECKPOINT_BASE
from evaluate import _free, generate_responses, load_adapter_policy
from utils.model_utils import load_tokenizer

# 1. Pick 3 representative prompts
# - Transcript 1: Ambiguous prompt (SFT guesses vs. DPO asks)
# - Transcript 2: Clear factual prompt (both answer directly)
# - Transcript 3: Subtle/edge-case prompt (demonstrating behavior shift or failure)
PROMPTS = [
    # Transcript 1: Ambiguous
    "Can you help me fix the code?",
    # Transcript 2: Clear
    "What is the capital of France and what language is spoken there?",
    # Transcript 3: Subtle / Edge Case
    "How do I install the package on my system?",
]

# Set the winning beta you want to showcase
BEST_BETA = 0.5


def run():
    tokenizer = load_tokenizer(SFT_CHECKPOINT_DIR)

    # --- Generate SFT ("Before") ---
    print(f"Loading SFT model from {SFT_CHECKPOINT_DIR}...")
    sft_model = load_adapter_policy(SFT_CHECKPOINT_DIR)
    sft_responses = generate_responses(sft_model, tokenizer, PROMPTS)
    _free(sft_model)

    # --- Generate DPO ("After") ---
    dpo_dir = f"{DPO_CHECKPOINT_BASE}/beta_{BEST_BETA}"
    print(f"Loading DPO model (beta={BEST_BETA}) from {dpo_dir}...")
    dpo_model = load_adapter_policy(dpo_dir)
    dpo_responses = generate_responses(dpo_model, tokenizer, PROMPTS)
    _free(dpo_model)

    # --- Display Side-by-Side Transcripts ---
    print("\n" + "=" * 80)
    print("QUALITATIVE TRANSCRIPTS FOR REPORT")
    print("=" * 80 + "\n")

    for i, (prompt, sft_out, dpo_out) in enumerate(
        zip(PROMPTS, sft_responses, dpo_responses), 1
    ):
        print(f"### Transcript {i}")
        print(f"**Prompt:**\n{prompt}\n")
        print(f"**SFT Output (Before):**\n{sft_out}\n")
        print(f"**DPO Output (After, beta={BEST_BETA}):**\n{dpo_out}\n")
        print("-" * 80 + "\n")


if __name__ == "__main__":
    run()