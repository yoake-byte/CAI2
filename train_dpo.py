"""
Part B -- DPO training entrypoint.

DPO sharpens the Part A policy on your preference pairs: push up the log-prob of the
preferred response (ask when ambiguous / answer when clear) relative to the dispreferred
one, under a KL leash to a frozen reference (your Part A checkpoint). You must sweep beta
over at least three values -- run this script once per beta (`--beta`), each writing its
own adapter.

CLI args, loading your Part A LoRA checkpoint as the policy's starting point, and
per-beta checkpointing are wired up already. You implement sequence_log_prob(),
dpo_loss(), and the training loop around them -- see the TODO blocks and README.md's
Part B for the three things to know going in (the disable_adapter() trick, sequence
log-prob masking, the loss formula). build_dpo_example() itself is wired, built on top of
utils/masking_utils.tokenize_with_response_mask() -- see Part A for where that TODO lives.
"""

import argparse

import torch
from peft import PeftModel

from configs.paths_config import SFT_CHECKPOINT_DIR, dpo_checkpoint_dir
from configs.train_config import (
    DPO_BATCH_SIZE,
    DPO_EPOCHS,
    DPO_LEARNING_RATE,
    DPO_BETA,
    DPO_MAX_SEQ_LEN,
)
from data_loader import load_dpo_examples
from utils.model_utils import load_tokenizer, load_base_model
from utils.masking_utils import tokenize_with_response_mask


def build_policy_and_tokenizer(sft_checkpoint_dir: str):
    """
    Loads the base model and attaches your Part A LoRA adapter as the DPO policy's
    starting point. The *same* model, with its adapter disabled via
    `policy.disable_adapter()`, doubles as the frozen reference model -- see
    README.md Part B, gotcha #1. You do not need to load a second copy of the base model.
    """
    tokenizer = load_tokenizer(sft_checkpoint_dir)
    base_model = load_base_model()
    policy = PeftModel.from_pretrained(base_model, sft_checkpoint_dir, is_trainable=True)

    return policy, tokenizer


def build_dpo_example(
    prompt: str,
    chosen_messages: list[dict],
    rejected_messages: list[dict],
    tokenizer,
    max_seq_len: int = DPO_MAX_SEQ_LEN,
) -> dict:
    """
    Wired up: calls utils.masking_utils.tokenize_with_response_mask() once for
    chosen_messages and once for rejected_messages. Once you implement that one
    function (see Part A), this works for free -- nothing to implement here.

    `chosen_messages`/`rejected_messages` are each a list of {"role", "content"} dicts:
    the shared user prompt followed by the differing final assistant completion -- see
    data_loader.py. `prompt` itself isn't needed here (it's already the shared prefix
    inside both messages lists) but is accepted for a uniform call signature with
    data_loader's (prompt, chosen, rejected) triples.

    Returns a dict with, for both "chosen" and "rejected":
      - "{chosen,rejected}_input_ids", "{chosen,rejected}_attention_mask",
        "{chosen,rejected}_response_mask" -- see
        utils.masking_utils.tokenize_with_response_mask()'s return value.
    """
    chosen = tokenize_with_response_mask(chosen_messages, tokenizer, max_seq_len)
    rejected = tokenize_with_response_mask(rejected_messages, tokenizer, max_seq_len)

    return {
        "chosen_input_ids": chosen["input_ids"],
        "chosen_attention_mask": chosen["attention_mask"],
        "chosen_response_mask": chosen["response_mask"],
        "rejected_input_ids": rejected["input_ids"],
        "rejected_attention_mask": rejected["attention_mask"],
        "rejected_response_mask": rejected["response_mask"],
    }


def sequence_log_prob(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor, response_mask: torch.Tensor
) -> torch.Tensor:
    """
    TODO (student): computes, for each sequence in the batch, the sum of log-probs of
    the response tokens only (response_mask == 1), under `model`'s current weights.
    Returns a 1D tensor of shape (batch_size,).

    Call this once with the policy active, and once more inside a
    `with model.disable_adapter():` block to get the reference log-probs -- see
    README.md Part B, gotcha #1.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    predictions = logits[:, :-1, :]  # Shift for next-token prediction
    target_ids = input_ids[:, 1:]  # Shift for next-token prediction
    masks = attention_mask[:, 1:] * response_mask[:, 1:]  # Shift for next-token prediction

    log_probs = torch.nn.functional.log_softmax(predictions, dim=-1)

    # Step 4: Gather the log-probs of the actual target tokens
    selected_log_probs = log_probs.gather(dim=2, index=target_ids.unsqueeze(-1)).squeeze(-1)
    response_log_probs = selected_log_probs * masks
    sum_log_probs = response_log_probs.sum(dim=1)

    return sum_log_probs


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = DPO_BETA,
) -> torch.Tensor:
    """
    TODO (student): implements the DPO loss --

        L_DPO = -log sigma( beta * [ (policy_chosen_logps - reference_chosen_logps)
                                      - (policy_rejected_logps - reference_rejected_logps) ] )

    averaged over the batch.

    This function's signature must not change. The GTA reference tests exercise it
    directly (tests/test_dpo_loss.py).
    """
    chosen_diff = policy_chosen_logps - reference_chosen_logps
    rejected_diff = policy_rejected_logps - reference_rejected_logps
    loss = -torch.nn.functional.logsigmoid(beta * (chosen_diff - rejected_diff))
    return loss.mean()


import torch
from torch.nn.utils.rnn import pad_sequence

def dpo_collate_fn(batch_list, pad_token_id):
    # 'batch_list' is a list of dictionaries, where each dictionary is one row.
    # We need to group them by key instead of by row.
    
    batched_dict = {}
    
    # Assuming your build_dpo_example outputs these 6 standard keys:
    keys = [
        "chosen_input_ids", "chosen_attention_mask", "chosen_response_mask",
        "rejected_input_ids", "rejected_attention_mask", "rejected_response_mask"
    ]
    
    for key in keys:
        # Extract the tensor for this specific key from every row in the batch
        # We also ensure they are actual PyTorch tensors
        tensors = [torch.tensor(row[key]) for row in batch_list]
        
        # Determine the correct padding value
        if "input_ids" in key:
            pad_val = pad_token_id
        else:
            pad_val = 0  # Masks (attention and response) are padded with 0
            
        # pad_sequence stacks them into a single tensor and pads the shorter ones
        batched_dict[key] = pad_sequence(tensors, batch_first=True, padding_value=pad_val)
        
    return batched_dict

def train_loop(policy, tokenizer, dataset, beta: float = DPO_BETA) -> None:
    """
    Implements the Part B DPO training loop for a single beta.

    TODO (student):
      1. Preprocess `dataset` ONCE, up front, by calling build_dpo_example() per
         (prompt, chosen, rejected) triple (e.g. via `dataset.map(...)`) -- do not call
         it inside the per-step loop below, or you'll re-tokenize every example on every
         epoch for no benefit.
      2. For each batch of preprocessed examples: pad/collate, then compute
         policy_chosen_logps / policy_rejected_logps via sequence_log_prob() with the
         adapter enabled.
      3. Compute reference_chosen_logps / reference_rejected_logps via the same
         sequence_log_prob() call, inside `with policy.disable_adapter():`.
      4. Compute dpo_loss(..., beta=beta), backward, optimizer step.
      5. Log the loss periodically.

    Hyperparameters (DPO_BATCH_SIZE, DPO_EPOCHS, DPO_LEARNING_RATE, DPO_MAX_SEQ_LEN) are
    in configs/train_config.py -- yours to tune. `beta` is passed in per run (--beta) so
    the sweep trains a separate adapter for each value.
    """
    print("Preprocessing dataset...")
    # dataset.map applies your function to every item in the dataset efficiently
    # Assuming you import _to_messages from data_loader
    def preprocess_fn(row):
        # Extract the raw strings from the dataset row
        prompt = row["prompt"]
        chosen = row["chosen"]
        rejected = row["rejected"]
        
        # Pass them to the provided tokenization function
        # Note: Pass the tokenizer and max sequence length if the function requires them
        tokenized_example = build_dpo_example(
            tokenizer=tokenizer,
            prompt=prompt,
            chosen_messages=chosen,
            rejected_messages=rejected,
            #max_seq_len=DPO_MAX_SEQ_LEN  # Uncomment if your function takes this argument
        )
        
        # Return the dictionary of tokenized tensors 
        # (e.g., chosen_input_ids, rejected_input_ids, masks)
        return tokenized_example
    # Assuming your dataset has columns like ['prompt', 'chosen', 'rejected']
    original_columns = dataset.column_names

    processed_dataset = dataset.map(
        preprocess_fn,
        batched=False,               # Process one row at a time
        remove_columns=original_columns # CRITICAL STEP
    )

    # Ensure your tokenizer actually has a pad token set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Initialize the DataLoader
    dataloader = torch.utils.data.DataLoader(
        processed_dataset,
        batch_size=DPO_BATCH_SIZE,  # From your train_config
        shuffle=True,               # Shuffle the training data
        collate_fn=lambda batch: dpo_collate_fn(batch, pad_token_id=tokenizer.pad_token_id)
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=DPO_LEARNING_RATE)

    policy.train() # Set model to training mode
    
    for epoch in range(DPO_EPOCHS):
        for step, batch in enumerate(dataloader):
            # 1. Move everything to the GPU
            chosen_input_ids = batch["chosen_input_ids"].to(policy.device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(policy.device)
            chosen_response_mask = batch["chosen_response_mask"].to(policy.device)
            rejected_input_ids = batch["rejected_input_ids"].to(policy.device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(policy.device)  
            rejected_response_mask = batch["rejected_response_mask"].to(policy.device)  

            policy_chosen_logps = sequence_log_prob(policy, chosen_input_ids, chosen_attention_mask, chosen_response_mask)
            policy_rejected_logps = sequence_log_prob(policy, rejected_input_ids, rejected_attention_mask, rejected_response_mask)


            with torch.no_grad():
                with policy.disable_adapter():
                    ref_chosen_logps = sequence_log_prob(
                        model=policy, 
                        input_ids=chosen_input_ids, 
                        attention_mask=chosen_attention_mask, 
                        response_mask=chosen_response_mask
                    )
                    ref_rejected_logps = sequence_log_prob(
                        model=policy, 
                        input_ids=rejected_input_ids, 
                        attention_mask=rejected_attention_mask, 
                        response_mask=rejected_response_mask
                    )
            dpo_loss_value = dpo_loss(
                policy_chosen_logps=policy_chosen_logps,
                policy_rejected_logps=policy_rejected_logps,
                reference_chosen_logps=ref_chosen_logps,
                reference_rejected_logps=ref_rejected_logps,
                beta=beta
            )
            optimizer.zero_grad()
            dpo_loss_value.backward()
            optimizer.step()
            if step % 10 == 0:  # Print every 10 steps
                print(f"Epoch {epoch} | Step {step} | DPO Loss: {dpo_loss_value.item():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-checkpoint-dir", type=str, default=SFT_CHECKPOINT_DIR)

    parser.add_argument(
        "--beta",
        type=float,
        default=DPO_BETA,
        help="DPO beta for this run (sweep by running once per value).",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override; defaults to a per-beta directory.",
    )

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir or dpo_checkpoint_dir(args.beta)

    torch.manual_seed(args.seed)

    print(f"Loading Part A checkpoint from {args.sft_checkpoint_dir} as DPO policy init...")
    policy, tokenizer = build_policy_and_tokenizer(args.sft_checkpoint_dir)

    print("Loading DPO preference pairs...")
    dataset = load_dpo_examples()

    print(f"Training starts (beta={args.beta})...")
    train_loop(policy, tokenizer, dataset, beta=args.beta)

    print(f"Saving DPO-aligned LoRA adapter to {checkpoint_dir}...")
    policy.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print(f"Part B DPO complete (beta={args.beta}).")


if __name__ == "__main__":
    main()
