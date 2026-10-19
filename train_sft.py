"""
Part A -- LoRA SFT training entrypoint.

SFT warm-starts the target behavior: it fine-tunes the instruct base model on the
PREFERRED response of each preference pair you built (a clarifying question for an
ambiguous prompt, a direct answer for a clear one), so that DPO in Part B has a sensible
policy to sharpen rather than starting cold.

CLI args, tokenizer setup, base model + LoRA construction, and checkpointing are wired
up already. You implement utils/masking_utils.tokenize_with_response_mask() and
train_loop() -- see the TODO blocks and README.md's Part A for two gotchas: freezing
base weights, masking prompt tokens.
"""

import argparse

import torch
from peft import LoraConfig, get_peft_model

from configs.paths_config import SFT_CHECKPOINT_DIR
from configs.train_config import (
    MODEL_ID,
    RANK,
    ALPHA,
    DROPOUT,
    TARGET_MODULES,
    SFT_BATCH_SIZE,
    SFT_EPOCHS,
    SFT_LEARNING_RATE,
    SFT_MAX_SEQ_LEN,
)
from data_loader import load_sft_examples
from utils.model_utils import load_tokenizer, load_base_model
from utils.masking_utils import tokenize_with_response_mask


def build_model_and_tokenizer():
    """Loads the base model + tokenizer, and wraps the model with a LoRA adapter."""
    tokenizer = load_tokenizer(MODEL_ID)
    model = load_base_model()

    lora_config = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        lora_dropout=DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Sanity check: should be a small fraction of total params.

    return model, tokenizer


def build_sft_example(messages: list[dict], tokenizer, max_seq_len: int = SFT_MAX_SEQ_LEN) -> dict:
    """
    Wired up: converts utils.masking_utils.tokenize_with_response_mask()'s response_mask into
    the -100-masked labels tensor the causal-LM loss needs. Once you implement
    tokenize_with_response_mask(), this works for free -- nothing to implement here.

    Returns a dict with "input_ids", "attention_mask", "labels": labels equal
    input_ids where response_mask == 1 (the final assistant turn), -100 elsewhere.
    """
    example = tokenize_with_response_mask(messages, tokenizer, max_seq_len)

    labels = example["input_ids"].clone()
    labels[example["response_mask"] == 0] = -100

    return {
        "input_ids": example["input_ids"],
        "attention_mask": example["attention_mask"],
        "labels": labels,
    }


def train_loop(model, tokenizer, dataset) -> None:
    """
    Implements the Part A SFT training loop.

    TODO (student):
      1. Preprocess `dataset` ONCE, up front, by calling build_sft_example() per example
         (e.g. via `dataset.map(...)`) to get input_ids/attention_mask/labels -- do not
         call build_sft_example() inside the per-step loop below, or you'll re-tokenize
         the same examples on every epoch for no benefit.
      2. For each batch of preprocessed examples: pad/collate into a batch, forward pass,
         compute loss, backward, optimizer step. Only LoRA parameters should have
         gradients -- true by default from build_model_and_tokenizer() above, but verify
         it if you change the LoRA setup (README.md Part A, gotcha #1).
      3. Log the loss periodically so you can sanity-check training is converging.

    Hyperparameters (SFT_BATCH_SIZE, SFT_EPOCHS, SFT_LEARNING_RATE, SFT_MAX_SEQ_LEN) are
    in configs/train_config.py -- yours to tune, not fixed by this skeleton.
    """
    print("Preprocessing dataset...")
    # dataset.map applies your function to every item in the dataset efficiently
    # Assuming you import _to_messages from data_loader
    processed_dataset = dataset.map(
            lambda example: build_sft_example(
                example["messages"],  # <-- Extract the list from the row dictionary here
                tokenizer, 
                SFT_MAX_SEQ_LEN
            ),
            remove_columns=dataset.column_names, 
            batched=False 
        )

    def collate_fn(batch):
        """Pads examples in the batch to the same length."""
        # Extract individual lists/tensors
        input_ids = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
        attention_mask = [torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in batch]
        labels = [torch.tensor(ex["labels"], dtype=torch.long) for ex in batch]

        # Pad sequences to the maximum length in this specific batch
        # pad_token_id is usually used for inputs
        padded_inputs = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        # Attention mask padding must be 0 so the model ignores the padded space
        padded_attention = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        # Labels padding MUST be -100 because PyTorch CrossEntropyLoss ignores -100
        padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        # Move tensors to the same device as the model (e.g., "cuda")
        return {
            "input_ids": padded_inputs.to(model.device),
            "attention_mask": padded_attention.to(model.device),
            "labels": padded_labels.to(model.device)
        }

    dataloader = torch.utils.data.DataLoader(
        processed_dataset, 
        batch_size=SFT_BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    #train lora
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=SFT_LEARNING_RATE)

    # -------------------------------------------------------------------------
    # 3. The actual Training Loop
    # -------------------------------------------------------------------------
    print("Starting training...")
    model.train() # Set model to training mode

    for epoch in range(SFT_EPOCHS):
        for step, batch in enumerate(dataloader):
            # Clear old gradients
            optimizer.zero_grad()

            # Forward pass. HuggingFace models automatically calculate loss if 'labels' are provided
            outputs = model(**batch)
            loss = outputs.loss

            # Backward pass (compute gradients)
            loss.backward()

            # Update weights
            optimizer.step()

            # Log periodically
            if step % 10 == 0:
                print(f"Epoch {epoch + 1}/{SFT_EPOCHS} | Step {step} | Loss: {loss.item():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default=SFT_CHECKPOINT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"Loading base model {MODEL_ID} and building LoRA adapter...")
    model, tokenizer = build_model_and_tokenizer()

    print("Loading SFT examples (preferred responses from your preference data)...")
    dataset = load_sft_examples()

    print("Training starts...")
    train_loop(model, tokenizer, dataset)

    print(f"Saving LoRA adapter to {args.checkpoint_dir}...")
    model.save_pretrained(args.checkpoint_dir)
    tokenizer.save_pretrained(args.checkpoint_dir)
    print("Part A SFT complete.")


if __name__ == "__main__":
    main()
