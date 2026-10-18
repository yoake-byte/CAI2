"""
Part 1 -- build your preference data.

You construct at least MIN_PREFERENCE_PAIRS (1,000) preference pairs over two input
classes:

  - ambiguous: a request that is genuinely underspecified. The PREFERRED (chosen)
    response asks a specific clarifying question; the DISPREFERRED (rejected) response
    is a confident guess that commits to one interpretation.

  - clear: a fully-specified request. The PREFERRED (chosen) response answers directly;
    the DISPREFERRED (rejected) response is a needless clarifying question.

You MAY synthesize these with an LLM. The course model (Llama-3.1-8B-Instruct) runs on
PACE-ICE at no cost -- load it locally the same way eval/judge.py's LocalHFJudge does.
You MUST hand-verify a sample of HAND_VERIFY_SAMPLE (100) pairs and report the
rejection rate of your own synthetic data (spec, Part 1 + the "preference data quality" rubric line).

What is wired for you: the output schema + writer, the verification sampler, and the
rejection-rate computation. What you implement: `synthesize_pairs()` -- the actual data generation.

INTEGRITY: build your own prompts. Do NOT reuse the held-out eval prompts in
eval/held_out_*.jsonl as training data -- that is training on the test set.

Usage:
    python3 generate_preference_data.py --generate   # Synthesize + write pairs, then
                                                      # sample 100 for hand-verification.

    # ... now open data/hand_verified.jsonl and fill in each "verdict" ...
    python3 generate_preference_data.py --report      # Compute + print the rejection rate.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from configs.train_config import (
    DATA_SEED,
    HAND_VERIFY_SAMPLE,
    MIN_PREFERENCE_PAIRS,
)
from configs.paths_config import HAND_VERIFIED_PATH, PREFERENCE_DATA_PATH

VALID_CLASSES = {"ambiguous", "clear"}
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
def synthesize_pairs(n: int) -> list[dict]:
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    """
    TODO (student): produce `n` preference pairs, balanced across the two classes.

    Return a list of dicts, each:
        {"class": "ambiguous" | "clear",
         "prompt": "<user request>",
         "chosen": "<preferred response>",
         "rejected": "<dispreferred response>"}

    (The "id" field is added for you by write_pairs().)

    Approach is up to you. A common recipe: prompt an LLM to generate ambiguous and
    clear requests across varied domains, then for each request generate both a
    clarifying-question response and a direct-answer response, and assign chosen/rejected
    by the request's class. Keep DATA_SEED fixed wherever you sample, so your data is reproducible.
    """
    # Calculate target counts
    n_ambiguous = n // 2
    n_clear = n - n_ambiguous
    pairs = []
    
    # 2. Define a helper function for generation
    def generate_single_scenario(system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        #model generate
        with torch.no_grad():
            out = model.generate(
                **inputs, 
                max_new_tokens=256, 
                temperature=0.8, # Temperature > 0 ensures diversity across 1000 samples
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        gen = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(gen, skip_special_tokens=True)

    base_system = """You are a strict data formatting AI. You must generate exactly ONE scenario.
    DO NOT output any conversational text like "Here is your scenario."
    You MUST use these exact exact tags: [PROMPT], [CHOSEN], [REJECTED].

    Example format:
    [PROMPT] Can you fix the bug?
    [CHOSEN] I'd be happy to help. Could you please provide the code you are working with and the error message you are receiving?
    [REJECTED] The bug is on line 42. You need to change the variable to an integer."""

    print("Generating ambiguous pairs...")
    # 2. Use a while loop to ensure we hit the target count
    while len(pairs) < n_ambiguous:
        user_prompt = "Generate an AMBIGUOUS user request. The [CHOSEN] response must ask a clarifying question. The [REJECTED] response must be a confident guess."
        raw_text = generate_single_scenario(base_system, user_prompt)
        
        # 3. Use Regex to safely extract the text between tags
        try:
            # DOTALL allows the regex to match across multiple lines
            prompt_match = re.search(r'\[PROMPT\](.*?)\[CHOSEN\]', raw_text, re.DOTALL)
            chosen_match = re.search(r'\[CHOSEN\](.*?)\[REJECTED\]', raw_text, re.DOTALL)
            rejected_match = re.search(r'\[REJECTED\](.*)', raw_text, re.DOTALL)
            
            if prompt_match and chosen_match and rejected_match:
                pairs.append({
                    "class": "ambiguous",
                    "prompt": prompt_match.group(1).strip(),
                    "chosen": chosen_match.group(1).strip(),
                    "rejected": rejected_match.group(1).strip()
                })
            else:
                print("Tags missing in generation, retrying...")
        except Exception as e:
            print(f"Parsing error, retrying... ({e})")

    print("Generating clear pairs...")
    clear_pairs_count = 0
    while clear_pairs_count < n_clear:
        user_prompt = "Generate a CLEAR and specific user request. The [CHOSEN] response must answer it directly. The [REJECTED] response must ask a needless clarifying question."
        raw_text = generate_single_scenario(base_system, user_prompt)
        
        try:
            prompt_match = re.search(r'\[PROMPT\](.*?)\[CHOSEN\]', raw_text, re.DOTALL)
            chosen_match = re.search(r'\[CHOSEN\](.*?)\[REJECTED\]', raw_text, re.DOTALL)
            rejected_match = re.search(r'\[REJECTED\](.*)', raw_text, re.DOTALL)
            
            if prompt_match and chosen_match and rejected_match:
                pairs.append({
                    "class": "clear",
                    "prompt": prompt_match.group(1).strip(),
                    "chosen": chosen_match.group(1).strip(),
                    "rejected": rejected_match.group(1).strip()
                })
                clear_pairs_count += 1
            else:
                print("Tags missing in generation, retrying...")
        except Exception as e:
            print(f"Parsing error, retrying... ({e})")

    return pairs

    # # 3. Define your strict formatting prompts
    # base_system = """You are an AI data generator. Generate exactly ONE scenario. 
    # You MUST format your response exactly like this, with no extra text:
    # [PROMPT] <the user's request>
    # [CHOSEN] <the preferred response>
    # [REJECTED] <the dispreferred response>"""

    # # 4. Generation Loop for Ambiguous
    # print("Generating ambiguous pairs...")
    # for _ in tqdm(range(n_ambiguous)):
    #     # Provide domain ideas so all 500 aren't "fix my code"
    #     user_prompt = "Generate an AMBIGUOUS user request. The [CHOSEN] response must ask a clarifying question. The [REJECTED] response must be a confident guess."
        
    #     raw_text = generate_single_scenario(base_system, user_prompt)
        
    #     # Parse the delimiters
    #     try:
    #         prompt = raw_text.split("[PROMPT]")[1].split("[CHOSEN]")[0].strip()
    #         chosen = raw_text.split("[CHOSEN]")[1].split("[REJECTED]")[0].strip()
    #         rejected = raw_text.split("[REJECTED]")[1].strip()
            
    #         pairs.append({
    #             "class": "ambiguous",
    #             "prompt": prompt,
    #             "chosen": chosen,
    #             "rejected": rejected
    #         })
    #     except IndexError:
    #         # If the model fails to follow the format, skip and you might need a retry mechanism
    #         print("Failed to parse a generation, skipping...")

    # # 5. Generation Loop for Clear
    # print("Generating clear pairs...")
    # for _ in tqdm(range(n_clear)):
    #     user_prompt = "Generate a CLEAR and specific user request. The [CHOSEN] response must answer it directly. The [REJECTED] response must ask a needless clarifying question."
        
    #     raw_text = generate_single_scenario(base_system, user_prompt)
        
    #     try:
    #         prompt = raw_text.split("[PROMPT]")[1].split("[CHOSEN]")[0].strip()
    #         chosen = raw_text.split("[CHOSEN]")[1].split("[REJECTED]")[0].strip()
    #         rejected = raw_text.split("[REJECTED]")[1].strip()
            
    #         pairs.append({
    #             "class": "clear",
    #             "prompt": prompt,
    #             "chosen": chosen,
    #             "rejected": rejected
    #         })
    #     except IndexError:
    #         pass

    # return pairs


# --- Wired: everything below is provided -------------------------------------------


def write_pairs(pairs: list[dict], path: str | Path = PREFERENCE_DATA_PATH) -> None:
    """Validates, assigns stable ids, and writes pairs as JSONL."""
    required = {"class", "prompt", "chosen", "rejected"}

    for i, p in enumerate(pairs):
        missing = required - p.keys()

        if missing:
            raise ValueError(f"pair #{i} missing keys {sorted(missing)}")

        if p["class"] not in VALID_CLASSES:
            raise ValueError(f"pair #{i} has invalid class {p['class']!r}")

    if len(pairs) < MIN_PREFERENCE_PAIRS:
        raise ValueError(
            f"Only {len(pairs)} pairs; the assignment requires at least " f"{MIN_PREFERENCE_PAIRS}."
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for i, p in enumerate(pairs):
            rec = {
                "id": f"pair_{i:06d}",
                "class": p["class"],
                "prompt": p["prompt"],
                "chosen": p["chosen"],
                "rejected": p["rejected"],
            }

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_amb = sum(1 for p in pairs if p["class"] == "ambiguous")
    n_clr = sum(1 for p in pairs if p["class"] == "clear")
    print(f"Wrote {len(pairs)} pairs ({n_amb} ambiguous, {n_clr} clear) -> {path}")


def sample_for_verification(
    pairs_path: str | Path = PREFERENCE_DATA_PATH,
    out_path: str | Path = HAND_VERIFIED_PATH,
    k: int = HAND_VERIFY_SAMPLE,
    seed: int = DATA_SEED,
) -> None:
    """Samples k pairs into a hand-verification file with a blank 'verdict' field.

    Open the file and set each pair's "verdict" to "accept" or "reject" by hand
    (a pair is a reject if the chosen/rejected labels are wrong, the prompt is
    mislabeled ambiguous/clear, or either response is low quality). Then run
    --report.
    """
    pairs = [
        json.loads(l)
        for l in Path(pairs_path).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    rng = random.Random(seed)
    sample = rng.sample(pairs, min(k, len(pairs)))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in sample:
            rec = dict(rec)
            rec["verdict"] = ""  # Fill with "accept" or "reject" by hand.
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sample)} pairs to verify -> {out_path}")
    print("Fill in each 'verdict' (accept/reject) by hand, then run --report.")


def rejection_rate(path: str | Path = HAND_VERIFIED_PATH) -> float:
    """Computes the rejection rate over the hand-verified sample."""
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    verdicts = [str(r.get("verdict", "")).strip().lower() for r in rows]

    unfilled = [i for i, v in enumerate(verdicts) if v not in {"accept", "reject"}]
    if unfilled:
        raise ValueError(
            f"{len(unfilled)} of {len(rows)} verdicts are not 'accept'/'reject' "
            f"(first at row {unfilled[0]}). Finish hand-verifying before reporting."
        )

    n_reject = sum(1 for v in verdicts if v == "reject")
    rate = n_reject / len(rows)

    print(f"Hand-verified {len(rows)} pairs: {n_reject} rejected -> rejection rate {rate:.1%}")
    return rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generate",
        action="store_true",
        help="synthesize + write pairs, then sample for verification",
    )

    parser.add_argument(
        "--n", type=int, default=MIN_PREFERENCE_PAIRS, help="how many pairs to synthesize"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="compute the rejection rate from the hand-verified file",
    )

    args = parser.parse_args()

    if args.generate:
        pairs = synthesize_pairs(args.n)
        write_pairs(pairs)
        sample_for_verification()

    if args.report:
        rejection_rate()

    if not (args.generate or args.report):
        parser.print_help()


if __name__ == "__main__":
    main()
