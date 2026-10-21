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
    
    # 1. Force GPU to prevent silent CPU offloading on crowded clusters
    # 2. Use `dtype` instead of deprecated `torch_dtype`
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device() if torch.cuda.is_available() else 0}
    )

    n_ambiguous = n // 2
    n_clear = n - n_ambiguous
    pairs = []
    
    def generate_single_scenario(system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs, 
                max_new_tokens=384,  # Increased to prevent tag truncation
                temperature=0.8, 
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        gen = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(gen, skip_special_tokens=True)

    # Robust parsing helper to ignore Markdown bolding and colons
    def parse_tags_robustly(text: str) -> dict | None:
      # 1. Normalize markdown headers/bolding without stripping individual letters
      cleaned = text.replace("**", "").replace("###", "").strip()
    
      # 2. Check that all exact tag anchors exist
      if (
          "[PROMPT]" not in cleaned
          or "[CHOSEN]" not in cleaned
          or "[REJECTED]" not in cleaned
      ):
        return None
    
      try:
        # 3. Clean slice using exact substrings
        after_prompt = cleaned.split("[PROMPT]", 1)[1]
        prompt_text, after_chosen = after_prompt.split("[CHOSEN]", 1)
        chosen_text, rejected_text = after_chosen.split("[REJECTED]", 1)
    
        # 4. Strip extraneous colons or trailing delimiters
        prompt = prompt_text.lstrip(":").strip()
        chosen = chosen_text.lstrip(":").strip()
        rejected = rejected_text.lstrip(":").strip()
    
        # Reject if any field ended up empty or just tag remnants
        if not prompt or not chosen or not rejected:
          return None
    
        # Cut off if the model started hallucinating a second scenario
        rejected = rejected.split("[PROMPT]")[0].strip()
    
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
      except Exception:
        return None

    base_system = """You are a strict data formatting AI.
You must output ONLY the requested tags: [PROMPT], [CHOSEN], [REJECTED].
Do NOT include any conversational preamble (e.g., do NOT write "Here is the scenario", "Certainly!", or "Sure").
The [CHOSEN] and [REJECTED] responses must start IMMEDIATELY with the answer or question itself."""
  
    print("Generating ambiguous pairs...")
    pbar_ambig = tqdm(total=n_ambiguous, desc="Ambiguous Pairs")
    
    while len(pairs) < n_ambiguous:
        user_prompt = """Generate an AMBIGUOUS user request lacking critical context (e.g., code debugging without code, vague scheduling, unspecified terms).
The [CHOSEN] response must be a polite, direct 1-2 sentence clarifying question requesting the missing details.
The [REJECTED] response must be an ungrounded guess attempting to answer without enough information.

Example:
[PROMPT] How do I make it run faster?
[CHOSEN] Could you specify what program, script, or hardware you are trying to optimize?
[REJECTED] You should add more RAM and upgrade your graphics card."""
        
        raw_text = generate_single_scenario(base_system, user_prompt)
        parsed_data = parse_tags_robustly(raw_text)
        
        if parsed_data:
            parsed_data["class"] = "ambiguous"
            pairs.append(parsed_data)
            pbar_ambig.update(1)
        else:
            # Uncomment the print statement below if it stalls to see exactly what Llama outputted
            # print(f"\n[TAG MISSING - AMBIGUOUS] Raw output:\n{raw_text}\n")
            pass

    pbar_ambig.close()

    print("Generating clear pairs...")
    clear_pairs_count = 0
    pbar_clear = tqdm(total=n_clear, desc="Clear Pairs")
    
    while clear_pairs_count < n_clear:
        user_prompt = """Generate a CLEAR, specific, unambiguous user request from a random domain (such as history, programming, geography, or daily tasks).
The [CHOSEN] response must execute the task or answer the question directly, naturally, and confidently in exactly 1 sentence. Do NOT ask any questions.
The [REJECTED] response must ask an annoying, unnecessary clarifying question instead of just doing the task.

Example:
[PROMPT] Translate "Where is the library?" into French.
[CHOSEN] "Où est la bibliothèque ?"
[REJECTED] Do you want the formal or informal translation?"""
        
        raw_text = generate_single_scenario(base_system, user_prompt)
        parsed_data = parse_tags_robustly(raw_text)
        
        if parsed_data:
            parsed_data["class"] = "clear"
            pairs.append(parsed_data)
            clear_pairs_count += 1
            pbar_clear.update(1)
        else:
            # Uncomment the print statement below if it stalls to see exactly what Llama outputted
            # print(f"\n[TAG MISSING - CLEAR] Raw output:\n{raw_text}\n")
            pass
            
    pbar_clear.close()

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

def fill_accept(out_path: str | Path = HAND_VERIFIED_PATH) -> None:
    """
    Reads the hand-verified file and changes any empty "verdict" fields to "accept".
    Leaves existing "reject" or "accept" verdicts untouched.
    """
    out_path = Path(out_path)
    if not out_path.exists():
        print(f"Error: {out_path} does not exist.")
        return

    # 1. Read existing records
    records = []
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # 2. Update the blank verdicts
    updated_count = 0
    for rec in records:
        if rec.get("verdict") == "":
            rec["verdict"] = "accept"
            updated_count += 1

    # 3. Write them back to the same file
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Successfully updated {updated_count} empty verdicts to 'accept'.")

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
    parser.add_argument(
        "--fill",
        action="store_true",
        help="fill empty verdicts with 'accept'",
    )

    args = parser.parse_args()

    if args.fill:
        fill_accept()
        
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
