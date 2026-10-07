"""Build the training dataset: labeled pairs + binary human docs -> tokenized Arrow.
Run from scripts/:  uv run python makedataset.py"""
import json
from bisect import bisect_right

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-1.7B"
MAX_LEN = 512
N_HUMAN_DOCS = 30000
IGNORE = -100  # pytorch ignore-index: padded positions teach nothing
LABEL_IDS = {"human": 0, "ai_assisted": 1, "ai_generated": 2}


def char_span_to_label(clauses: list, a: int, b: int) -> int:
    """Which clause contains this token's midpoint? Token inherits clause label."""
    mid = (a + b) // 2
    starts = [c["start"] for c in clauses]
    i = bisect_right(starts, mid) - 1  # last clause starting at/before mid
    if 0 <= i < len(clauses) and clauses[i]["start"] <= mid < clauses[i]["end"]:
        return LABEL_IDS[clauses[i]["label"]]
    return LABEL_IDS["human"]  # gaps between clauses (spaces) default to human


def build_pair_example(tok, row: dict) -> dict | None:
    enc = tok(row["t_text"], truncation=True, max_length=MAX_LEN, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    labels = [char_span_to_label(row["clauses"], a, b) for a, b in offsets]
    return {"input_ids": ids, "token_labels": labels, "f_ai": row["f_ai"]}


def finalize(tok, ex: dict) -> dict:
    """Pad to MAX_LEN, add attention mask, bucket f_ai, compute is_mixed."""
    n = len(ex["input_ids"])
    pad = MAX_LEN - n
    labels = ex["token_labels"] + [IGNORE] * pad
    arr = [l for l in ex["token_labels"] if l != IGNORE]
    dominant = max(set(arr), key=arr.count)
    non_dominant = sum(1 for l in arr if l != dominant)
    return {
        "input_ids": ex["input_ids"] + [tok.pad_token_id] * pad,
        "attention_mask": [1] * n + [0] * pad,
        "token_labels": labels,
        "f_ai_bucket": min(14, int(ex["f_ai"] * 15)),
        "is_mixed": int(non_dominant / len(arr) > 0.15),
        "is_humanized": 0,
    }


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 1. labeled pairs
    examples = []
    with open("../data/mirage_labeled.jsonl") as f:
        for line in f:
            ex = build_pair_example(tok, json.loads(line))
            if ex and len(ex["input_ids"]) > 32:
                examples.append(ex)
    print(f"pairs kept: {len(examples)}")

    # 2. human docs from HC3 (f_ai = 0, every token human; empty clauses -> default)
    hc3 = load_from_disk("../data/raw/hc3")["train"].select(range(N_HUMAN_DOCS))
    added = 0
    for ans in hc3["human_answers"]:
        for text in ans if isinstance(ans, list) else [ans]:
            ex = build_pair_example(tok, {"t_text": text, "clauses": [], "f_ai": 0.0})
            if ex and len(ex["input_ids"]) > 32:
                examples.append(ex)
                added += 1
    print(f"total examples: {len(examples)} (+{added} human)")

    # 3. finalize + shuffle + split + save
    ds = Dataset.from_list(examples)
    ds = ds.map(lambda ex: finalize(tok, ex), remove_columns=["f_ai"])
    ds = ds.shuffle(seed=42)
    split = ds.train_test_split(test_size=0.02, seed=42)
    split["train"].save_to_disk("../data/processed/train")
    split["test"].save_to_disk("../data/processed/val")
    print(split)
    print("saved to data/processed/")


if __name__ == "__main__":
    main()
