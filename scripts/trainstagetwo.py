"""Stage-2 training (WIP): humanizer head on humanized documents.

collaterepeat2 doubles each example: the first copy is unlabeled context
(token_labels all -100), the second copy carries the real labels.
"""
import torch

from model import load_model, count_params

EPOCHS = 2
BATCH = 16
LR = 1e-4
HUMANIZER_WEIGHT = 0.25
IGNORE = -100
STAGE1_CKPT = "../checkpoints/stage1.pt"
CKPT = "../checkpoints/stage2.pt"


def collaterepeat2(batch):
    """Stack a list of row dicts into doubled sequences (context + labeled)."""
    return {
        "input_ids": torch.tensor(
            [b["input_ids"] + b["input_ids"] for b in batch], dtype=torch.long),
        "attention_mask": torch.tensor(
            [b["attention_mask"] + b["attention_mask"] for b in batch], dtype=torch.long),
        "token_labels": torch.tensor(
            [[IGNORE] * len(b["input_ids"]) + b["token_labels"] for b in batch],
            dtype=torch.long),
        "f_ai_bucket": torch.tensor([b["f_ai_bucket"] for b in batch]),
        "is_humanized": torch.tensor([b["is_humanized"] for b in batch]),
        "is_mixed": torch.tensor([b["is_mixed"] for b in batch]),
    }
