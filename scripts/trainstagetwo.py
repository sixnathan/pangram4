"""Stage 2 training: merge stage-1 LoRA into base, fresh LoRA, add tokenwise +
mixed-authorship heads with Repeat2 inputs (paper §4.1, Table 2).
Run from scripts/:  uv run python trainstagetwo.py
(only AFTER trainstageone.py has finished — it loads ../checkpoints/stage1.pt)"""
import os
import time

import torch
import wandb
from datasets import load_from_disk
from peft import get_peft_model
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model import load_model, count_params, build_lora_config

EPOCHS = 2
BATCH = 16          # Repeat2 doubles sequence length to 1024; attention cost
                    # scales ~4x per sequence, so halve the batch vs stage 1
LR = 1e-4
HUMANIZER_WEIGHT = 0.25  # same as stage 1: humanizer can't distort the main task
IGNORE = -100            # pytorch ignore-index; copy 1 of Repeat2 is all IGNORE
STAGE1_CKPT = "../checkpoints/stage1.pt"
CKPT = "../checkpoints/stage2.pt"


def collate_repeat2(batch):
    """batch is a LIST of 16 example-dicts (plain Python lists inside).
    Repeat2 (paper §4.1): x̂ = (x, x) -> (B, 1024). For lists, '+' means
    concatenation (NOT addition): [5,6] + [5,6] -> [5,6,5,6].
    Copy 1 gets all-IGNORE labels (no loss there); copy 2 carries the real
    token_labels, which makedataset already padded with IGNORE."""
    return {
        "input_ids":      torch.tensor([b["input_ids"] + b["input_ids"] for b in batch],
                                       dtype=torch.long),
        "attention_mask": torch.tensor([b["attention_mask"] + b["attention_mask"] for b in batch],
                                       dtype=torch.long),
        "token_labels":   torch.tensor([[IGNORE] * len(b["input_ids"]) + b["token_labels"]
                                        for b in batch], dtype=torch.long),
        "f_ai_bucket":    torch.tensor([b["f_ai_bucket"] for b in batch]),
        "is_humanized":   torch.tensor([b["is_humanized"] for b in batch]),
        "is_mixed":       torch.tensor([b["is_mixed"] for b in batch]),
    }


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    """Val pass: no backward, no step. Returns a dict of 4 losses + 4 accuracies.
    Tokenwise accuracy counts ONLY real (non-IGNORE) positions = copy 2."""
    model.eval()
    sums = {"seg_loss": 0.0, "hum_loss": 0.0, "tok_loss": 0.0, "mixed_loss": 0.0}
    correct = {"seg_acc": 0, "hum_acc": 0, "tok_acc": 0, "mixed_acc": 0}
    n_batches = tok_total = 0
    for batch in loader:
        batch = {k: v.to("cuda") for k, v in batch.items()}
        seg, hum, tok, mixed = model(batch["input_ids"], batch["attention_mask"])
        labels = batch["token_labels"]

        sums["seg_loss"] += loss_fn(seg, batch["f_ai_bucket"]).item()
        sums["hum_loss"] += loss_fn(hum, batch["is_humanized"]).item()
        sums["tok_loss"] += loss_fn(tok.reshape(-1, tok.size(-1)), labels.reshape(-1)).item()
        sums["mixed_loss"] += loss_fn(mixed, batch["is_mixed"]).item()

        correct["seg_acc"] += (seg.argmax(dim=1) == batch["f_ai_bucket"]).sum().item()
        correct["hum_acc"] += (hum.argmax(dim=1) == batch["is_humanized"]).sum().item()
        mask = labels.ne(IGNORE)  # (B, 1024) bool: True = real token on copy 2
        correct["tok_acc"] += (tok.argmax(dim=-1) == labels)[mask].sum().item()
        tok_total += mask.sum().item()
        correct["mixed_acc"] += (mixed.argmax(dim=1) == batch["is_mixed"]).sum().item()
        n_batches += 1
    model.train()  # dropout back on
    out = {k: v / n_batches for k, v in sums.items()}
    out.update({k: v / (len(loader.dataset)) for k, v in correct.items() if k != "tok_acc"})
    out["tok_acc"] = correct["tok_acc"] / tok_total  # per-TOKEN, not per-example
    return out


def merge_stage1():
    """Load the stage-1 checkpoint, fold its trained LoRA into the base weights
    (merge_and_unload), then attach a FRESH random LoRA. Stage-1 knowledge is
    now baked into the frozen weights; the new adapter learns the tokenwise
    task from scratch without destroying it (paper Table 2)."""
    model = load_model(stage=2)
    missing, unexpected = model.load_state_dict(torch.load(STAGE1_CKPT), strict=False)
    # strict=False: stage1.pt has no tokenwise/mixed keys -> they stay random.
    # everything TRAINED (backbone LoRA + segment + humanizer) must be covered:
    bad = [k for k in missing if "tokenwise_head" not in k and "mixed_head" not in k]
    assert not bad, f"stage-1 checkpoint is missing trained keys: {bad}"
    assert not unexpected, f"checkpoint has keys this model doesn't: {unexpected}"
    print(f"stage-1 loaded; left randomly initialized: {len(missing)} new-head tensors")

    merged = model.backbone.merge_and_unload()  # LoRA delta folded into W; wrapper gone
    model.backbone = get_peft_model(merged, build_lora_config())  # fresh adapter, same recipe
    return model


def main():
    if not os.path.exists(STAGE1_CKPT):
        raise SystemExit(f"no stage-1 checkpoint at {STAGE1_CKPT} — "
                         f"let trainstageone.py finish first")

    run = wandb.init(project="pangram4", name="stage2", config={
        "epochs": EPOCHS, "batch": BATCH, "lr": LR, "seq_len": 1024,
        "humanizer_weight": HUMANIZER_WEIGHT, "repeat2": True,
        "backbone": "Qwen/Qwen3-1.7B", "stage1_ckpt": STAGE1_CKPT,
    })

    model = merge_stage1()
    total, trainable = count_params(model)
    print(f"params: {total:,} total | {trainable:,} trainable ({trainable / total:.3%})")

    os.makedirs("../checkpoints", exist_ok=True)

    train_ds = load_from_disk("../data/processed/train")
    val_ds = load_from_disk("../data/processed/val")
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          collate_fn=collate_repeat2, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=BATCH, collate_fn=collate_repeat2, num_workers=4)
    print(f"train: {len(train_ds)} | val: {len(val_ds)}")

    # one-time collate sanity: copy 1 must be pure IGNORE, shapes must be 1024
    check = next(iter(DataLoader(val_ds, batch_size=2, collate_fn=collate_repeat2)))
    assert (check["token_labels"][:, :512] == IGNORE).all(), "copy 1 is not fully masked!"
    assert check["input_ids"].shape == (2, 1024), f"bad shape {check['input_ids'].shape}"
    print(f"collate OK: input {tuple(check['input_ids'].shape)}, "
          f"copy 1 masked, {int((check['token_labels'] != IGNORE).sum())} real label positions")

    # CE with ignore_index skips IGNORE positions when computing loss/mean;
    # seg/hum/mixed targets are never IGNORE so it changes nothing for them
    loss_fn = CrossEntropyLoss(ignore_index=IGNORE)
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        running = 0.0
        for step, batch in enumerate(train_dl):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            seg, hum, tok, mixed = model(batch["input_ids"], batch["attention_mask"])
            loss_seg = loss_fn(seg, batch["f_ai_bucket"])
            loss_hum = loss_fn(hum, batch["is_humanized"]) * HUMANIZER_WEIGHT
            loss_tok = loss_fn(tok.reshape(-1, tok.size(-1)), batch["token_labels"].reshape(-1))
            loss_mixed = loss_fn(mixed, batch["is_mixed"])
            loss = loss_seg + loss_tok + loss_mixed + loss_hum
            loss.backward()   # gradient of loss wrt every trainable param
            opt.step()        # move params against those gradients
            opt.zero_grad()   # clear grads so next batch starts fresh
            running += loss.item()
            wandb.log({"train/loss": loss.item(),
                       "train/loss_seg": loss_seg.item(),
                       "train/loss_tok": loss_tok.item(),
                       "train/loss_mixed": loss_mixed.item(),
                       "train/loss_hum": loss_hum.item()},
                      step=epoch * len(train_dl) + step)
            if step % 200 == 0:
                print(f"epoch {epoch} step {step:>5}/{len(train_dl)}  "
                      f"loss {loss.item():.4f}  {time.time() - t0:.0f}s")
            if step > 0 and step % 1000 == 0:
                torch.save(model.state_dict(), CKPT)  # crash insurance mid-epoch
                print("   checkpoint saved (mid-epoch)")

        vals = evaluate(model, val_dl, loss_fn)
        wandb.log({f"val/{k}": v for k, v in vals.items()},
                  step=(epoch + 1) * len(train_dl))
        print(f"== epoch {epoch}: train_loss {running / len(train_dl):.4f} | "
              f"val seg {vals['seg_loss']:.4f}/{vals['seg_acc']:.3f} "
              f"tok {vals['tok_loss']:.4f}/{vals['tok_acc']:.3f} "
              f"mixed {vals['mixed_loss']:.4f}/{vals['mixed_acc']:.3f} "
              f"hum {vals['hum_loss']:.4f}/{vals['hum_acc']:.3f} | {time.time() - t0:.0f}s")
        torch.save(model.state_dict(), CKPT)  # full model incl. merged backbone, ~3.4GB
        print(f"   checkpoint saved -> {CKPT}")

    print("training done")
    run.finish()


if __name__ == "__main__":
    main()
