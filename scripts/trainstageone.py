"""Stage 1 training: segment + humanizer heads on LoRA backbone.
Run from scripts/:  uv run python trainstageone.py"""
import os
import time

import torch
import wandb
from datasets import load_from_disk
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader  # class -> PascalCase, not a function

from model import load_model, count_params

EPOCHS = 2
BATCH = 32
LR = 1e-4
HUMANIZER_WEIGHT = 0.25  # paper Table 2: humanizer loss x0.25 so it can't distort the main task
CKPT = "../checkpoints/stage1.pt"


def collate(batch):
    """batch is a LIST of 32 dicts (one per example) ->
    one tensor per field: b['input_ids'] for b in batch pulls each row's field,
    torch.tensor stacks the 32 lists into a (32, 512) matrix."""
    return {
        "input_ids":      torch.tensor([b["input_ids"] for b in batch], dtype=torch.long),
        "attention_mask": torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long),
        "f_ai_bucket":    torch.tensor([b["f_ai_bucket"] for b in batch]),
        "is_humanized":   torch.tensor([b["is_humanized"] for b in batch]),
    }


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    """Val pass: no backward, no step, no_grad = no gradient graph (faster, less memory)."""
    model.eval()  # dropout off
    seg_loss = hum_loss = 0.0
    seg_correct = hum_correct = total = 0
    for batch in loader:
        batch = {k: v.to("cuda") for k, v in batch.items()}  # CPU -> GPU
        seg_logits, hum_logits = model(batch["input_ids"], batch["attention_mask"])
        seg_loss += loss_fn(seg_logits, batch["f_ai_bucket"]).item()  # .item() = tensor -> float
        hum_loss += loss_fn(hum_logits, batch["is_humanized"]).item()
        seg_correct += (seg_logits.argmax(dim=1) == batch["f_ai_bucket"]).sum().item()
        hum_correct += (hum_logits.argmax(dim=1) == batch["is_humanized"]).sum().item()
        total += len(batch["f_ai_bucket"])
    model.train()  # dropout back on
    n_batches = len(loader)
    return seg_loss / n_batches, hum_loss / n_batches, seg_correct / total, hum_correct / total


def main():
    # opens a run in project 'pangram4'; config = hyperparams, so every future
    # run is logged and compared side-by-side on the dashboard
    run = wandb.init(project="pangram4", name="stage1", config={
        "epochs": EPOCHS, "batch": BATCH, "lr": LR,
        "humanizer_weight": HUMANIZER_WEIGHT, "backbone": "Qwen/Qwen3-1.7B",
    })

    model = load_model()
    total, trainable = count_params(model)
    print(f"params: {total:,} total | {trainable:,} trainable ({trainable / total:.3%})")

    # torch.save does NOT create missing folders — last run had no checkpoints/
    # dir, so even a clean finish would have crashed on the final save
    os.makedirs("../checkpoints", exist_ok=True)

    train_ds = load_from_disk("../data/processed/train")
    val_ds = load_from_disk("../data/processed/val")
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          collate_fn=collate, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=BATCH, collate_fn=collate, num_workers=4)
    print(f"train: {len(train_ds)} | val: {len(val_ds)}")

    # optimizer over TRAINABLE params only (LoRA adapters + the two heads);
    # the frozen backbone params are excluded because requires_grad=False
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    loss_fn = CrossEntropyLoss()

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        running = 0.0
        for step, batch in enumerate(train_dl):
            batch = {k: v.to("cuda") for k, v in batch.items()}  # CPU -> GPU
            seg_logits, hum_logits = model(batch["input_ids"], batch["attention_mask"])
            loss_seg = loss_fn(seg_logits, batch["f_ai_bucket"])
            loss_hum = loss_fn(hum_logits, batch["is_humanized"]) * HUMANIZER_WEIGHT
            loss = loss_seg + loss_hum
            loss.backward()   # gradient of loss wrt every trainable param
            opt.step()        # move params against those gradients
            opt.zero_grad()   # clear grads so next batch starts fresh
            running += loss.item()
            wandb.log({"train/loss": loss.item(),
                       "train/loss_seg": loss_seg.item(),
                       "train/loss_hum": loss_hum.item()},
                      step=epoch * len(train_dl) + step)
            if step % 200 == 0:
                print(f"epoch {epoch} step {step:>5}/{len(train_dl)}  "
                      f"loss {loss.item():.4f}  {time.time() - t0:.0f}s")
            if step > 0 and step % 1000 == 0:
                torch.save(model.state_dict(), CKPT)  # crash insurance mid-epoch
                print("   checkpoint saved (mid-epoch)")

        vseg, vhum, vseg_acc, vhum_acc = evaluate(model, val_dl, loss_fn)
        wandb.log({"val/seg_loss": vseg, "val/hum_loss": vhum,
                   "val/seg_acc": vseg_acc, "val/hum_acc": vhum_acc},
                  step=(epoch + 1) * len(train_dl))
        print(f"== epoch {epoch}: train_loss {running / len(train_dl):.4f} | "
              f"val seg_loss {vseg:.4f} acc {vseg_acc:.3f} | "
              f"val hum_loss {vhum:.4f} acc {vhum_acc:.3f} | {time.time() - t0:.0f}s")
        torch.save(model.state_dict(), CKPT)  # full model incl. frozen backbone, ~3.4GB
        print(f"   checkpoint saved -> {CKPT}")

    print("training done")
    run.finish()


if __name__ == "__main__":
    main()