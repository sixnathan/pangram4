"""Pangram-4 model: backbone + LoRA + classification heads.
Stage 1 heads: segment (15-way f_ai buckets) + humanizer (4-way, probe).
Run: imported by train_stage1.py. Smoke test: uv run python model.py"""
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

BACKBONE = "Qwen/Qwen3-1.7B"
N_BUCKETS = 15   # segment head classes: ordered f_ai buckets
N_HUMANIZER = 4  # human / ai / humanized_ai / mixed


class PangramModel(nn.Module):
    def __init__(self, backbone_name: str = BACKBONE):
        super().__init__()
        # backbone: causal LM in bf16; we want hidden states, not generated tokens
        base = AutoModelForCausalLM.from_pretrained(backbone_name, torch_dtype=torch.bfloat16)

        # LoRA on attention + dense projections ONLY — paper Table 1 showed
        # adding routed experts was catastrophic (85% FNR), so we never touch them
        lora = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",   # attention
                "gate_proj", "up_proj", "down_proj",       # dense / MLP
            ],
        )
        self.backbone = get_peft_model(base, lora)
        hidden_dim = self.backbone.config.hidden_size

        # both heads read the LAST real token's hidden state: under causal
        # attention that position has seen the entire window (paper §4.1)
        self.segment_head = nn.Linear(hidden_dim, N_BUCKETS)
        self.heads = nn.Linear(hidden_dim, N_HUMANIZER)  # humanizer head

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = out.hidden_states[-1]                     # (B, S, D)
        last_idx = attention_mask.sum(dim=1) - 1           # (B,) index of last real token
        hidden_last = hidden[torch.arange(hidden.size(0)), last_idx]  # (B, D)

        segment_logits = self.segment_head(hidden_last.float())      # (B, 15)
        humanizer_logits = self.heads(hidden_last.detach().float())  # probe: stop-gradient           # (B, 4)
        return segment_logits, humanizer_logits


def load_model(backbone_name: str = BACKBONE) -> PangramModel:
    model = PangramModel(backbone_name).to("cuda")
    return model


def count_params(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # fake batch of 2 sequences, length 64
    model = load_model()
    total, trainable = count_params(model)
    print(f"params: {total:,} total | {trainable:,} trainable ({trainable / total:.3%})")

    ids = torch.randint(0, 150000, (2, 64)).to("cuda")
    mask = torch.ones(2, 64, dtype=torch.long).to("cuda")
    seg, hum = model(ids, mask)
    print(f"segment_logits: {seg.shape}   humanizer_logits: {hum.shape}")
    assert seg.shape == (2, 15) and hum.shape == (2, 4)
    print("smoke test OK")
