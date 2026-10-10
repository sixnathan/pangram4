"""Pangram-4 model: backbone + LoRA + classification heads.
Stage 1 heads: segment (15-way f_ai buckets) + humanizer (4-way, probe).
Stage 2 adds: tokenwise (3-way, every position) + mixed-authorship (2-way).
Run: imported by trainstageone.py / trainstagetwo.py. Smoke test: uv run python model.py"""
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

BACKBONE = "Qwen/Qwen3-1.7B"
N_BUCKETS = 15
N_HUMANIZER = 4
N_TOKEN = 3
N_MIXED = 2

def build_lora_config() -> LoraConfig:
    """One place for the LoRA recipe (paper Table 1): attention + dense ONLY,
    never routed experts (85% FNR). Shared by stage 1 and stage 2 so both
    adapters have identical key names (required for checkpoint loading)."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

class PangramModel(nn.Module):
    def __init__(self, backbone_name: str = BACKBONE, stage: int = 1):
        super().__init__()
        self.stage = stage
        base = AutoModelForCausalLM.from_pretrained(backbone_name, torch_dtype=torch.bfloat16)
        self.backbone = get_peft_model(base, build_lora_config())
        hidden_dim = self.backbone.config.hidden_size

        self.segment_head = nn.Linear(hidden_dim, N_BUCKETS)
        self.heads = nn.Linear(hidden_dim, N_HUMANIZER)
        if stage >= 2:
            self.tokenwise_head = nn.Linear(hidden_dim, N_TOKEN)
            self.mixed_head = nn.Linear(hidden_dim, N_MIXED)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = out.hidden_states[-1]
        last_idx = attention_mask.sum(dim=1) - 1
        hidden_last = hidden[torch.arange(hidden.size(0)), last_idx]

        segment_logits = self.segment_head(hidden_last.float())
        humanizer_logits = self.heads(hidden_last.detach().float())
        if self.stage == 1:
            return segment_logits, humanizer_logits

        token_logits = self.tokenwise_head(hidden.float())
        mixed_logits = self.mixed_head(hidden_last.float())
        return segment_logits, humanizer_logits, token_logits, mixed_logits

def load_model(backbone_name: str = BACKBONE, stage: int = 1) -> PangramModel:
    model = PangramModel(backbone_name, stage).to("cuda")
    return model

def count_params(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

if __name__ == "__main__":
    model1 = load_model(stage=1)
    total, trainable = count_params(model1)
    print(f"stage1 params: {total:,} total | {trainable:,} trainable ({trainable / total:.3%})")
    ids = torch.randint(0, 150000, (2, 64)).to("cuda")
    mask = torch.ones(2, 64, dtype=torch.long).to("cuda")
    seg, hum = model1(ids, mask)
    print(f"stage1: segment {seg.shape}   humanizer {hum.shape}")
    assert seg.shape == (2, 15) and hum.shape == (2, 4)
    del model1
    torch.cuda.empty_cache()

    model2 = load_model(stage=2)
    total, trainable = count_params(model2)
    print(f"stage2 params: {total:,} total | {trainable:,} trainable ({trainable / total:.3%})")
    seg, hum, tok, mixed = model2(ids, mask)
    print(f"stage2: segment {seg.shape}   humanizer {hum.shape}   "
          f"tokenwise {tok.shape}   mixed {mixed.shape}")
    assert tok.shape == (2, 64, 3) and mixed.shape == (2, 2)
    print("smoke test OK")
