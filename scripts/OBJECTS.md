# Objects reference — what every import actually creates

Everything below was **actually executed** against this repo (`.venv`, real Qwen3-1.7B,
real `data/processed/`). These are literal dumps, not illustrations.

---

## Part A — The complete inventory

### First, the vocabulary

| Term | What it is | Example |
|---|---|---|
| **module** | one `.py` file's worth of names, wrapped in a single object. `import json` binds the *name* `json` to this object | `<module 'json' from '.../json/__init__.py'>` |
| **class** | a blueprint object. `class Foo:` creates one. Calling it (`Foo()`) makes an **instance** | `torch.Tensor`, `DataLoader`, `LoraConfig` |
| **function** | a callable object created by `def` | `torch.tensor`, `load_from_disk`, `load_model` |
| **instance** | what you get when you *call* a class: `LoraConfig(r=16)` → a `LoraConfig` instance | your `model`, `opt`, `ds`, `run` |

The key mental model: `import X` never creates your data. It only binds a *name* to a
*thing* (module / class / function). **Objects only exist when something is called.**

### A1. `model.py`

| Import | What the name itself is | What calling it creates |
|---|---|---|
| `torch` | module | — |
| `torch.nn as nn` | module | — |
| `nn.Module` | class | base class of `PangramModel` (never called directly) |
| `peft.LoraConfig` | class | one config *record* (37 plain fields) |
| `peft.get_peft_model` | function | a `PeftModel` that wraps your base model **in place** |
| `transformers.AutoModelForCausalLM` | class (a factory) | `.from_pretrained(...)` → `Qwen3ForCausalLM` instance (~1.7B tensors) |
| `PangramModel` (own) | class | the model: 3 children — `backbone` (PeftModel), `segment_head` (Linear), `heads` (Linear) |
| `load_model`, `count_params` (own) | functions | `PangramModel` instance / a tuple `(total, trainable)` |

### A2. `trainstageone.py` (same set in `trainstagetwo.py`)

| Import | The name is | Calling it creates |
|---|---|---|
| `os`, `time` | modules | — |
| `torch` | module | — |
| `wandb` | module | — |
| `datasets.load_from_disk` | function | one `datasets.Dataset` (Arrow table) |
| `peft.LoraConfig` / `get_peft_model` | class / function | (only used in `model.py`) |
| `torch.nn.CrossEntropyLoss` | class | one tiny callable holding `reduction='mean'`, `ignore_index=-100` |
| `torch.optim.AdamW` | class | optimizer holding `param_groups` = settings + the list of trainable `Parameter`s |
| `torch.utils.data.DataLoader` | class | an iterable that yields **dicts of stacked Tensors** |
| `model.load_model` / `count_params` | functions | `PangramModel` / `(int, int)` |
| `wandb.init` | function | one `Run` object (`.config`, `.log`, `.finish`) |

### A3. `makedataset.py`

| Import | The name is | Calling it creates |
|---|---|---|
| `json` | module | — |
| `bisect.bisect_right` | function | returns an `int` |
| `datasets.Dataset` | class | `Dataset.from_list(...)` → Dataset; `.map/.shuffle/.save_to_disk` return/save Datasets |
| `datasets.load_from_disk` | function | `Dataset` |
| `transformers.AutoTokenizer` | class (factory) | `.from_pretrained(...)` → `Qwen2Tokenizer` |
| calling `tok(text)` | — | a `BatchEncoding`: dict of `input_ids`, `attention_mask`, `offset_mapping` |

### A4. `labeler.py`, `softgrams.py`, `chargnramsf1.py`

| Import | The name is | Calling it creates |
|---|---|---|
| `argparse` | module | `ArgumentParser()` → parser; `.parse_args()` → `Namespace` |
| `json` | module | — |
| `torch` | module | — |
| `sentence_transformers.SentenceTransformer` | class | a MiniLM encoder (BertModel + mean-pool + normalize) |
| `re` | module | `re.compile(...)` → `Pattern`; `.finditer` yields `Match` objects |
| `pathlib.Path` | class | `PosixPath('data/raw')` |
| `collections.Counter` | class | `Counter({'the': 1, 'he ': 1, ...})` |
| `splitclauses` (own) | function | `list[tuple[str, int, int]]` |
| `make_grams` / `charngramf1` / `f1_from_grams` (own) | functions | `Counter` / `float` / `float` |
| `label_pair` (own) | function | a plain `dict` (see Part B.14) |

### A5. `download_data.py`, `validate_labeller.py`

| Import | The name is | Calling it creates |
|---|---|---|
| `datasets.load_dataset` | function | `DatasetDict` (a dict of Datasets: train/dev/test) |
| `pandas` | module | `pd.read_csv(...)` → `DataFrame` |
| `scipy.stats.spearmanr` | function | a result object with `.statistic`, `.pvalue` (unpackable to `rho, p`) |

---

## Part B — What each object LITERALLY looks like

### B1. A module object — the thing `import json` gives you
```
>>> type(json)
<class 'module'>
>>> json
<module 'json' from '/home/.../lib/python3.12/json/__init__.py'>
>>> type(json.dumps)
<class 'function'>
```
A module is just a bag of names. `torch` is also a module; `torch.nn` is another
module *inside* it; `torch.nn.CrossEntropyLoss` is a class inside *that*.

### B2. `Counter` — what `make_grams("the cat")` returns
```
Counter({'the': 1, 'he ': 1, 'e c': 1, ' ca': 1, 'cat': 1})
```
A dict `{trigram: count}`. Missing keys give `0` instead of KeyError.
`ga & gb` = multiset intersection → `Counter({'the': 1, 'he ': 1})` →
`f1_from_grams(ga, gb)` = `2*2 / (5+5) = 0.4`.

### B3. `splitclauses` output — list of tuples
```
[('I like dogs.', 0, 12), (' You like cats!', 12, 27), (' Ok?', 27, 31)]
```
Each tuple = `(text, start_char, end_char)`. Inside, `re.Pattern.finditer` yields
`<re.Match object; span=(11, 12), match='.'>` and `m.end()` is where the clause ended.

### B4. `Path` and `Namespace`
```
PosixPath('data/raw')          # RAW / "raid" -> PosixPath('data/raw/raid')
Namespace(pairs_jsonl='in.jsonl', out_jsonl='out.jsonl', inspect=False, limit=5)
```
`Namespace` = an object whose only job is holding the CLI flags as attributes.

### B5. `torch.Tensor` — the universal container
```
>>> torch.tensor([[5303, 366, 10061, 13],
...               [9707,  11,   101,  861]])
tensor([[ 5303,   366, 10061,    13],
        [ 9707,    11,   101,   861]])
.shape -> torch.Size([2, 4])   .dtype -> torch.int64   .device -> cpu
```
Every tensor knows three things: **shape, dtype, device**. Indexing:
`t2[1]` → a 1-D tensor (a row); `t2[1, 2]` → `tensor(101)` (0-D scalar).
`.to("cuda")` returns the *same values* living in GPU memory: `device: cuda:0`.

### B6. `CrossEntropyLoss()` — almost empty
```
CrossEntropyLoss()
loss_fn.__dict__ = {'training': True, '_parameters': {}, '_buffers': {'weight': None},
                    '_modules': {}, 'reduction': 'mean', 'ignore_index': -100,
                    'label_smoothing': 0.0, ...hooks...}
```
All it stores is *how* to reduce and the ignore index. The math happens in `__call__`.

### B7. `AdamW` — settings + pointers to your weights
```
AdamW (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    lr: 0.0001
    weight_decay: 0.01
    ...
)
```
`opt.param_groups[0]["params"]` is the actual list of `Parameter` objects it will
update. Each `Parameter` **is** a tensor wearing a grad-flag:
```
Parameter containing:
tensor([[-1.3338,  0.9621, -0.7132], ...], requires_grad=True)
```
Lifecycle per batch: `loss.backward()` fills `p.grad` (same shape as `p`) →
`opt.step()` moves `p` against it → `opt.zero_grad()` sets `p.grad` back to `None`.

### B8. `nn.Linear` — what the heads (and every q_proj) are
```
Linear(in_features=8, out_features=4, bias=True)
weight: Parameter shape (4, 8)     # (out, in)
bias:   Parameter shape (4,)
```
The whole "layer" is literally two tensors plus `y = x @ weight.T + bias`.

### B9. `datasets.Dataset` — your real training set
```
Dataset({
    features: ['input_ids', 'token_labels', 'attention_mask', 'f_ai_bucket', 'is_mixed', 'is_humanized'],
    num_rows: 145671
})
```
Schema (`ds.features`):
```
{'input_ids': List(Value('int32')), 'token_labels': List(Value('int64')),
 'attention_mask': List(Value('int8')), 'f_ai_bucket': Value('int64'),
 'is_mixed': Value('int64'), 'is_humanized': Value('int64')}
```
One row (`ds[0]` is a plain dict):
```
keys: ['input_ids', 'token_labels', 'attention_mask', 'f_ai_bucket', 'is_mixed', 'is_humanized']
input_ids:      [1655, 220, 18, 23, 1635, 2310, 11, 21998, 23413, 331, ...]  # 512 ints
attention_mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...]                          # 512 ints
token_labels:   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...]                          # 512 ints
f_ai_bucket:    11        is_mixed: 1        is_humanized: 0
```
The tail is padding: `input_ids[-8:] = [151643]*8` (pad = `<|endoftext|>`),
`attention_mask[-8:] = [0]*8`, `token_labels[-8:] = [-100]*8`.
Decode the head and it's real text: `'At 38 years old, Carl Froch, a native of
Nottingham and a four-time world champion, has officially'`.

### B10. `DataLoader` — the packaging machine
```
<torch.utils.data.dataloader.DataLoader object at 0x...>
.batch_size = 32   .dataset = Dataset({... 145671 rows})
.sampler = <RandomSampler>        # shuffle=True
```
What iteration yields:
* before `collate_fn`: a **list of 32 dicts** (each dict = one row, exactly as in B9)
* after your `collate`: **one dict of tensors**:
```
batch['input_ids']:      Tensor shape=(32, 512) dtype=torch.int64
batch['attention_mask']: Tensor shape=(32, 512) dtype=torch.int64
batch['f_ai_bucket']:    Tensor shape=(32,)
batch['is_humanized']:   Tensor shape=(32,)
```
`torch.tensor([b["input_ids"] for b in batch])` = "walk down the column, stack
the 32 lists into a matrix". In `trainstagetwo.py`'s `collaterepeat2`,
`b["input_ids"] + b["input_ids"]` is **list concatenation** (not tensor addition) —
each 512-list becomes a 1024-list, so the batch is `(32, 1024)`: every example
played twice in a row, first copy fully label-masked with `-100`.

### B11. `Qwen2Tokenizer` and its output
```
Qwen2Tokenizer(name_or_path='Qwen/Qwen3-1.7B', vocab_size=151643,
               model_max_length=131072, padding_side='right',
               special_tokens={'eos_token': '<|im_end|>', 'pad_token': '<|endoftext|>'},
               added_tokens_decoder={151643: AddedToken('<|endoftext|>', ...), ...})
```
Calling it returns a `BatchEncoding` (a dict subclass):
```
enc = tok("Carl Froch is a boxer.", return_offsets_mapping=True)
enc['input_ids']      -> [76164, 23413, 331, 374, 264, 78457, 13]
enc['offset_mapping'] -> [(0, 4), (4, 8), (8, 10), (10, 13), (13, 15), (15, 21), (21, 22)]

  id  76164  chars[0:4]   -> 'Carl'
  id  23413  chars[4:8]   -> ' Fro'
  id    331  chars[8:10]  -> 'ch'      # one word can be SEVERAL tokens
  id    374  chars[10:13] -> ' is'
  ...
```
The offsets are how `makedataset.py` gives each token its clause's label
(token midpoint → which clause contains it).

### B12. `LoraConfig` — a 37-field record, nothing else
```
LoraConfig(task_type=None, peft_type=<PeftType.LORA: 'LORA'>, r=16,
           target_modules={'k_proj','up_proj','v_proj','gate_proj','down_proj','o_proj','q_proj'},
           lora_alpha=32, lora_dropout=0.05, bias='none', ...)
```
It never touches weights. It's instructions for `get_peft_model`.

### B13. The model — three nested objects deep

**(a) `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")`** →
`Qwen3ForCausalLM`:
```
Qwen3ForCausalLM(
  (model): Qwen3Model(
    (embed_tokens): Embedding(151936, 2048)
    (layers): ModuleList(
      (0-27): 28 x Qwen3DecoderLayer(
        (self_attn): Qwen3Attention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (k_proj): ...  (v_proj): ...  (o_proj): ...
          (q_norm): Qwen3RMSNorm  (k_norm): Qwen3RMSNorm )
        (mlp): Qwen3MLP(...)
        (input_layernorm / post_attention_layernorm): Qwen3RMSNorm ) ) )
  (lm_head): Linear(in_features=2048, out_features=151936, bias=False)
)
```
config: `hidden_size=2048, num_hidden_layers=28, num_attention_heads=16,
vocab_size=151936`. The raw `q_proj` is just `Linear(2048→2048)` with a
`(2048, 2048)` bf16 tensor.

**(b) `get_peft_model(base, lora)`** — same object returned, but every target
`Linear` is **replaced in place** by a `lora.Linear`:
```
lora.Linear(
  (base_layer): Linear(in_features=2048, out_features=2048, bias=False)   # the ORIGINAL, now frozen
  (lora_dropout): ModuleDict((default): Dropout(p=0.05))
  (lora_A): ModuleDict((default): Linear(in_features=2048, out_features=16, bias=False))
  (lora_B): ModuleDict((default): Linear(in_features=16, out_features=2048, bias=False))
  (lora_embedding_A/B): ParameterDict()   (lora_magnitude_vector): ModuleDict()
)
```
Its three parameters tell the whole story:
```
base_layer.weight       (2048, 2048)  requires_grad=False   # frozen genius
lora_A.default.weight   (16, 2048)    requires_grad=True    # random init
lora_B.default.weight   (2048, 16)    requires_grad=True    # ALL ZEROS at start
scaling['default'] = 2.0    # lora_alpha / r = 32 / 16
```
Forward = `base_layer(x) + scaling * lora_B(lora_A(dropout(x)))`. Since `lora_B`
starts at zero, the model behaves *exactly* like stock Qwen3 on step 0.

**(c) `load_model()`** → your `PangramModel`:
```
children:  backbone  PeftModel      # (b) above
           segment_head  Linear(in_features=2048, out_features=15, bias=True)
           heads         Linear(in_features=2048, out_features=4, bias=True)
count_params -> 1,738,046,483 total | 17,471,507 trainable (1.005%)
```
First trainable param names (what AdamW actually receives):
```
backbone.base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
backbone.base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight
...
```
The silly-looking path decodes as: `backbone`(PangramModel's child) →
`.base_model`(peft's LoraModel wrapper) → `.model`(Qwen3ForCausalLM) →
`.model`(Qwen3Model) → `.layers[0]` → ... Same object, four naming layers.

**`torch.save(model.state_dict(), ...)`** writes a dict of
`{name: tensor}` — 707 entries, first ones:
```
backbone.base_model.model.model.embed_tokens.weight                         (151936, 2048) bfloat16
backbone...layers.0.self_attn.q_proj.base_layer.weight                      (2048, 2048)  bfloat16
backbone...layers.0.self_attn.q_proj.lora_A.default.weight                  (16, 2048)    float32
backbone...layers.0.self_attn.q_proj.lora_B.default.weight                  (2048, 16)    float32
```
Total ≈ **4.13 GB** on disk (the frozen backbone included, which is why it's so big
even though only ~17M params train).

### B14. The forward pass, literally

Backbone output (a `CausalLMOutputWithPast`, dict-like):
```
out.keys() -> odict_keys(['logits', 'past_key_values', 'hidden_states'])
len(out.hidden_states) = 29           # embedding + 28 layers
hidden[-1].shape = (2, 512, 2048)     # (batch, seq, hidden), bfloat16

hidden[-1][0, :4, :6]:
tensor([[ 0.2236,  0.9648, -1.7812,  1.8672,  0.1865,  1.0234],
        [ 0.3574, -0.4824, -0.6016, -1.0391, -0.2930,  0.2871],
        [-0.4805, -0.2578,  0.1104, -1.1016,  0.5117, -0.5391],
        [-1.0625, -0.1982,  0.5898,  0.0245, -0.9297, -0.0193]])
```
Heads on a real val batch (`f_ai_bucket=[9,0]`, `is_humanized=[0,0]`):
```
seg_logits (2, 15):
tensor([[ 2.1947,  2.0113,  0.1548, -2.6972,  0.9280, -1.4136, -0.4580, -1.0846,
         -0.3810, -0.9609,  1.2658,  0.9248, -0.0645,  0.4331, -0.9395],
        [ 1.4091, -0.0137, -0.1526, -3.7510, ... ]])
humanizer_logits (2, 4):
tensor([[ 4.1040,  1.0424,  2.4310, -0.1194],
        [ 3.1026,  0.0606,  1.4437, -0.2732]])
loss_seg = tensor(2.9672)   loss_hum = tensor(0.2316)   combined = 3.0251
softmax(seg[0]) = [0.286, 0.238, 0.037, 0.002, 0.08, 0.008, 0.02, 0.011, 0.022,
                   0.012, 0.113, 0.08, 0.03, 0.049, 0.012]   # 15 numbers summing to 1
```
Each row of logits = one untrained opinion about the document; `CrossEntropyLoss`
compares it to the target int; `backward()` turns that number into `.grad` on the
17M LoRA + head params.

### B15. `SentenceTransformer` and `label_pair` output
```
SentenceTransformer(
  (0): Transformer({'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean'})
  (2): Normalize()
)
st.encode(["a", "b"], convert_to_tensor=True, normalize_embeddings=True)
  -> Tensor (2, 384) float32 on cuda:0, each row norm=1.0
torch.dot(enc[0], enc[1]) -> 0.2088...      # that IS semantic_sim's cosine
```
`label_pair(...)` returns the dict that becomes one line of `mirage_labeled.jsonl`:
```json
{
  "t_text": "The cat rested on the cozy mat. It was wonderfully warm and soft.",
  "clauses": [
    {"text": "The cat rested on the cozy mat.",  "start": 0,  "end": 31, "label": "ai_assisted"},
    {"text": " It was wonderfully warm and soft.", "start": 31, "end": 65, "label": "ai_assisted"}
  ],
  "f_ai": 0.5
}
```

### B16. `wandb.Run`
```
run = wandb.init(project="pangram4", name="stage1", config={...})
type(run) -> <class 'wandb.sdk.wandb_run.Run'>
run.config["epochs"] -> 2          # dict-like snapshot of your hyperparams
run.project -> 'pangram4'   run.name -> 'stage1'   run.id -> 'hbjvjppa'
run.step -> 0
run.log({"train/loss": 2.713})  ->  run.step -> 1    # one .log = one step
run.finish()                                          # closes the run, writes files
```
It also creates the `wandb/` folder next to your script — one dir per run, with the
logs, metrics, and config inside.

---

## Part C — The object graph of one training step

```
Dataset (145,671 dicts on disk, Arrow)
   │  DataLoader(batch_size=32, collate=collate, shuffle=RandomSampler)
   ▼
list of 32 row-dicts ──collate──► dict of 4 Tensors on CPU
   │  {k: v.to("cuda")}
   ▼
dict of 4 Tensors on GPU
   │  PangramModel(input_ids, attention_mask)
   │    ├─ PeftModel/Qwen3 (28 layers, LoRA on 7 projections × 28 layers)
   │    ├─ hidden_states[-1]  → (32, 512, 2048)
   │    ├─ last real token per row → (32, 2048)
   │    ├─ segment_head: (32, 2048) @ (2048,15)ᵀ + bias → seg_logits (32, 15)
   │    └─ heads:        (32, 2048) @ (2048,4)ᵀ  + bias → hum_logits (32, 4)
   ▼
CrossEntropyLoss(seg_logits, f_ai_bucket) + 0.25 × CE(hum_logits, is_humanized)
   │  loss.backward()  → .grad appears on the 17.4M trainable params
   │  AdamW.step()     → those params move a little
   │  AdamW.zero_grad()→ grads reset
   ▼
wandb.Run.log({"train/loss": ...})   → dashboard
```
