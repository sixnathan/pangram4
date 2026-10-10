# The mental model — one universe, two hemispheres, four shapes

This is the map to hold in your head. Everything in this repo — every dict, tensor,
list, batch, model, loss — is located somewhere on this map, and nothing is outside it.

---

## 0. The universe: everything is numbers

Two kinds, and only two, matter here:

- **int** — numbers for *counting and identity*: token id `8251`, label `1`,
  char offset `11`, bucket `7`. "Which one?"
- **float** — numbers for *measuring*: weight `-0.008`, activation `0.9648`, loss `2.9672`. "How much?"

A **str** is a locked sequence of bytes, and bytes are numbers.
So: **the entire project is numbers being moved, packed, multiplied, and nudged.**

---

## 1. The two hemispheres

Every number lives in one of two worlds, and the border between them is where
all the interesting code happens.

```
┌─────────────────────────────────────┬─────────────────────────────────────┐
│   PYTHON-OBJECT WORLD               │   RAW-MEMORY WORLD                  │
│                                     │                                     │
│  every value wears a ~28-byte       │  numbers packed back-to-back,       │
│  government ID:                     │  NO headers, NO per-number objects: │
│                                     │                                     │
│  ┌─────────┬────────┬───────┐       │  [785│8251│7578│13│151643│...]     │
│  │ refcount│ type → │ 5303  │       │   8 bytes each, wall to wall        │
│  └─────────┴────────┴───────┘       │                                     │
│  the number is 4 of those bytes;    │  plus ONE small label per grid:     │
│  the rest is bookkeeping            │  shape, dtype, device               │
│                                     │                                     │
│  self-describing, flexible          │  fast, parallel, GPU-able           │
│  slow, memory-hungry                │  dumb, rigid, fast                  │
│                                     │                                     │
│  dicts, lists, tuples, strings,     │  torch.Tensor, Parameters,          │
│  json, Counters, your loops         │  weights, activations, losses       │
└─────────────────────────────────────┴─────────────────────────────────────┘
        │            ▲                          ▲              │
        └── torch.tensor(list_of_lists) ────────┘              │
        └── ds[0] (Arrow bytes → fresh dict) ◄─────────────────┘
        └── .tolist(), .item(), .to("cuda")  ◄─ border crossings,
            this is what collate / ds[0] / .item() ARE
```

Names (variables) are **stickers, not boxes**: `b = a` never copies — it sticks a
second label on the same object. Proven live: `b.append(3)` changed `a` too,
and `ds[0] is ds[0]` is `False` (each call builds a fresh dict from disk bytes).

---

## 2. The four shapes (+ one pipe)

Every container ever written is one of these, or a nesting of these.

### Shape 1 — SHELF (sequence): `list`, `tuple`, `str`
Ordered slots, addressed **by position**.
```
ids = [785, 8251, 7578, 13]
      ┌────┬────┬────┬────┐
ids ─►│ p0 │ p1 │ p2 │ p3 │     the list holds 8-byte ADDRESSES
      └──┬─┴──┬─┴──┬─┴──┬─┘
         ▼    ▼    ▼    ▼            each int is its own 28-byte
      (785)(8251)(7578)(13)          object somewhere else
```
- `list` — mutable shelf. `[a] + [b]` copies addresses into a NEW shelf
  (that's why `b["input_ids"] + b["input_ids"]` *doubles length*, no math).
  `[IGNORE] * 512` = 512 slots all pointing at the SAME -100 (safe; ints never change).
- `tuple` — the same shelf, **locked**. Used in this repo when the shape is fixed
  and meaningful by position: `("I like dogs.", 0, 12)` = (text, start, end).
  Locking makes it hashable → could be a dict key.
- `str` — a locked shelf of bytes: `'The cat sat.'`

### Shape 2 — TABLE (mapping): `dict`, `Counter`, `set`, `Namespace`, `Dataset`
Slots addressed **by name/key** (a hash table). Insertion order is kept (guaranteed
since Python 3.7) but that's a courtesy, not the point — the point is lookup.
```
row = {"input_ids": [...], "f_ai_bucket": 7, ...}
         key ──hash──► slot ──► ► the value
```
- `dict` — the workhorse: dataset rows, collated batches, `label_pair` output,
  `state_dict`, `loss_fn.__dict__`. Keys must be hashable (str/int/tuple yes, list no).
- `Counter` — dict subclass whose missing keys read `0`. `ga & gb` = keep the
  *minimum* count per shared key. That's the char-ngram F1.
- `set` — dict with no values, exists to make `x in s` instant
  (`LoraConfig.target_modules` is one).
- `Namespace` — a dict you access with dots (`args.limit`).
- `Dataset` — a table **outside Python entirely**: Arrow columns on disk,
  memory-mapped; `ds[0]` assembles a fresh dict on demand.

### Shape 3 — GRID (tensor): `torch.Tensor`
Numbers packed in a typed N-dimensional block. A tensor = one label + one pointer:
```
wrapper ─► { shape: (2, 8)   dtype: int64   device: cpu   data_ptr ──┐ }
                                                                    ▼
raw memory: [785│8251│7578│13│151643│151643│151643│151643│785│6534│…]
             16 × 8 bytes = 128 bytes. Nothing else. No headers, no ids.
```
Rules that follow from this design:
- **dtype fixes bytes-per-cell**: int64=8, int32=4, float32=4, bfloat16=2.
  1.74B params × 2 bytes = the 3.5–4 GB checkpoint.
- **slices are views, not copies**: `t[1:3]`'s data_ptr is `t.data_ptr() + 8`.
  Mutate the view, the original changes (proven live).
- `batch['input_ids'][0]` is a view too — collate literally packed each row's
  ints into one grid, and rows are windows into it.
- shape is part of identity: `(2, 512, 2048)` hidden vs `(2, 2048)` gathered
  are different objects with different grids.

### Shape 4 — TREE (objects): `nn.Module`, packages/modules
Things that contain *named* things, recursively. `nn.Module` is literally:
```
self._parameters: dict[str → Parameter]     # leaves
self._modules:    dict[str → Module]        # branches
plus a forward() method
```
```
PangramModel
├── backbone: PeftModel ── LoraModel ── Qwen3ForCausalLM ── Qwen3Model
│                                              └── 28 × DecoderLayer
│                                                    └── lora.Linear
│                                                          ├── base_layer.weight (2048,2048) frozen
│                                                          ├── lora_A (16,2048)  trains
│                                                          └── lora_B (2048,16)  trains (starts zeroed)
├── segment_head: Linear ── {weight (15,2048), bias (15,)}
└── heads: Linear ── {weight (4,2048), bias (4,)}
```
A **tree of tables, whose leaves are grids.** `state_dict()` flattens the tree into
one table: 707 rows of `name → tensor`. `load_model()` = build the tree;
`AdamW` = hold pointers to just the leaves with `requires_grad=True` (17.4M of
1.74B numbers = 1.005%).

### The pipe — CONVEYOR BELT (iterator)
Not a container: a lazy pipe that dispenses one item on demand.
`range`, `re.finditer`, `zip`, `enumerate`, `DataLoader`, `dict.items()`.
```
finditer(text) ──► Match ──► Match ──► Match ──► …   (made when you ask)
DataLoader    ──► batch ──► batch ──► batch ──► …    (32 dicts → 1 dict each)
zip(clauses, E_matrix) pairs two belts item by item
enumerate(belt) adds the position: (0, item), (1, item), …
```
`for x in belt` = keep pulling until empty. That's ALL a for-loop is.

---

## 3. The journey of one sentence (traced live, real Qwen ids)

```
'The cat sat.'                          TEXT — 1 str object
     │  tok(text)
     ▼
[785, 8251, 7578, 13]                   SHELF of int atoms
 785='The' 8251=' cat' 7578=' sat' 13='.'
     │  pad to MAX_LEN + inherit clause label
     ▼
row = {"input_ids":      [785,8251,7578,13,151643,151643,151643,151643],
       "attention_mask": [1,1,1,1,0,0,0,0],
       "token_labels":   [1,1,1,1,-100,-100,-100,-100],     # -100 = "teach nothing"
       "f_ai_bucket": 7, "is_mixed": 0, "is_humanized": 0}
                                          TABLE: 3 shelves + 3 atoms
     │  save_to_disk → Arrow columns on disk (leaves Python-object world)
     │  DataLoader + collate
     ▼
batch["input_ids"] = tensor([[785,8251,7578,13,151643,×4],
                             [785,6534,1948,13,151643,×4]])   GRID (2,8) int64
     │  .to("cuda")  (same grid, GPU memory)
     ▼
Qwen3 = a TREE of grids; hidden_states[-1] → GRID (2, 512, 2048) bf16
     │  gather last real token (mask.sum(1) - 1)
     ▼
(2, 2048)  ──segment_head──►  logits (2, 15)   ─┐
           ──heads──────────►  logits (2, 4)    ├─ CrossEntropyLoss
                                                ▼
                              loss = tensor(2.9672)   ONE atom. The point.
     │  loss.backward()
     ▼
.grad grids appear — same shape as every trainable weight
     │  AdamW.step()
     ▼
17,471,507 numbers move a tiny bit. Epoch over. Again.
```

**The whole project in one sentence: text → shelves of ints → tables → grids →
one scalar → nudge 17.4M floats → repeat.**

---

## 4. The decoder ring — 3 questions for ANY structure

1. **One value or many?**
   One → int/float (atom) or str (locked byte shelf).
2. **If many: addressed by position, by name, or packed?**
   Position → shelf (list mutable / tuple locked).
   Name → table (dict; Counter=0-default; set=no values).
   Packed numbers with a shape → grid (tensor).
   Contains named things recursively → tree (nn.Module).
   Lazy → conveyor (range/finditer/zip/DataLoader).
3. **Who else points at it?** (aliasing) — assignment never copies; `is` asks
   "same object?"; `==` asks "same contents?"

## 5. Complete shape inventory of THIS repo

| Structure | Shape | Made of |
|---|---|---|
| text, clause text | str | bytes |
| clause | `tuple` (text, start, end) | str + 2 ints |
| token ids / mask / token_labels | `list[int]`, len 512 | int atoms |
| dataset row (`ds[0]`) | `dict`: 3×`list[int]` + 3 ints | shelves + atoms |
| Dataset | Arrow table on disk | columns, outside Python |
| collated batch | `dict`: 3–4 `Tensor` | grids (32,512)/(32,) |
| logits | `Tensor` (32,15) and (32,4) | float grids |
| hidden states | `Tensor` (32,512,2048) ×29 | float grids (bf16) |
| loss | `Tensor` 0-dim | one atom |
| weight / grad | `Tensor` (out,in) | float grids; grad same shape |
| the model | `nn.Module` tree | tables → grids at leaves |
| state_dict / checkpoint | `dict[str → Tensor]`, 707 rows | flattened tree |
| grams | `Counter` | table, 0-default |
| LoraConfig / args | config record / `Namespace` | inert tables |
| clauses loop, finditer, DataLoader, zip | conveyors | lazy pipes |
| module imports | module objects | bags of names |

---

## 6. How to read any line of this repo with the map

`torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)`
— "walk the belt of row-tables, pull the `input_ids` shelf from each, make a new
shelf of those shelves, then CROSS THE BORDER into raw-memory world as a grid."

`batch = {k: v.to("cuda") for k, v in batch.items()}`
— "build a new table; for each (name, grid) pair, stick the name on the same
grid but resident in GPU memory."

`clauses.append((text[start:end], start, end))`
— "lock a 3-slot tuple: the byte-shelf slice, and the two int atoms marking where."

`counts[label] += end - start`
— "table lookup by label-string → int atom; add the span length; write back."
