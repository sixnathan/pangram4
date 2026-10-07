# Pangram 4 Replication — Master Plan

Repo: `~/Desktop/pangram` · Hardware: DGX Spark (GB10, 128GB unified, 20 ARM cores, 3TB disk)
Reference: `pangram4.pdf` / `pangram4.txt` (arXiv:2607.27183) · Research: `research/dataset-hunt.md`

## 0. Guiding principles

1. **Methods identical to the paper; scale honest to our hardware.** The paper's own lineage
   (EditLens/Pangram 3) trained 3–24B QLoRA on a single GPU with ~60k examples — we are genuinely
   in that class. Final target backbone: Qwen3-30B-A3B (open-weight MoE, 3B active) via QLoRA.
   Iterate on a 1.5–4B dense model first, scale up only when the pipeline is proven.
2. **The user writes all code; the assistant reviews against the paper, drafts test cases, debugs.**
3. **Eval exists before training.** Every component gets measured against held-out gold the moment
   it exists (see §5), so "is this the right implementation?" is always answerable with a number.
4. **Nothing downstream trusts anything upstream blindly.** The labeler is validated against
   EditLens's released scores before any model training starts.

## 1. Phase 0 — Infrastructure (mostly done)

- [x] `uv` project with torch/transformers/peft/datasets/accelerate/sentence-transformers/sklearn
- [x] Paper fetched and digested; dataset research complete (`research/dataset-hunt.md`)
- [ ] Fix dataset IDs in `scripts/download_data.py` (see corrections table in dataset-hunt.md)
- [ ] Download EditLens test/val CSVs (free, GitHub) → `data/raw/editlens/`
- [ ] **Apply for gated `pangram/editlens_iclr`** (user action, latency days)
- [ ] Download CoAuthor zip (~50MB) → `data/raw/coauthor/`
- [ ] Monitoring: `tensorboard` (zero-account; add to deps) — log per-head losses/metrics
- [ ] Agree git hygiene: commit per working component; `.gitignore` data/ and .venv

## 2. Phase 1 — Build the labeled dataset (the paper's core contribution)

Nothing public contains token labels; we manufacture them exactly as the paper does.

### 2a. `scripts/build_pairs.py` — synthetic mirroring (paper §3.5, EditLens recipe)
- Source human docs (S): wikitext-103 + HC3 human answers + RAID `model=='human'` rows.
  Diversity domains: essays, news-ish, reviews, creative, Q&A (paper Fig. 2 mix).
- Generate T by editing S with llama-server (localhost:8000, OpenAI-compatible) at 3 intensities:
  1. light polish (grammar/style only) → mostly `human` clauses
  2. partial rewrite (reword ~50%, keep structure) → `ai-assisted` heavy
  3. full regeneration from prompt/outline → `ai-generated` heavy
- Keep a prompt file (≥9 templates × 3 intensities; EditLens used 303 prompts — we start smaller,
  rotate to avoid style lock-in). Hold out Llama-family edits for OOD test if available.
- Output: JSONL `{s_text, t_text, intensity, source_dataset, prompt_id}`.
- Target volume: 20k–60k pairs (EditLens: 60k train / 6k test / 2.4k val, cost ~$530).
- **Exit criterion:** spot-check 20 pairs by eye — intensities should feel different.

### 2b. `scripts/labeler.py` — Soft N-Grams Labeling ★ (paper Algorithm 1, §3.5)
Current state: `softgrams.py` holds the P3 reference engine (word n-grams 6–12, embed,
cosine ≥ 0.8, precision-style score) + `splitclauses` (5 bugs to fix). `labeler.py` wraps it:

1. `splitclauses(text) -> [(text, start, end)]` — rule-based v1; offsets into original text.
   Upgrade path: LLM clause-splitting via llama-server (paper found this more robust).
2. `char_ngram_f1(a, b)` + word-phrase precision (from softgrams engine) → lexical L(x,y).
3. `semantic_sim` — MiniLM-L6-v2 embeddings (same as reference impl), cosine.
4. `label_clause(y, S, model)`:
   - best-match search over S (clauses + unions of 2–3 adjacent clauses, for merge/split edits)
   - `L` high (≈ ≥0.8, reference threshold) → `human`
   - `L` low but `E` high (≈ ≥0.7) → `ai_assisted`
   - else → `ai_generated`
   Thresholds: hand-label ~20 clauses, calibrate, record in constants block with provenance note.
5. `label_pair(S, T, model)` → `{t_text, clauses:[{text,start,end,label}], f_ai}` where
   `f_ai = (0.5·C_assisted + C_generated) / total_chars` (character-weighted, paper eq.).
6. `--inspect` colorized output; pytest suite (the 7 invariant tests in chat: identical→all human,
   novel→all generated, one-synonym-swap→assisted, reorder invariance, deletion invariance,
   offset round-trip, empty/single-clause edges).
- **Validation gate:** run on EditLens `test.csv` pairs; our per-pair score must correlate
  strongly with their released `soft_ngrams_score` (report Spearman; investigate <0.8).
- **Exit criterion:** eyeball `--inspect` on ≥10 diverse pairs; labels match your judgment;
  EditLens correlation gate passed; tests green.

### 2c. Humanizer data (paper §3.4, 4-way humanizer head)
- Apply to AI-generated docs (from 2a intensity-3): typo injection, casing changes,
  homoglyph substitution (Cyrillic/Greek lookalikes), synonym-swap chains — paper's listed attacks.
- Label: `humanized_ai`. Incidental corruption (OCR/encoding) excluded by construction.
- Size: ~2–5k docs to start (head is a probe with stop-gradient; needs less data).

### 2d. `scripts/make_dataset.py` — unified schema
- Merge: token-labeled pairs (2b) + binary docs (RAID/HC3; f_ai=1 or 0, tokens inherit doc label)
  + humanized (2c). Per-source explicit label mapping (dataset conventions differ!).
- Schema (Arrow): `{text, input_ids, token_labels(0/1/2), f_ai, f_ai_bucket(0–14),
  is_mixed, is_humanized, source, generator, domain}`.
- f_AI → 15 ordered buckets (paper §4.1). Mixed-authorship target: >15% non-dominant tokens.
- Splits: train/val/test with source-domain stratification. **Eval gold sets are NEVER merged in**
  (SemEval-C, TriBERT, LLMTrace, RoFT, DAMASHA, CoAuthor, EditLens test — see research/).
- **Exit criterion:** dataset loads in one `load_from_disk`; label histograms per source printed
  and sanity-checked (human-heavy sources ↔ f_ai≈0, etc.); no NaNs; tokenization round-trips.

## 3. Phase 2 — Model

### 3a. `scripts/model.py`
- Backbone ladder: (i) Qwen2.5-1.5B or Qwen3-1.7B (iteration), (ii) Qwen3-8B, (iii) Qwen3-30B-A3B
  QLoRA (final). Same code, config-only change.
- LoRA targets: **attention + dense** (paper Table 1: attention-only is worse; adding routed
  experts was catastrophic — FNR 85% — do NOT target experts). r=16–32, α=2r starting point.
- 4 heads (all `nn.Linear` on hidden states):
  - segment head: 15-way, input = h_S (last-position; full-window visibility under causal attention)
  - tokenwise head: 3-way, applied at every position (paper: shared linear, Z = H·Wᵀ)
  - mixed-authorship head: 2-way, h_S
  - humanizer head: 4-way, h_S, **probe** (stop-gradient; loss weight 0.25)
- **Exit criterion:** forward pass on a dummy batch produces correct shapes; parameter count
  printed; LoRA-only trainability verified (frozen backbone check).

### 3b. `scripts/collator.py` — Repeat2 + windowing (paper §4.1 "Repeat2")
- Window: 512 tokens, stride 256 at inference; training samples full windows.
- Repeat2: `x̂ = (x, x)`; labels `ŷ = (ignore×512, y)`; loss only on second copy.
- Inference-time assembly: align logits by source-token offsets, average across overlapping
  windows → document-length 3-way sequence.
- **Exit criterion:** unit test — window of known labels round-trips through collate;
  mask verified (no loss contribution from copy 1).

## 4. Phase 3 — Training (paper Table 2)

- **Stage 1**: fresh LoRA on backbone; segment (15-way) + humanizer (4-way, weight 0.25,
  stop-grad); single-copy inputs. LR ~1e-4 (LoRA-typical), cosine schedule, 1–3 epochs.
- **Stage 2**: merge stage-1 LoRA into base; fresh LoRA; add tokenwise (3-way, Repeat2 inputs)
  + mixed-authorship (2-way); keep segment + humanizer heads training.
- Checkpoint selection by val FNR @ fixed FPR (paper's own criterion).
- **Active-learning round (paper §4.2)**: run inference on held-out human pool → hard negatives
  (human flagged AI) → synthetically mirror → fold into train → retrain stage 1. One round only.
- Practical: bf16, gradient accumulation to taste, `tensorboard` logging of per-head losses;
  keep llama-server memory in mind (ask before stopping — other user's process).
- **Exit criterion per stage:** val AUROC / FPR / FNR trends logged; no train/val divergence;
  humanizer head not degrading main FPR (paper's motivation for the probe design).

## 5. Evaluation protocol (`scripts/eval.py`) — measurable from day one

- Metrics: AUROC; FPR @ operating point; FNR; per-generator breakdown; accuracy-vs-length sweep;
  **boundary accuracy** (predicted transition index vs gold, MAE ± tolerance bands).
- Gold sets (never trained on):
  | Skill tested | Dataset |
  |---|---|
  | binary human/AI | RAID labeled rows, HC3, SemEval subtask A |
  | boundary (tokenwise) | **SemEval subtask C** (word-index transition), TriBERT essays, LLMTrace char intervals, RoFT |
  | adversarial boundary | DAMASHA |
  | real ai-assisted | CoAuthor (char masks) |
  | score calibration | EditLens test (soft_ngrams_score) |
- Baselines to beat/compare: Binoculars, FastDetectGPT, Ghostbuster (RAID leaderboard numbers);
  **crucially: our own backbone+heads *before* training** (untrained baseline = sanity floor).
- Honest expectations: paper's 0.9916 AUROC is on their private benchmark; success = competitive
  numbers on public benchmarks + working boundary detection.

## 6. Phase 4 — Inference & decoding (`scripts/infer.py`)
- v1: windowed inference + overlap-averaged token logits → majority/smoothing → segments.
- v2 (paper-faithful): CRF structured decoding over the 3-class token sequence, confidence from
  CRF marginals; document f_ai from decoded segments. Only after v1 numbers exist.
- CLI: `infer.py --model ckpt --text file.txt --json` output shaped like Pangram's model card
  (segments, ai_assistance_score, humanizer_score, fractions).

## 7. Milestones (dependency-ordered, each with exit criteria above)

| # | Milestone | Depends on |
|---|---|---|
| M0 | env + data downloaded + EditLens application sent | — (now) |
| M1 | `labeler.py` passes tests + EditLens correlation gate | M0 |
| M2 | `build_pairs.py` produces 20k+ validated pairs | M0 (+llama-server) |
| M3 | unified dataset `make_dataset.py` | M1+M2 |
| M4 | model+collator forward-pass checks | — (parallel with M1–M3) |
| M5 | stage-1 trained, val AUROC logged | M3+M4 |
| M6 | stage-2 trained, tokenwise boundary numbers exist | M5 |
| M7 | full eval suite vs baselines | M6 |
| M8 | CRF decoding + active-learning round + scale to 30B backbone | M7 |

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Labeler thresholds wrong → garbage training data | EditLens correlation gate BEFORE any training; hand-labeled calibration set |
| 30B QLoRA OOM/slow on GB10 | ladder approach; 8B results first; llama-server memory negotiation |
| Dataset ID/label drift (DMCA proven) | local copies now; explicit per-source label maps; research/ verified 2026-10-04 |
| Synthetic-only "ai-assisted" doesn't transfer to real co-writing | CoAuthor eval will expose it; Mitigation: consider CoAuthor-derived training data later |
| Humanizer head pollutes main task FPR | probe + stop-gradient exactly as paper; monitor FPR per register |
| Clause splitter too brittle | upgrade path to llama-server splitting; measure label stability when swapping |

## 9. Open decisions (log choices here as made)

- [ ] Clause splitting: rule-based v1 → when to switch to LLM splitting?
- [ ] Semantic model: MiniLM-L6-v2 (reference default) vs stronger embedder
- [ ] Iteration backbone: Qwen2.5-1.5B vs Qwen3-1.7B
- [ ] CRF in v1 or defer
- [ ] Whether/how to use CoAuthor and MIRAGE (`JiachenFu/`) pairs in training vs eval-only
