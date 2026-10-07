# Dataset Hunt — Master Summary (2026-10-04)

Full briefs: [hf-dataset-ids.md](hf-dataset-ids.md) · [editlens-data.md](editlens-data.md) · [coauthor-dataset.md](coauthor-dataset.md) · [boundary-datasets.md](boundary-datasets.md)
All IDs verified live via HF/GitHub APIs on this date.

## Our roles (decided)

| Role | Dataset | ID / Link | Notes |
|---|---|---|---|
| **Tokenwise eval (gold)** | SemEval-2024 Task 8 **subtask C** | `d0rj/SemEval2024-task8`, config `"subtaskC"` | label = word index of human→machine transition. 3,649/505/11,123. ⚠️ contains WikiHow rows (DMCA risk) — save local copy |
| **f_AI / score validation** | EditLens test+val | `github.com/pangramlabs/EditLens/tree/main/data` (free CSVs) | `soft_ngrams_score` computed with their exact reference impl — validate our labeler against it |
| **Training pairs (real S/T)** | EditLens full dataset | `huggingface.co/datasets/pangram/editlens_iclr` | gated, free, manual approval — APPLY NOW. 60k train S/T pairs, CC BY-NC-SA |
| **Real ai-assisted eval** | CoAuthor | `cs.stanford.edu/~minalee/zip/chi2022-coauthor-v1.0.zip` | 1,445 sessions, char-level user/api provenance masks. Eval-only (2021-era GPT-3) |
| **Boundary eval #2** | TriBERT hybrid essays | `github.com/douglashiwo/BoundaryDetection` (data.xlsx) | 17,136 essays, sentence labels + boundary indices (AAAI 2024) |
| **Boundary eval #3** | LLMTrace | `iitolstykh/LLMTrace_detection` (HF) | character-level AI intervals, Apache-2.0 |
| **Boundary eval #4** | RoFT | `liamdugan/roft` (HF) or `kirubarajan/roft` (MIT) | gold boundary sentence + 21k human guesses |
| **Adversarial boundary eval** | DAMASHA | `saiteja33/DAMASHA` (HF, CC-BY-4.0) | transition points under adversarial edits |
| **Binary training docs** | RAID | `liamdugan/raid` ✅ | no label col (human ⇔ model=='human'); test split UNLABELED; ~16.7GB — load selectively |
| **Binary training docs** | HC3 | `Hello-SimpleAI/HC3` ✅ config `"all"` | human/chatgpt answer list columns |
| **S/T-style pairs (alt)** | MIRAGE (DetectAnyLLM) | `JiachenFu/MIRAGE` | `original`(human)/`rewritten`(machine) pairs, 93.8k rows, custom license |
| **Bonus span-provenance** | OpAI-Bench | `github.com/VILA-Lab/OpAI-Bench` | token/span provenance under 5 AI edit ops — closest academic analog of our synthetic pipeline |

## Dropped / corrections

- ❌ `YAIFS/semEval2024-Task8`, `bigdateam/...` — 401. Use `d0rj/SemEval2024-task8`
- ❌ `nebula-program/M4` — 404. M4 has no HF repo (GitHub `mbzuai-nlp/M4` only)
- ❌ `yaful/MIRAGE` — doesn't exist; `yaful` org only has `yaful/MAGE` (⚠️ labels REVERSED: 0=machine)
- ⚠️ Label conventions inconsistent across sources — `make_dataset.py` needs an explicit per-source mapping
- ⚠️ Official SemEval GitHub repo DMCA'd 2026-09-14 — keep local copies of everything

## Action items

- [ ] Apply for gated `pangram/editlens_iclr` (approval takes days)
- [ ] Fix `scripts/download_data.py` IDs per corrections above
- [ ] Download EditLens test/val CSVs (no auth) for labeler validation
- [ ] Build `scripts/labeler.py` (P4 Algorithm 1) on top of `softgrams.py`
