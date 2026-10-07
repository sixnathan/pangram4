import argparse
import json

import torch
from sentence_transformers import SentenceTransformer

from softgrams import splitclauses
from chargnramsf1 import charngramf1, make_grams, f1_from_grams

LEX_HUMAN = 0.8
SEM_ASSISTED = 0.7
UNION_DEPTH = 3


def semantic_sim(model, a: str, b: str) -> float:
    ea, eb = model.encode([a, b], convert_to_tensor=True, normalize_embeddings=True)
    return float(torch.dot(ea, eb))
# what is this function doing — it calls the model thing and then torch.dot on ea/eb,
# how does this thing work? is it applying the dot to a tensor of things?


def candidates(s_clauses: list) -> list:
    """Singles + unions of up to UNION_DEPTH adjacent clauses."""
    out = []
    for i in range(len(s_clauses)):
        text = ""
        for j in range(i, min(i + UNION_DEPTH, len(s_clauses))):
            text += s_clauses[j][0]
            out.append(text)
    return out
# what are the candidates doing? for i in range(len) then for j in range —
# it creates some text thing and appends... I don't get what candidates is doing.


def label_clause(model, clause: str, s_clauses: list) -> str:
    if not clause.strip():
        return "human"
    # how is it possible to call a logical operation on a string like this?
    # is it saying: if it is empty then return human?
    cands = candidates(s_clauses)
    if not cands:
        return "ai_generated"
    cands.sort(key=lambda c: charngramf1(clause, c), reverse=True)
    # .sort() on a list takes a key which tells it how to sort? and reverse=True?
    x = cands[0]
    L = charngramf1(clause, x)
    E = semantic_sim(model, clause, x)
    if L >= LEX_HUMAN:
        return "human"
    if E >= SEM_ASSISTED:
        return "ai_assisted"
    return "ai_generated"


def label_pair(model, s_text: str, t_text: str) -> dict:
    t_clauses = splitclauses(t_text)
    s_clauses = splitclauses(s_text)
    cands = candidates(s_clauses)

    # precompute ONCE per pair: grams + embeddings for every candidate
    cand_grams = [make_grams(c) for c in cands]
    cand_emb = model.encode(cands, convert_to_tensor=True, normalize_embeddings=True)

    texts = [c[0] for c in t_clauses]
    clause_emb = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    E_matrix = (clause_emb @ cand_emb.T).tolist()  # cosine sim, every clause × candidate

    clauses = []
    counts = {"human": 0, "ai_assisted": 0, "ai_generated": 0}
    # initialised a dict + empty list; splitclauses we DID define (in softgrams.py);
    # the dict stores total chars per label.
    for (text, start, end), erow in zip(t_clauses, E_matrix):
        if not text.strip():
            label = "human"
        elif not cands:
            label = "ai_generated"
        else:
            g = make_grams(text)
            lex = [f1_from_grams(g, cg) for cg in cand_grams]
            xi = max(range(len(cands)), key=lambda i: lex[i])  # best-match span
            L, E = lex[xi], erow[xi]
            label = "human" if L >= LEX_HUMAN else ("ai_assisted" if E >= SEM_ASSISTED else "ai_generated")
        clauses.append({"text": text, "start": start, "end": end, "label": label})
        counts[label] += end - start
        # what is counts[label] += end - start doing? calling that on the dict?
    total = counts["human"] + counts["ai_assisted"] + counts["ai_generated"]
    f_ai = (0.5 * counts["ai_assisted"] + counts["ai_generated"]) / total if total else 0.0
    return {"t_text": t_text, "clauses": clauses, "f_ai": round(f_ai, 4)}


LABEL_COLORS = {"human": "\033[92m", "ai_assisted": "\033[93m", "ai_generated": "\033[91m"}
RESET = "\033[0m"


def pretty_print(labeled: dict):
    for c in labeled["clauses"]:
        print(f"{LABEL_COLORS[c['label']]}{c['text']}{RESET}", end="")
    print(f"\n  f_ai = {labeled['f_ai']}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs_jsonl")
    ap.add_argument("out_jsonl")
    ap.add_argument("--inspect", action="store_true", help="print colorized labels instead of writing file")
    ap.add_argument("--limit", type=int, default=0, help="only label the first N pairs")
    args = ap.parse_args()

    model = SentenceTransformer("all-MiniLM-L6-v2")

    n = 0
    with open(args.pairs_jsonl) as fin, open(args.out_jsonl, "w") as fout:
        for line in fin:
            row = json.loads(line)
            labeled = label_pair(model, row["s"], row["t"])
            if args.inspect:
                pretty_print(labeled)
            else:
                fout.write(json.dumps({**row, **labeled}) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break
    print(f"labeled {n} pairs")


if __name__ == "__main__":
    main()
