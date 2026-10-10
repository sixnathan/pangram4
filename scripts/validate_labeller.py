
import pandas as pd
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from labeler import label_pair

df = pd.read_csv("../data/raw/editlens/val.csv").dropna(subset=["source_text", "text", "soft_ngrams_score"])
model = SentenceTransformer("all-MiniLM-L6-v2")

ours, theirs = [], []
for _, row in df.iterrows():
    labeled = label_pair(model, row["source_text"], row["text"])
    ours.append(labeled["f_ai"])
    theirs.append(row["soft_ngrams_score"])
    if len(ours) % 200 == 0:
        print(f"{len(ours)}/{len(df)}")

rho, p = spearmanr(ours, theirs)
print(f"\nSpearman: {rho:.3f} (p={p:.1e}), n={len(ours)}")
print(f"MAE: {sum(abs(o - t) for o, t in zip(ours, theirs)) / len(ours):.3f}")
print("GATE:", "PASS" if rho >= 0.7 else "FAIL — investigate before training")
