"""Character n-gram overlap: the lexical similarity measure used by the labeler."""
from collections import Counter


def make_grams(s: str, n: int = 3) -> Counter:
    """Map every length-n character window of s to its occurrence count."""
    s = s.lower()
    if not s:
        return Counter()
    if len(s) < n:
        return Counter([s])
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def f1_from_grams(ga: Counter, gb: Counter) -> float:
    """Dice/F1 overlap of two gram count multisets, in [0.0, 1.0]."""
    shared = sum((ga & gb).values())   # & on Counters: per-key minimum
    total = sum(ga.values()) + sum(gb.values())
    return 2.0 * shared / total if total else 0.0
