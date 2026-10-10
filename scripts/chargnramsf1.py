from collections import Counter

def make_grams(s: str, n: int = 3) -> Counter:
    s = s.lower()
    if not s:
        return Counter()
    if len(s) < n:
        return Counter([s])
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))

def f1_from_grams(ga: Counter, gb: Counter) -> float:
    shared = sum((ga & gb).values())
    total = sum(ga.values()) + sum(gb.values())
    return 2.0 * shared / total if total else 0.0 


