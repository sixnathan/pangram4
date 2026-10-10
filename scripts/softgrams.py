"""Clause segmentation: split text into clauses at punctuation runs."""
import re


def splitclauses(text: str) -> list[tuple[str, int, int]]:
    """Split text at runs of [.!?;:] into (clause_text, start, end) tuples.

    Each clause includes its trailing punctuation; clauses are adjacent and
    cover the whole text. A trailing segment without punctuation is appended.
    """
    if not text:
        return []
    breakre = re.compile(r"[.!?;:]+")
    clauses = []
    start = 0
    for m in breakre.finditer(text):
        end = m.end()
        clauses.append((text[start:end], start, end))
        start = end
    if start < len(text):
        clauses.append((text[start:], start, len(text)))
    return clauses
