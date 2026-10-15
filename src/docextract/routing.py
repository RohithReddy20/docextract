"""Send each evidence window only the schema field groups its text is about.

A schema too large for one prompt is split into field groups, and without routing every group is
asked about every window: a 29-page tax return with a 22-group schema costs 22 requests per window,
most of which find nothing. Routing scores each (window, group) pair by the words that set the group
apart from the other groups (from property names, titles, descriptions and enum values) and keeps the
pairs whose window mentions them. Every group keeps at least its best-matching window, so no field is
ever left unasked, and a window always keeps its best groups. Pure, deterministic, no model.
"""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """the and for with from this that are was were has have had not but all any each per its into
    such than then there their they them what when where which while who whom whose why how also
    other only same both more most some very can may must should would could will shall item items
    value values field fields number amount total name names type list entry entries record records
    data document page line lines box boxes section part form null true false string integer
    boolean array object description optional required format text""".split()
)


def words(text: str) -> set[str]:
    """Lower-case content words: letters/digits runs of three or more characters, no stop words."""
    tokens = re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower().replace("_", " "))
    return {token for token in tokens if len(token) >= 3 and token not in STOPWORDS}


def vocabulary(node) -> set[str]:
    """Words of a (partial) schema: property names, titles, descriptions and enum values."""
    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "properties" and isinstance(value, dict):
                for name, child in value.items():
                    found |= words(name) | vocabulary(child)
            elif key in ("title", "description") and isinstance(value, str):
                found |= words(value)
            elif key in ("enum", "const", "examples"):
                for item in value if isinstance(value, list) else [value]:
                    if isinstance(item, str):
                        found |= words(item)
            else:
                found |= vocabulary(value)
    elif isinstance(node, list):
        for item in node:
            found |= vocabulary(item)
    return found


def route(window_texts: list[str], groups: list[dict], keep_ratio=0.4, share=0.34) -> list[list[int]]:
    """For each window, the indexes of the groups to ask about it (in group order).

    A word is distinctive for a group when at most `share` of the groups use it. A window keeps every
    group scoring at least `keep_ratio` of its best group's score (and always its best group); every
    group is added to its best-scoring window if no window kept it.
    """
    if len(groups) <= 1:
        return [list(range(len(groups))) for _ in window_texts]
    vocabularies = [vocabulary(group) for group in groups]
    counts: dict[str, int] = {}
    for vocab in vocabularies:
        for word in vocab:
            counts[word] = counts.get(word, 0) + 1
    limit = max(1, int(share * len(groups)))
    distinctive = [{word for word in vocab if counts[word] <= limit} for vocab in vocabularies]
    seen = [words(text) for text in window_texts]
    scores = [[len(found & marks) for marks in distinctive] for found in seen]
    chosen: list[set[int]] = []
    for row in scores:
        best = max(row)
        keep = {index for index, score in enumerate(row) if best and score >= max(1, keep_ratio * best)}
        keep.add(row.index(best))
        chosen.append(keep)
    for group in range(len(groups)):
        if not any(group in keep for keep in chosen):
            window = max(range(len(window_texts)), key=lambda index: (scores[index][group], -index))
            chosen[window].add(group)
    return [sorted(keep) for keep in chosen]
