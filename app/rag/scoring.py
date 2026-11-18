import math
import re
from collections import Counter, deque
from typing import Protocol, Sequence


TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+")


class ForeignKeyLike(Protocol):
    referred_schema: str
    referred_table: str


class ColumnLike(Protocol):
    name: str
    type: str
    description: str
    sample_values: Sequence[str]


class TableLike(Protocol):
    schema_name: str
    name: str
    description: str
    columns: Sequence[ColumnLike]
    foreign_keys: Sequence[ForeignKeyLike]


def tokenize(text: str) -> list[str]:
    """Split Russian/English identifiers (snake_case, camelCase, words) into normalized tokens."""
    if not text:
        return []
    decamelized = re.sub(r"([a-zа-яё])([A-ZА-ЯЁ])", r"\1 \2", text)
    return [tok.lower() for tok in TOKEN_RE.findall(decamelized) if len(tok) >= 2]


def build_table_search_text(table: TableLike) -> str:
    """Construct weighted descriptive text for indexing and lexical matching."""
    parts = [f"{table.schema_name}.{table.name}", table.name]
    if table.description:
        parts.append(table.description)
    for col in table.columns:
        parts.append(col.name)
        if col.description:
            parts.append(col.description)
        if col.sample_values:
            parts.append(" ".join(col.sample_values[:10]))
    for fk in table.foreign_keys:
        parts.append(f"{fk.referred_schema}.{fk.referred_table}")
    return "\n".join(parts)


def rank_lexical(question: str, tables: Sequence[TableLike]) -> list[tuple[int, float]]:
    """Return (index, score) pairs sorted descending by BM25-style lexical relevance."""
    q_tokens = tokenize(question)
    if not q_tokens or not tables:
        return [(idx, 0.0) for idx in range(len(tables))]

    docs_tokens: list[list[str]] = []
    name_tokens_list: list[set[str]] = []
    col_tokens_list: list[set[str]] = []

    for table in tables:
        tokens = tokenize(build_table_search_text(table))
        docs_tokens.append(tokens)
        name_tokens_list.append(set(tokenize(table.name)))
        col_set: set[str] = set()
        for col in table.columns:
            col_set.update(tokenize(col.name))
        col_tokens_list.append(col_set)

    n_docs = len(tables)
    avg_len = sum(len(d) for d in docs_tokens) / max(1, n_docs)
    df = Counter()
    for tokens in docs_tokens:
        df.update(set(tokens))

    k1 = 1.5
    b = 0.75
    scored: list[tuple[int, float]] = []

    for idx, tokens in enumerate(docs_tokens):
        tf = Counter(tokens)
        doc_len = len(tokens) or 1
        score = 0.0
        for q_tok in set(q_tokens):
            # Exact token BM25
            freq = tf.get(q_tok, 0)
            if freq > 0:
                idf = math.log(1 + (n_docs - df[q_tok] + 0.5) / (df[q_tok] + 0.5))
                denom = freq + k1 * (1 - b + b * (doc_len / avg_len))
                score += idf * (freq * (k1 + 1)) / denom
            else:
                # Prefix match helps Russian inflections (e.g. билет/билеты, рейс/рейсов)
                if len(q_tok) >= 4:
                    prefix = q_tok[:4]
                    if any(t.startswith(prefix) for t in tf):
                        score += 0.35

            if q_tok in name_tokens_list[idx]:
                score += 2.0
            if q_tok in col_tokens_list[idx]:
                score += 1.0

        scored.append((idx, score))

    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored


def rrf_merge(ranked_lists: Sequence[Sequence[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion over multiple lists of table indices."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return [idx for idx, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def expand_fk_neighbors(
    seed_indices: Sequence[int],
    tables: Sequence[TableLike],
    top_k: int,
) -> list[int]:
    """Ensure JOIN-bridge and 1-hop FK-referenced tables are included up to top_k."""
    if not tables or top_k <= 0:
        return []

    key_to_idx = {(t.schema_name, t.name): i for i, t in enumerate(tables)}
    adjacency: dict[int, list[int]] = {i: [] for i in range(len(tables))}

    for i, table in enumerate(tables):
        for fk in table.foreign_keys:
            target = key_to_idx.get((fk.referred_schema, fk.referred_table))
            if target is not None and target != i:
                if target not in adjacency[i]:
                    adjacency[i].append(target)
                if i not in adjacency[target]:
                    adjacency[target].append(i)

    selected: list[int] = []
    seen: set[int] = set()

    # Keep high-priority seeds first (leave room for FK neighbors when catalog is large)
    seed_cap = max(1, min(len(seed_indices), top_k))
    queue = deque(seed_indices[:seed_cap])

    for idx in seed_indices[:seed_cap]:
        if idx not in seen and len(selected) < top_k:
            seen.add(idx)
            selected.append(idx)

    while queue and len(selected) < top_k:
        current = queue.popleft()
        for neighbor in adjacency.get(current, []):
            if neighbor not in seen:
                seen.add(neighbor)
                selected.append(neighbor)
                queue.append(neighbor)
                if len(selected) >= top_k:
                    break

    return selected
