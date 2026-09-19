"""Adapter interface every DB implements (PLAN §6).

Contract notes that keep the comparison fair:
- `search` returns (doc_id, score) with score as a *similarity* (higher = better),
  so DBs that natively return a distance must convert. Lets one ranking/metric
  path serve all DBs.
- `search` should retrieve `k` results; the caller passes config.FETCH_K and does
  self-match exclusion + truncation uniformly (not per-adapter).
- `stats` returns whatever free-tier meter the DB exposes (size, rows, dims...),
  recorded as the "before/after" diff for the consumption table (PLAN §8).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

# AWS region token (us-west-2, ap-northeast-1, ...) + city label, so the
# same-region control (Supabase vs Turso both ap-northeast-1) is machine-checkable
# from summary.csv instead of being buried in the free-text --note.
_AWS_CITY = {
    "ap-northeast-1": "Tokyo",
    "us-west-2": "Oregon",
    "us-east-1": "N.Virginia",
    "us-east-2": "Ohio",
    "us-west-1": "N.California",
    "eu-central-1": "Frankfurt",
    "eu-west-1": "Ireland",
    "eu-west-2": "London",
    "sa-east-1": "SaoPaulo",
}


# Anchor to the real AWS region prefixes + token boundaries so we don't grab a
# mid-token run (old r"[a-z]{2}-[a-z]+-\d" matched 'oo-bar-1' inside 'foo-bar-1').
# \d+ (not \d) so 2-digit suffixes aren't truncated; lookarounds = token boundary.
_AWS_REGION_RE = re.compile(r"(?<![a-z0-9])(?:af|ap|ca|eu|il|me|sa|us)-[a-z]+-\d+(?![0-9a-z])")


def aws_region_label(endpoint: str | None) -> str:
    """Extract an AWS region from a host/URL/DSN -> "ap-northeast-1 (Tokyo)".
    Returns "" when no AWS-region-shaped token is present (e.g. a bare
    <db>.turso.io URL) — the caller should treat "" as "region unknown"."""
    if not endpoint:
        return ""
    m = _AWS_REGION_RE.search(endpoint)
    if not m:
        return ""
    r = m.group(0)
    city = _AWS_CITY.get(r)
    return f"{r} ({city})" if city else r


class VectorStore(ABC):
    name: str = "base"
    # The vector-index build strategy, recorded as a structured summary.csv column
    # (turso=incremental-on-empty / pgvector=post-hoc HNSW). Set per adapter.
    upsert_strategy: str = ""
    # AWS region label, set in each adapter's __init__ from its real endpoint.
    region: str = ""

    @abstractmethod
    def create(self, dim: int, metric: str = "cosine") -> None:
        """Create the index/table/collection for `dim` vectors under `metric`."""

    @abstractmethod
    def upsert(self, ids: list[str], vecs, payloads: list[dict] | None = None) -> None:
        """Insert/replace vectors. `vecs` is an (N, dim) array-like."""

    @abstractmethod
    def search(self, qvec, k: int) -> list[tuple[str, float]]:
        """Return [(doc_id, similarity)] best-first, length <= k."""

    @abstractmethod
    def stats(self) -> dict:
        """Free-tier meters (size/rows/dims/etc.) for before/after diffing."""

    def sample_server_ms(self, qvec, k: int) -> float | None:
        """Optional: server-side processing time for ONE query, via a separate
        sampled path (Qdrant REST `time` / pgvector EXPLAIN ANALYZE / CF Workers
        timing). Returns None when unavailable. Enables the e2e-vs-server split
        (§5-1). Sampled (not in the e2e hot loop) and is a different request than
        `search`, so the caller must report it as a separate-path estimate."""
        return None

    @abstractmethod
    def teardown(self) -> None:
        """Drop the index/collection (free up the free tier)."""
