"""Load ArguAna corpus / queries / qrels from the extracted BEIR files.

BEIR format:
  corpus.jsonl   -> {"_id", "title", "text", "metadata"}
  queries.jsonl  -> {"_id", "text", "metadata"}
  qrels/test.tsv -> header row then: query-id <tab> corpus-id <tab> score
"""
from __future__ import annotations

import json

from harness import config


def doc_text(title: str, text: str) -> str:
    """Preprocessing policy: title + text concatenation (BEIR convention).

    Recorded in BUILD_LOG. Used identically for every DB so it does not affect
    cross-DB comparison.
    """
    title = (title or "").strip()
    text = (text or "").strip()
    return f"{title} {text}".strip() if title else text


def load_corpus() -> dict[str, str]:
    """Return {doc_id: concatenated_text}, insertion-ordered."""
    corpus: dict[str, str] = {}
    with open(config.CORPUS_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            corpus[obj["_id"]] = doc_text(obj.get("title", ""), obj.get("text", ""))
    return corpus


def load_queries() -> dict[str, str]:
    """Return {query_id: text}, insertion-ordered."""
    queries: dict[str, str] = {}
    with open(config.QUERIES_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            queries[obj["_id"]] = (obj.get("text", "") or "").strip()
    return queries


def load_qrels() -> dict[str, dict[str, int]]:
    """Return {query_id: {doc_id: relevance}} from qrels/test.tsv."""
    qrels: dict[str, dict[str, int]] = {}
    with open(config.QRELS_TSV, encoding="utf-8") as f:
        header = f.readline()  # skip "query-id\tcorpus-id\tscore"
        _ = header
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid, did, score = line.split("\t")
            qrels.setdefault(qid, {})[did] = int(score)
    return qrels


if __name__ == "__main__":
    corpus = load_corpus()
    queries = load_queries()
    qrels = load_qrels()
    print(f"corpus  : {len(corpus):,} docs")
    print(f"queries : {len(queries):,}")
    print(f"qrels   : {len(qrels):,} queries with judgments")
    # quick sanity: query ids ARE present in the corpus -> that's the self-match
    # (verified: ~1298/1406 query ids appear as corpus ids; raw top-1 is self 91%).
    overlap = len(set(queries) & set(corpus))
    print(f"query-id ∩ corpus-id : {overlap} (these are the self-matches to exclude)")
