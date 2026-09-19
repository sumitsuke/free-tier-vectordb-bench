"""Embed ArguAna corpus + queries locally on CPU with all-MiniLM-L6-v2.

Saves normalized float32 vectors + id arrays as .npy so every DB adapter
upserts the *identical* vectors (controlled condition, PLAN §5). Records CPU
time as evidence that the whole pipeline runs free + local (RUNBOOK STEP 2).

Usage:
    python -m harness.embed
"""

from __future__ import annotations

import time

import numpy as np

from harness import config, dataset


def embed_texts(model, texts: list[str], label: str) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    vecs = model.encode(
        texts,
        normalize_embeddings=True,  # unit vectors -> cosine == dot product
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    dt = time.perf_counter() - t0
    print(f"[embed] {label}: {len(texts):,} texts -> {vecs.shape} in {dt:.1f}s ({len(texts) / dt:.0f} texts/s)")
    return vecs, dt


def main() -> int:
    from sentence_transformers import SentenceTransformer

    print(f"[load] corpus + queries from {config.DATASET_DIR}")
    corpus = dataset.load_corpus()
    queries = dataset.load_queries()
    print(f"[load] corpus={len(corpus):,} queries={len(queries):,}")

    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[i] for i in corpus_ids]
    query_ids = list(queries.keys())
    query_texts = [queries[i] for i in query_ids]

    print(f"[model] {config.MODEL_NAME} (CPU)")
    model = SentenceTransformer(config.MODEL_NAME, device="cpu")
    model.max_seq_length = config.MAX_SEQ_LEN  # explicit: 256-token truncation
    print(
        f"[model] max_seq_length={model.max_seq_length} "
        f"(long ArguAna docs are truncated -> low absolute nDCG, identical across DBs)"
    )

    corpus_vecs, corpus_dt = embed_texts(model, corpus_texts, "corpus")
    query_vecs, query_dt = embed_texts(model, query_texts, "queries")

    assert corpus_vecs.shape[1] == config.DIM, corpus_vecs.shape
    assert query_vecs.shape[1] == config.DIM, query_vecs.shape

    np.save(config.CORPUS_VECS_NPY, corpus_vecs)
    np.save(config.CORPUS_IDS_NPY, np.array(corpus_ids, dtype=object))
    np.save(config.QUERY_VECS_NPY, query_vecs)
    np.save(config.QUERY_IDS_NPY, np.array(query_ids, dtype=object))

    print(f"[save] {config.CORPUS_VECS_NPY.name} {corpus_vecs.shape}")
    print(f"[save] {config.QUERY_VECS_NPY.name} {query_vecs.shape}")
    print(f"[done] total embed time: {corpus_dt + query_dt:.1f}s (corpus {corpus_dt:.1f}s + queries {query_dt:.1f}s)")
    print(
        f"[stored-dims] corpus {len(corpus_ids):,} x {config.DIM} = "
        f"{len(corpus_ids) * config.DIM:,} (Cloudflare free stored cap = 5,000,000)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
