"""Latency measurement protocol (PLAN §5-1, §5-3) — ported from the prior work.

Principles that survive peer review:
- discard warm-up trials,
- repeat each query N times; p50/p95 are taken over the POOLED samples
  (all queries x all repeats) -> the tail is dominated by slow queries +
  network jitter, NOT by the repeat count (the caller must describe it that way),
- split e2e (client wall-clock, RTT included) from server-side processing time.

The region confound (client<->DB distance) is the dominant risk, so e2e numbers
are always reported alongside the environment (region, JST time) by the caller.

NOT YET IMPLEMENTED (TODO, recorded honestly — RUNBOOK STEP6/7 / PLAN §5-1,§5-3):
- no-op control window + free-tier meter-diff (consumption module),
- HTTP connect / RTT baseline recording,
- multi-batch variance (CI / boxplot) to show p95 run-to-run stability.
These land in M3/M5; until then the harness reports single-batch e2e/server only.
"""
from __future__ import annotations

import statistics


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile (== numpy default 'linear'; p in [0,100]).

    NOT nearest-rank: it interpolates between the two surrounding samples, so it
    can return a value that is not an observed data point (e.g. p95 of 1..10 = 9.55).
    """
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (p / 100) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def summarize(samples_ms: list[float]) -> dict:
    if not samples_ms:
        return {"n": 0, "p50": float("nan"), "p95": float("nan"),
                "mean": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": len(samples_ms),
        "p50": percentile(samples_ms, 50),
        "p95": percentile(samples_ms, 95),
        "mean": statistics.mean(samples_ms),
        "min": min(samples_ms),
        "max": max(samples_ms),
    }
