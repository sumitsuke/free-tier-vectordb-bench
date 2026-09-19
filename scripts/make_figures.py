"""Generate the two article figures (PNG + SVG) from results/ data:
fig1: §7 "律速ユニット" — horizontal bar of each DB's free-tier consumption %,
      colored/labeled by which unit is the bottleneck.
fig2: §6 scale sweep — ANN Recall@10 vs N for ef=64 / ef=16, ceiling dashed.

  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m scripts.make_figures
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# --- Japanese font (Windows-bundled Meiryo) ---
for fp in (r"C:\Windows\Fonts\meiryo.ttc", r"C:\Windows\Fonts\YuGothM.ttc", r"C:\Windows\Fonts\msgothic.ttc"):
    if Path(fp).exists():
        font_manager.fontManager.addfont(fp)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    print(f"[written] {OUT / name}.png / .svg")


def fig1_bottleneck():
    # (DB, consumption %, bottleneck label, is_estimate)
    data = [
        ("Cloudflare", 66.6, "stored次元", False),
        ("Turso", 14.6, "索引肥大(storage)", False),
        ("Supabase", 6.6, "DB容量", False),
        ("Qdrant", 3.8, "RAM(概算・索引バイト非露出)", True),
    ]
    labels = [d[0] for d in data]
    vals = [d[1] for d in data]
    colors = ["#e8743b", "#19a979", "#1f77b4", "#945ecf"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.barh(labels, vals, color=colors, hatch=["", "", "", "////"], edgecolor="white")
    for d, b in zip(data, bars):
        ax.text(
            b.get_width() + 1.2,
            b.get_y() + b.get_height() / 2,
            f"{d[1]:.1f}%  ← 律速: {d[2]}",
            va="center",
            fontsize=11,
        )
    ax.set_xlim(0, 100)
    ax.set_xlabel(
        "無料枠の消費率（同一の8,674件×384次元を投入）"
        "／※各バーは別々の無料枠メーターの%＝大小比較でなく〈どの単位が先に詰まるか〉を見る図"
    )
    ax.set_title(
        "図2: 同じデータ・同じ品質でも「最初に詰まる枠」がDBごとに別物\n"
        "（律速ユニットの非対称・client=JP・2026-06-25実測）",
        fontsize=12,
    )
    ax.invert_yaxis()
    ax.margins(y=0.15)
    fig.tight_layout()
    save(fig, "fig2_律速ユニット")
    plt.close(fig)


def fig2_scale_sweep():
    rows = list(csv.DictReader((ROOT / "results" / "scale_sweep.csv").open(encoding="utf-8")))

    def series(ef):
        pts = sorted((int(r["n"]), float(r["ann_recall10"])) for r in rows if r["ef_search"] == str(ef))
        return [p[0] for p in pts], [p[1] for p in pts]

    x64, y64 = series(64)
    x16, y16 = series(16)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.axhline(1.0, ls="--", color="#888", lw=1, label="厳密top10の天井 (1.000)")
    ax.plot(x64, y64, "o-", color="#19a979", lw=2, label="ef=64（実用ノブ）")
    ax.plot(x16, y16, "s--", color="#e8743b", lw=2, label="ef=16（既定寄り・近似強め）")
    for x, y in zip(x64, y64):
        ax.annotate(
            f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color="#0f6b4a"
        )
    ax.set_xlabel("コーパス件数 N（ローカル faiss 実HNSW・FiQA埋め込み）")
    ax.set_ylabel("ANN Recall@10（厳密top10との重なり）")
    ax.set_ylim(0.88, 1.005)
    ax.set_title(
        "図1: 実HNSWでも規模を上げて崩れない（57,638件まで崖なし）\n"
        "ef=64は0.65ptの単調漸減のみ・efで回復可能／100k超は未検証(外挿)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    save(fig, "fig1_規模スイープ")
    plt.close(fig)


if __name__ == "__main__":
    fig1_bottleneck()
    fig2_scale_sweep()
    print("[done] figures in", OUT)
