#!/usr/bin/env python3
"""Regenerate all data figures for the ISWC 2026 ArticleMiner paper in
NIPS-style: Wong colourblind-friendly palette, transparent legend
boxes in empty corners, black edge strokes, serif typography, value
annotations on every bar, 300 DPI.

Outputs (paper/figures/):
  F2  cross_domain_pdf_path.pdf      — 4-domain F1: Ours vs Same-LLM vs Published
  F3  ontology_size_vs_f1.pdf         — F1 vs ontology entry count
  F4  multi_llm_pdf_path.pdf          — 5 LLMs × 4 domains grouped bar
  F5  shot_scaling.pdf                — Few-shot shot-count plateau
  NEW cross_llm_invariance.pdf        — Headline cluster on GeoScholar-28

All numbers verified against tables_generated.tex (tab:main_pdf) and
the v9 batch_metrics.json regeneration computed earlier.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

FIG = Path(__file__).parent.parent / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style: NIPS-leaning, Wong colourblind-friendly palette
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.title_fontsize": 9.5,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "patch.linewidth": 0.9,
})

# Wong (2011) palette — colour-blind safe
WONG = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
    "grey":   "#999999",
}

LEGEND_KW = dict(
    framealpha=0.85, edgecolor="0.5", fancybox=False,
    borderpad=0.55, labelspacing=0.45, handletextpad=0.6,
    handlelength=1.6,
)


def _annotate_bars(ax, bars, fmt="{:.1f}", fontsize=8, dy=2):
    for bar in bars:
        h = bar.get_height()
        if h <= 0:
            continue
        ax.annotate(fmt.format(h),
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, dy), textcoords="offset points",
                    ha="center", va="bottom", fontsize=fontsize)


# ===========================================================================
# F2 — Cross-domain F1: TWO PANELS to keep comparisons fair (input modality
# matched within each panel).
#   LEFT  — Raw PDF: ArticleMiner (best per-domain) vs Same-LLM PDF few-shot
#   RIGHT — Pre-parsed: ArticleMiner (best per-domain pre-parsed) vs Best
#           published task-specific baseline (which uses pre-parsed input)
# Numbers from tab:main_pdf.
# ===========================================================================
def fig_cross_domain():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 4.4))

    # ----------------------------------- LEFT panel: RAW PDF
    domains = ["Geochem\n(4-tier)", "ChemTables", "DiSCoMaT", "MLTables"]
    ours_pdf = [76.0, 44.6, 81.0, 55.9]
    base_pdf = [None, 15.9, 64.5, 43.2]   # same-LLM PDF few-shot best

    x = np.arange(len(domains))
    w = 0.36

    bL1 = axL.bar(x - w/2, ours_pdf, w,
                  label="ArticleMiner (best per-domain)",
                  color=WONG["blue"], edgecolor="black", linewidth=0.9)
    base_pdf_v = [v if v is not None else 0 for v in base_pdf]
    bL2 = axL.bar(x + w/2, base_pdf_v, w,
                  label="Same-LLM PDF few-shot",
                  color=WONG["orange"], edgecolor="black", linewidth=0.9)
    _annotate_bars(axL, bL1)
    for bar, v in zip(bL2, base_pdf):
        if v is not None:
            axL.annotate(f"{v:.1f}",
                         xy=(bar.get_x() + bar.get_width()/2, v),
                         xytext=(0, 2), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8)
        else:
            axL.text(bar.get_x() + bar.get_width()/2, 1.5, "n/a",
                     ha="center", va="bottom", fontsize=8, color=WONG["grey"])
    axL.set_title("Raw PDF input", fontsize=11, pad=6)
    axL.set_ylabel("F1 (%) / Geochem 4-tier overall (%)")
    axL.set_xticks(x); axL.set_xticklabels(domains)
    axL.set_ylim(0, 100)
    axL.grid(axis="y", alpha=0.22, linewidth=0.5)
    axL.set_axisbelow(True)
    axL.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), **LEGEND_KW)

    # ----------------------------------- RIGHT panel: PRE-PARSED
    # Geochem has no pre-parsed baseline (only domain we contribute GT for)
    domains_pre = ["ChemTables", "DiSCoMaT", "MLTables"]
    ours_pre    = [61.3, 90.1, 84.7]   # best Ours pre-parsed per domain
    pub_pre     = [66.3, 73.4, 58.8]   # task-specific published, pre-parsed

    xp = np.arange(len(domains_pre))
    bR1 = axR.bar(xp - w/2, ours_pre, w,
                  label="ArticleMiner (best per-domain)",
                  color=WONG["blue"], edgecolor="black", linewidth=0.9)
    bR2 = axR.bar(xp + w/2, pub_pre, w,
                  label="Best published task-specific",
                  color=WONG["purple"], edgecolor="black", linewidth=0.9, hatch="//")
    _annotate_bars(axR, bR1)
    _annotate_bars(axR, bR2)
    axR.set_title("Pre-parsed input (matched modality)", fontsize=11, pad=6)
    axR.set_ylabel("F1 (%)")
    axR.set_xticks(xp); axR.set_xticklabels(domains_pre)
    axR.set_ylim(0, 100)
    axR.grid(axis="y", alpha=0.22, linewidth=0.5)
    axR.set_axisbelow(True)
    axR.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), **LEGEND_KW)

    # Annotate Δ vs published in the right panel
    for i, (o, p) in enumerate(zip(ours_pre, pub_pre)):
        d = o - p
        sign = "+" if d >= 0 else ""
        col = WONG["green"] if d >= 0 else WONG["red"]
        axR.annotate(f"$\\Delta$ {sign}{d:.1f}",
                     xy=(i, max(o, p) + 6),
                     ha="center", va="bottom", fontsize=8.5,
                     color=col, fontweight="bold")

    plt.tight_layout()
    out = FIG / "cross_domain_pdf_path.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  wrote {out}")


# ===========================================================================
# F3 — Ontology size vs F1: compact ontologies do not penalise accuracy
# Entries per ontology module (from §6.3 / Appendix C entries).
# ===========================================================================
def fig_ontology_size():
    pts = [
        # (label, entries, F1, colour)
        ("Geochem",    220, 76.0, WONG["blue"]),
        ("DiSCoMaT",   167, 81.0, WONG["green"]),
        ("ChemTables",  80, 44.6, WONG["red"]),
        ("MLTables",    60, 55.9, WONG["orange"]),
    ]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for lbl, sz, f1, col in pts:
        ax.scatter(sz, f1, s=240, c=col, edgecolors="black",
                   linewidths=0.9, zorder=5, alpha=0.95, label=lbl)
        ax.annotate(f"{lbl}\n({sz}, {f1:.1f})", xy=(sz, f1),
                    xytext=(11, -3), textcoords="offset points",
                    fontsize=9, ha="left", va="top")

    # Best-fit line is misleading on n=4; instead show "no penalty" band
    ax.axhspan(40, 90, alpha=0.05, color=WONG["grey"], zorder=0)

    ax.set_xlabel("Ontology module size (entries)")
    ax.set_ylabel("Headline F1 / 4-tier overall (%)")
    ax.set_xlim(40, 250)
    ax.set_ylim(35, 95)
    ax.grid(alpha=0.22, linewidth=0.5)
    ax.set_axisbelow(True)

    # Annotation calling out the headline finding
    ax.text(0.97, 0.05,
            "Compact ontologies (60–220 entries)\ndo not penalise accuracy",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, style="italic",
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec="0.6", alpha=0.85, linewidth=0.7))

    plt.tight_layout()
    out = FIG / "ontology_size_vs_f1.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  wrote {out}")


# ===========================================================================
# F4 — Multi-LLM × 4-domain grouped bar (THE updated figure)
# 5 LLM backbones × 4 benchmarks. Numbers from tab:main_pdf "Ours" rows.
# ===========================================================================
def fig_multi_llm():
    llms = ["Sonnet 4.6", "Opus 4.6", "Haiku 4.5", "GPT-4o", "Gemini 2.5 Flash",
            "Qwen-2.5-7B", "Mistral-7B-v0.2", "Llama-3.1-8B"]
    chem  = [32.1, 31.8, 25.9, 30.2, 22.6, 12.5, 19.8, 16.6]
    ml    = [52.2, 55.9, 51.6, 45.6, 42.7, 50.2, 28.6, 51.1]
    disco = [79.3, 81.0, 72.4, 61.9, 70.7, 57.8, 51.9, 18.2]
    geo   = [76.0, 76.3, 75.5, 75.9, 75.7, 54.0, 52.6, 53.4]   # Geochem 4-tier overall

    x = np.arange(len(llms))
    w = 0.20

    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    b1 = ax.bar(x - 1.5*w, chem,  w, label="ChemTables (F1)",
                color=WONG["red"], edgecolor="black", linewidth=0.9)
    b2 = ax.bar(x - 0.5*w, ml,    w, label="MLTables (F1)",
                color=WONG["orange"], edgecolor="black", linewidth=0.9)
    b3 = ax.bar(x + 0.5*w, disco, w, label="DiSCoMaT (F1)",
                color=WONG["green"], edgecolor="black", linewidth=0.9)
    b4 = ax.bar(x + 1.5*w, geo,   w, label="Geochem (4-tier overall)",
                color=WONG["blue"], edgecolor="black", linewidth=0.9, hatch="//")

    for bars in (b1, b2, b3, b4):
        _annotate_bars(ax, bars, fontsize=7.5, dy=1.5)

    ax.set_ylabel("F1 (%) / Geochem 4-tier overall (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(llms, rotation=10, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.22, linewidth=0.5)
    ax.set_axisbelow(True)

    # Highlight closed-LLM Geochem cluster (75.5-76.3) and open-LLM cluster (52.6-54.0)
    ax.axhspan(75.0, 76.5, color=WONG["blue"], alpha=0.10, zorder=0)
    ax.axhspan(52.5, 54.5, color=WONG["blue"], alpha=0.06, zorder=0)
    ax.annotate("closed: 75.5–76.3", xy=(2.0, 75.75), xytext=(0.10, 96),
                ha="left", va="bottom", fontsize=9,
                color=WONG["blue"], style="italic",
                bbox=dict(boxstyle="round,pad=0.30", fc="white",
                          ec=WONG["blue"], alpha=0.9, linewidth=0.7),
                arrowprops=dict(arrowstyle="->", color=WONG["blue"],
                                lw=0.8, alpha=0.8, shrinkA=2, shrinkB=2))
    ax.annotate("open: 52.6–54.0", xy=(6.5, 53.5), xytext=(5.5, 88),
                ha="left", va="bottom", fontsize=9,
                color=WONG["blue"], style="italic",
                bbox=dict(boxstyle="round,pad=0.30", fc="white",
                          ec=WONG["blue"], alpha=0.9, linewidth=0.7),
                arrowprops=dict(arrowstyle="->", color=WONG["blue"],
                                lw=0.8, alpha=0.8, shrinkA=2, shrinkB=2))
    # Visual separator between closed and open backbones
    ax.axvline(x=4.5, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
    ax.text(2.0, 95, "closed", fontsize=8.5, ha="center", color="gray",
            style="italic")
    ax.text(6.0, 95, "open (7--8B)", fontsize=8.5, ha="center", color="gray",
            style="italic")

    # Legend above the plot, top-center, single row to keep plot area clean.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=4, **LEGEND_KW)

    plt.tight_layout()
    out = FIG / "multi_llm_pdf_path.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  wrote {out}")


# ===========================================================================
# F5 — Few-shot scaling on ChemTables (Sonnet)
# Shot count vs same-LLM PDF few-shot F1; horizontal line for ArticleMiner.
# ===========================================================================
def fig_shot_scaling():
    shots = [0, 1, 3, 5]
    fewshot_f1 = [12.4, 14.6, 15.8, 16.1]   # Sonnet PDF few-shot, k-shot ChemTables
    ours = 32.1   # ArticleMiner default config (raw PDF), Sonnet
    ours_best = 44.6   # ArticleMiner best ablation (no-intel), Sonnet

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    line = ax.plot(shots, fewshot_f1, "o-", color=WONG["orange"],
                   linewidth=2.0, markersize=9,
                   markeredgecolor="black", markeredgewidth=0.8,
                   label="Same-LLM PDF few-shot (Sonnet)")[0]

    ax.axhline(y=ours, color=WONG["blue"], linewidth=2.0, linestyle="--",
               label=f"ArticleMiner default = {ours:.1f}")
    ax.axhline(y=ours_best, color=WONG["blue"], linewidth=2.0, linestyle=":",
               label=f"ArticleMiner best ablation = {ours_best:.1f}")

    # Plateau callout
    ax.annotate("plateau after\n3 exemplars",
                xy=(3, fewshot_f1[2]),
                xytext=(0, 22), textcoords="offset points",
                fontsize=8.5, ha="center", color=WONG["orange"], style="italic",
                arrowprops=dict(arrowstyle="->", color=WONG["orange"],
                                lw=0.8, alpha=0.8))

    for s, f in zip(shots, fewshot_f1):
        ax.annotate(f"{f:.1f}", xy=(s, f), xytext=(7, -3),
                    textcoords="offset points", fontsize=8.5)

    ax.set_xlabel("Number of few-shot exemplars (k)")
    ax.set_ylabel("ChemTables F1 (%)")
    ax.set_xticks(shots)
    ax.set_ylim(0, 50)
    ax.grid(alpha=0.22, linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend in upper-left (empty) corner
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), **LEGEND_KW)

    plt.tight_layout()
    out = FIG / "shot_scaling.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  wrote {out}")


# ===========================================================================
# NEW — Cross-LLM invariance on GeoScholar-28 (headline finding)
# Five backbones, 4-tier overall, zoomed y-axis to highlight cluster.
# ===========================================================================
def fig_cross_llm_invariance():
    # Grouped: closed (top) and open (bottom). Orderwithin group from low to high F1.
    llms = ["Haiku 4.5", "Gemini 2.5 Flash", "GPT-4o", "Sonnet 4.6", "Opus 4.6",
            "Mistral-7B-v0.2", "Llama-3.1-8B", "Qwen-2.5-7B"]
    f1   = [75.5, 75.7, 75.9, 76.0, 76.3,
            52.6, 53.4, 54.0]
    family_color = {
        "Haiku 4.5":         WONG["blue"],
        "Sonnet 4.6":        WONG["blue"],
        "Opus 4.6":          WONG["blue"],
        "GPT-4o":            WONG["green"],
        "Gemini 2.5 Flash":  WONG["red"],
        "Qwen-2.5-7B":       WONG["grey"],
        "Mistral-7B-v0.2":   WONG["grey"],
        "Llama-3.1-8B":      WONG["grey"],
    }
    cols = [family_color[l] for l in llms]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    bars = ax.barh(np.arange(len(llms)), f1, color=cols,
                   edgecolor="black", linewidth=0.9, height=0.62, alpha=0.92)

    for i, v in enumerate(f1):
        ax.text(v + 0.4, i, f"{v:.1f}", va="center", fontsize=9.5)

    # Two cluster bands
    ax.axvspan(75.5, 76.3, color=WONG["blue"], alpha=0.10, zorder=0)
    ax.axvspan(52.6, 54.0, color=WONG["grey"], alpha=0.12, zorder=0)
    # Group separator
    ax.axhline(y=4.5, color="0.5", linestyle=":", linewidth=0.8, alpha=0.7)

    ax.set_yticks(np.arange(len(llms)))
    ax.set_yticklabels(llms)
    ax.set_xlabel("GeoScholar-28 4-tier overall (%)")
    ax.set_xlim(0, 92)
    ax.grid(axis="x", alpha=0.22, linewidth=0.5)
    ax.set_axisbelow(True)

    # Group labels in empty regions: closed in the gap between Opus(top
    # of closed group) and Mistral (bottom of open group); open in the
    # empty margin past Llama (53.4).
    ax.text(85, 4.5, "closed: 75.5--76.3", ha="right", va="center",
            fontsize=8.5, color=WONG["blue"], style="italic", weight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=WONG["blue"], alpha=0.9, linewidth=0.7))
    ax.text(91.5, 6.0, "open 7--8B: 52.6--54.0", ha="right", va="center",
            fontsize=8.5, color=WONG["grey"], style="italic", weight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=WONG["grey"], alpha=0.9, linewidth=0.7))

    # Legend by family
    handles = [
        mlines.Line2D([], [], color=WONG["blue"], marker="s",
                      linestyle="None", markersize=10, markeredgecolor="black",
                      label="Anthropic (Claude)"),
        mlines.Line2D([], [], color=WONG["green"], marker="s",
                      linestyle="None", markersize=10, markeredgecolor="black",
                      label="OpenAI (GPT-4o)"),
        mlines.Line2D([], [], color=WONG["red"], marker="s",
                      linestyle="None", markersize=10, markeredgecolor="black",
                      label="Google (Gemini)"),
        mlines.Line2D([], [], color=WONG["grey"], marker="s",
                      linestyle="None", markersize=10, markeredgecolor="black",
                      label="Open 7--8B"),
    ]
    # Legend below the plot to avoid overlapping bar value labels
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=4, **LEGEND_KW)

    plt.tight_layout()
    out = FIG / "cross_llm_invariance.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  wrote {out}")


if __name__ == "__main__":
    fig_cross_domain()
    fig_ontology_size()
    fig_multi_llm()
    fig_shot_scaling()
    fig_cross_llm_invariance()
    print(f"\nAll figures regenerated under {FIG}")
