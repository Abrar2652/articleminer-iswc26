from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

FIG = Path(__file__).parent.parent / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Matplotlib configuration for NIPS-quality aesthetics
plt.rcParams.update({
    "font.size": 11, "font.family": "serif",
    "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9, "figure.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "axes.spines.top": False, "axes.spines.right": False,
})

C = {"ours": "#0072B2", "ours2": "#56B4E9", "baseline": "#E69F00",
     "published": "#CC79A7", "accent1": "#009E73", "accent2": "#D55E00"}

fig, ax = plt.subplots(figsize=(7.5, 6.5))

DOM_MARKER = {"ChemTables": "o", "MLTables": "s", "DiSCoMaT": "D"}
SYS_COLOR  = {"Ours-Sonnet":            C["ours"],
              "Ours-Sonnet (best abl.)": C["ours2"],
              "Ours-Opus":               C["accent2"],
              "Few-shot-Sonnet":         "#888888",
              "Ours-Open-7-8B":          "#8B4F9F"} 

# (P, R, domain, system, dx, ha, dy)
_pts_raw = [
    # --- ChemTables ---
    (30.6, 34.8, "ChemTables", "Ours-Sonnet",           10, "left",  0),
    (64.9, 34.0, "ChemTables", "Ours-Sonnet (best abl.)",-10, "right", 0),
    (41.7,  9.7, "ChemTables", "Few-shot-Sonnet",       10, "left",  0),
    # ChemTables Open-source (Split left/right)
    (11.9, 13.2, "ChemTables", "Ours-Open-7-8B",       -10, "right", 0),
    (22.3, 17.7, "ChemTables", "Ours-Open-7-8B",        10, "left",  0),
    (18.6, 14.9, "ChemTables", "Ours-Open-7-8B",       -10, "right", 0),

    # --- MLTables (Central Cluster - alternating left/right) ---
    (52.6, 51.7, "MLTables",   "Ours-Sonnet",           10, "left",  0),  # Moved F1=52.1 to the right
    (57.0, 54.9, "MLTables",   "Ours-Opus",             10, "left",  0),
    (54.8, 35.6, "MLTables",   "Few-shot-Sonnet",       10, "left",  0),
    (49.2, 51.3, "MLTables",   "Ours-Open-7-8B",        10, "left",  0),
    (42.8, 21.5, "MLTables",   "Ours-Open-7-8B",        10, "left",  0),
    (51.9, 50.4, "MLTables",   "Ours-Open-7-8B",       -10, "right", 0),

    # --- DiSCoMaT ---
    (98.2, 66.4, "DiSCoMaT",   "Ours-Sonnet",          -10, "right", 0),
    (97.0, 69.5, "DiSCoMaT",   "Ours-Opus",             10, "left",  0),
    (82.7, 52.9, "DiSCoMaT",   "Few-shot-Sonnet",       10, "left",  0),
    (88.3, 43.0, "DiSCoMaT",   "Ours-Open-7-8B",        10, "left",  0),
    (81.1, 38.2, "DiSCoMaT",   "Ours-Open-7-8B",       -10, "right", 0),
    (79.0, 10.3, "DiSCoMaT",   "Ours-Open-7-8B",        10, "left",  0),
]

def _f1(p, r):
    return (2*p*r)/(p+r) if (p+r) > 0 else 0.0

for p, r, dom, sys_, dx, ha, dy in _pts_raw:
    m = DOM_MARKER[dom]
    c = SYS_COLOR[sys_]
    ax.scatter(r, p, s=160, c=c, marker=m, edgecolors="black", linewidths=0.7, zorder=5)
    
    ax.annotate(f"F$_1$={_f1(p,r):.1f}", 
                xy=(r, p), 
                xytext=(dx, dy),
                textcoords="offset points", 
                fontsize=8, 
                ha=ha, va="center")

# F1 isolines
for fv in [0.3, 0.5, 0.7, 0.9]:
    rr = np.linspace(0.01, 1.0, 100)
    pp = (fv * rr) / (2 * rr - fv)
    v = (pp > 0) & (pp <= 1)
    ax.plot(rr[v]*100, pp[v]*100, "--", color="gray", alpha=0.3, linewidth=0.8)
    if v.any():
        ax.annotate(f"F1={fv:.0%}", xy=(90, (fv*0.9)/(2*0.9-fv)*100),
                    fontsize=8, color="gray", alpha=0.5)

# Legend
handles = [mlines.Line2D([], [], color="white", marker="", label="$\\bf{Domain}$")]
for dom, m in DOM_MARKER.items():
    handles.append(mlines.Line2D([], [], color="black", marker=m, linestyle="None", markersize=8, label=f" {dom}"))
handles.append(mlines.Line2D([], [], color="white", marker="", label="$\\bf{System}$"))
for sys_, c in SYS_COLOR.items():
    handles.append(mlines.Line2D([], [], color=c, marker="s", linestyle="None", markersize=8, label=f" {sys_}"))

ax.legend(handles=handles, loc="lower right", framealpha=0.95, fontsize=8)

ax.set_xlabel("Recall (%)"); ax.set_ylabel("Precision (%)")
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.grid(alpha=0.15, linewidth=0.5)
plt.tight_layout()
plt.savefig(FIG / "precision_recall_pdf_path.pdf")