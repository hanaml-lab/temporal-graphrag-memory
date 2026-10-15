"""Draw every figure of the LEDGER report from data.json, at the size it is printed in the arXiv preprint layout.

The text block is 6.5 in wide. A multi-panel figure is drawn at FULL = 6.5 in and a single panel at WIDE = 4.5 in,
centred; each is included at its natural size, so the fonts print at 8 pt (labels, titles) and 7 pt (ticks,
legends). Every PDF is exactly its figsize (constrained layout, no bbox cropping).

    python make_data.py && python figures.py   # writes figures/*.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
D = json.loads((ROOT / "data.json").read_text())

FULL, WIDE = 6.5, 4.5
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e3e3e3",
    "grid.linewidth": 0.4,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "figure.constrained_layout.h_pad": 0.03,
    "figure.constrained_layout.w_pad": 0.03,
})

# Colour-blind-safe palette (Okabe-Ito), with the lab's blue for LEDGER.
C = {
    "LEDGER": "#2141d6",
    "LEDGER, write-time inval.": "#56B4E9",
    "Graphiti (re-run)": "#D55E00",
    "Full context (128k)": "#7f7f7f",
    "Mem0 (re-run)": "#CC79A7",
    "Turn RAG": "#009E73",
    "Verbalised (reader)": "#D55E00",
    "LEDGER posterior": "#2141d6",
}
SHORT = {
    "LEDGER": "LEDGER",
    "LEDGER, write-time inval.": "LEDGER, write-time inval.",
    "Graphiti (re-run)": "Graphiti",
    "Full context (128k)": "Full context",
    "Mem0 (re-run)": "Mem0",
    "Turn RAG": "Turn RAG",
}
T95 = 2.776  # t quantile, 4 degrees of freedom


def band(rows):
    a = np.array(rows, dtype=float)
    m = a.mean(axis=1)
    h = T95 * a.std(axis=1, ddof=1) / np.sqrt(a.shape[1])
    return m, m - h, m + h


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# Figure 1 (teaser): the order-dependence counterexample, and accuracy against the share of out-of-order ingest.
# The worked example is set in 2024 (a leap year): day 0 is 1 January, 1 March is day 60, 1 June day 152 and
# 1 August day 213.
def teaser():
    fig = plt.figure(figsize=(FULL, 2.3), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0])
    ax = fig.add_subplot(gs[0])
    ax.grid(False)
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    berlin, lisbon = "#E69F00", "#0072B2"
    rows = [
        ("A  said 30 Jan: ``I live in Berlin''", [(0, 30, berlin, "Berlin", True)], None),
        ("B  said 1 Jun: ``moved to Lisbon on 1 Mar''", [(60, 152, lisbon, "Lisbon", True)], None),
        ("C  said 1 Aug: ``back when I lived in Berlin…''", [], 213),
        ("Write-time invalidation, arrival A, B, C", [(0, 60, berlin, "Berlin", False), (60, 213, lisbon, "Lisbon", False), (213, 300, berlin, "Berlin", False)], None),
        ("Write-time invalidation, arrival A, C, B", [(0, 60, berlin, "Berlin", False), (60, 300, lisbon, "Lisbon", False)], None),
        ("LEDGER, any arrival order", [(0, 60, berlin, "Berlin", False), (60, 300, lisbon, "Lisbon", False)], None),
    ]
    n = len(rows)
    for i, (label, bars, mark) in enumerate(rows):
        y = n - 1 - i
        ax.text(-6, y + 0.36, label.replace("``", "“").replace("''", "”"), ha="left", va="bottom", fontsize=7,
                color="#2141d6" if label.startswith("LEDGER") else "#222222", fontweight="bold" if label.startswith("LEDGER") else None)
        for x0, x1, col, name, evidence in bars:
            ax.add_patch(Rectangle((x0, y), x1 - x0, 0.32, facecolor=col, alpha=0.45 if evidence else 0.9, edgecolor="none"))
            if x1 - x0 > 30:
                ax.text((x0 + x1) / 2, y + 0.16, name, ha="center", va="center", fontsize=6.5, color="white" if not evidence else "#222222")
        if mark is not None:
            ax.plot([mark], [y + 0.16], marker="D", color=berlin, markersize=3.8)
            ax.text(mark + 5, y + 0.16, "no date: when did it end?", va="center", fontsize=6.5, color="#555555")
        if i == 3:
            ax.text(300, y + 0.36, "wrong: C reopens Berlin", ha="right", va="bottom", fontsize=6.5, color="#b03000")
            ax.add_patch(Rectangle((213, y - 0.03), 87, 0.38, facecolor="none", edgecolor="#b03000", lw=0.7))
    ax.axhline(n - 3 - 0.15, color="#bbbbbb", lw=0.4)
    months = {0: "Jan", 60: "Mar", 121: "May", 182: "Jul", 244: "Sep", 274: "Oct"}
    ax.set_xticks(list(months), list(months.values()))
    ax.set_yticks([])
    ax.set_xlim(-8, 304)
    ax.set_ylim(-0.25, n - 0.05)
    ax.set_xlabel("Valid time, 2024")
    ax.set_title("(a) Same evidence, different arrival orders", loc="left")

    ax = fig.add_subplot(gs[1])
    pi = np.array(D["order_sweep"]["pi"]) * 100
    order = ["LEDGER", "LEDGER, write-time inval.", "Graphiti (re-run)", "Full context (128k)", "Mem0 (re-run)", "Turn RAG"]
    for name in order:
        m, lo, hi = band(D["order_sweep"]["systems"][name])
        lw = 1.5 if name == "LEDGER" else 0.9
        ls = "--" if name == "LEDGER, write-time inval." else "-"
        ax.plot(pi, m, color=C[name], lw=lw, ls=ls, marker="o", markersize=2.4)
        ax.fill_between(pi, lo, hi, color=C[name], alpha=0.18, lw=0)
        dy = {"LEDGER, write-time inval.": 1.4, "Graphiti (re-run)": 1.1, "Full context (128k)": -1.7}.get(name, 0)
        ax.text(102, m[-1] + dy, SHORT[name] if name != "LEDGER, write-time inval." else "LEDGER, WTI", va="center", fontsize=7, color=C[name])
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Sessions ingested out of order (%)")
    ax.set_ylabel("BiTempQA accuracy (%)")
    ax.set_ylim(25, 82)
    ax.set_title("(b) Accuracy as arrival order departs from event order", loc="left")
    save(fig, "teaser")


# ---------------------------------------------------------------------------------------------------------------
# Figure 2: BiTempQA accuracy by question type (in-order ingest).
def btq_types():
    fig, ax = plt.subplots(figsize=(WIDE, 2.45), layout="constrained")
    types = D["btq"]["types"]
    names = ["Full context (128k)", "Graphiti (re-run)", "LEDGER, write-time inval.", "LEDGER"]
    x = np.arange(len(types))
    w = 0.2
    for i, name in enumerate(names):
        e = D["btq"]["systems"][name]["io"]["types"]
        m = [e[t]["mean"] for t in types]
        sd = [e[t]["sd"] for t in types]
        ax.bar(x + (i - 1.5) * w, m, w, color=C[name], label=SHORT[name], yerr=sd, error_kw=dict(lw=0.5, capsize=1.2), hatch="////" if "inval" in name else None, edgecolor="white" if "inval" in name else None, linewidth=0)
    ax.set_xticks(x, types)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper center", handlelength=1.2, columnspacing=1.0, bbox_to_anchor=(0.5, 1.13))
    save(fig, "btq_types")


# ---------------------------------------------------------------------------------------------------------------
# Figure 3: horizon scaling, modality-tagger noise and anchoring noise.
def robustness():
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.15), layout="constrained")
    ax = axes[0]
    H = np.array(D["horizon"]["H"])
    for name in ["LEDGER", "LEDGER, write-time inval.", "Graphiti (re-run)", "Full context (128k)"]:
        rows = D["horizon"]["curves"][name]
        keep = [i for i, r in enumerate(rows) if r is not None]
        m, lo, hi = band([rows[i] for i in keep])
        ax.plot(H[keep], m, color=C[name], lw=1.5 if name == "LEDGER" else 0.9, ls="--" if "inval" in name else "-", marker="o", markersize=2.2, label=SHORT[name])
        ax.fill_between(H[keep], lo, hi, color=C[name], alpha=0.18, lw=0)
    ax.axvspan(110, 340, color="#f2f2f2", lw=0, zorder=0)
    ax.text(190, 52.5, "history > 128k tokens", fontsize=6.5, color="#666666", ha="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks(H, [str(h) for h in H])
    ax.set_xlim(9, 340)
    ax.set_ylim(50, 84)
    ax.set_xlabel("Sessions in the history")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="lower left", handlelength=1.6)
    ax.set_title("(a) Horizon (BiTempQA-long)", loc="left")

    ax = axes[1]
    rho = np.array(D["robust"]["rho"]) * 100
    for name in ["LEDGER", "LEDGER, write-time inval.", "Graphiti (re-run)"]:
        m, lo, hi = band(D["robust"]["modality"][name])
        ax.plot(rho, m, color=C[name], lw=1.5 if name == "LEDGER" else 0.9, ls="--" if "inval" in name else ("-" if name == "LEDGER" else ":"), marker="o" if name != "Graphiti (re-run)" else None, markersize=2.2)
        ax.fill_between(rho, lo, hi, color=C[name], alpha=0.18, lw=0)
    ax.text(41, D["robust"]["modality"]["Graphiti (re-run)"][0][0] + 1.6, "Graphiti (no tagger)", ha="right", fontsize=7, color=C["Graphiti (re-run)"])
    ax.set_xlabel("Modality tags replaced at random (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(48, 80)
    ax.set_title("(b) Tagger noise (BiTempQA)", loc="left")

    ax = axes[2]
    sig = np.array(D["robust"]["sigma"])
    for name in ["LEDGER", "LEDGER, write-time inval.", "Graphiti (re-run)"]:
        m, lo, hi = band(D["robust"]["anchoring"][name])
        ax.plot(sig, m, color=C[name], lw=1.5 if name == "LEDGER" else 0.9, ls="--" if "inval" in name else "-", marker="o", markersize=2.2, label=SHORT[name])
        ax.fill_between(sig, lo, hi, color=C[name], alpha=0.18, lw=0)
    ax.set_xlabel("Anchoring noise, s.d. (days)")
    ax.set_ylabel("Point-in-time accuracy (%)")
    ax.set_xticks(sig)
    ax.set_ylim(34, 82)
    ax.legend(loc="lower left", handlelength=1.6)
    ax.set_title("(c) Anchoring noise", loc="left")
    save(fig, "robustness")


# ---------------------------------------------------------------------------------------------------------------
# Figure 4: write-path cost.
def latency():
    L = D["latency"]
    fig, ax = plt.subplots(figsize=(WIDE, 2.25), layout="constrained")
    H = np.array(L["H"])
    ax.plot(H, L["incremental_median_ms"], color=C["LEDGER"], lw=1.4, marker="o", markersize=2.2, label="Slot re-solve (median)")
    ax.fill_between(H, L["incremental_median_ms"], L["incremental_p95_ms"], color=C["LEDGER"], alpha=0.16, lw=0, label="to 95th percentile")
    ax.plot(H, L["full_resolve_median_ms"], color="#7f7f7f", lw=0.9, ls="--", marker="s", markersize=2, label="Re-solve every slot")
    ax.axhline(L["extraction_ms"], color="#D55E00", lw=0.8, ls=":")
    ax.text(11, L["extraction_ms"] * 1.25, "LLM extraction of one session", fontsize=7, color="#D55E00")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(H, [str(h) for h in H])
    ax.set_xlabel("Sessions in the history")
    ax.set_ylabel("Write-path time per append (ms)")
    ax.set_ylim(0.1, 6000)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.55), handlelength=1.6)
    save(fig, "latency")


# ---------------------------------------------------------------------------------------------------------------
# Figure 5: calibration, selective answering and sensitivity.
def calibration():
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.2), layout="constrained", gridspec_kw=dict(width_ratios=[1, 1, 1.2]))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.6, ls=":")
    for name in ["LEDGER posterior", "Verbalised (reader)"]:
        c = D["calibration"][name]
        keep = [i for i, k in enumerate(c["n"]) if k >= 30]
        ax.plot(np.array(c["conf"])[keep], np.array(c["acc"])[keep], marker="o", markersize=2.4, lw=1.2 if name.startswith("LEDGER") else 0.9, color=C[name],
                label=f"{'LEDGER posterior' if name.startswith('LEDGER') else 'Verbalised'}, ECE {c['ece']:.3f}")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", handlelength=1.4)
    ax.set_title("(a) Reliability", loc="left")

    ax = axes[1]
    rc = D["risk_coverage"]
    cov = np.array(rc["coverage"]) * 100
    for name in ["LEDGER posterior", "Verbalised (reader)"]:
        ax.plot(cov, rc[name], color=C[name], lw=1.2 if name.startswith("LEDGER") else 0.9, label="Abstain on LEDGER posterior" if name.startswith("LEDGER") else "Abstain on verbalised confidence")
    ax.set_xlabel("Questions answered (%)")
    ax.set_ylabel("Accuracy on answered (%)")
    ax.set_xlim(20, 100)
    ax.set_ylim(70, 100)
    ax.legend(loc="lower left", handlelength=1.4)
    ax.set_title("(b) Selective answering", loc="left")

    ax = axes[2]
    S = D["sensitivity"]
    grid = np.array(S["acc"])
    ax.imshow(grid, cmap="Blues", aspect="auto", vmin=grid.min() - 1, vmax=grid.max())
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=6, color="white" if (grid[i, j] - grid.min() + 1) / (grid.max() - grid.min() + 1) > 0.55 else "#222222")
    ax.set_xticks(range(len(S["lambda"])), [f"{x:g}" for x in S["lambda"]])
    ax.set_yticks(range(len(S["mult"])), [f"{x:g}" for x in S["mult"]])
    ax.set_xlabel("Switch penalty $\\lambda$ (weight-days)")
    ax.set_ylabel("$H_p$ multiplier")
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    ax.set_title("(c) Sensitivity (BiTempQA dev)", loc="left")
    save(fig, "calibration")


# ---------------------------------------------------------------------------------------------------------------
# Figure 6 (appendix): modality-tagger training.
def training():
    T = D["training"]
    steps = np.array(T["steps"])
    fig, ax = plt.subplots(figsize=(WIDE, 2.2), layout="constrained")
    for name, col in (("DeBERTa-v3-large", C["LEDGER"]), ("DeBERTa-v3-base", "#56B4E9")):
        a = np.array(T[name])
        ax.plot(steps, a.mean(0), color=col, lw=1.1, label=name)
        ax.fill_between(steps, a.min(0), a.max(0), color=col, alpha=0.2, lw=0)
    ax.axhline(T["cue rules"], color="#7f7f7f", lw=0.8, ls="--")
    ax.text(2350, T["cue rules"] + 0.012, "cue rules", ha="right", fontsize=7, color="#555555")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Dev macro-F1")
    ax.set_ylim(0.25, 0.92)
    ax.legend(loc="lower right")
    save(fig, "training")


if __name__ == "__main__":
    teaser()
    btq_types()
    robustness()
    latency()
    calibration()
    training()
    print(sorted(p.name for p in OUT.glob("*.pdf")))
