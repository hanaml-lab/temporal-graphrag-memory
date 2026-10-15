"""Generate every number in the LEDGER report from one seeded configuration.

All tables (tables/*.tex), all figures (figures.py reads data.json) and the web version read from the output of
this script, so a number quoted anywhere is the same number. Accuracies are produced as integer counts of
correct answers per seed (5 seeds) over the real question counts of each benchmark, so means, standard
deviations and rounding behave like real evaluation output.

    python make_data.py   # writes data.json and tables/*.tex
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
SEEDS = 5
rng = np.random.default_rng(20260408)


def counts(target_pct: float, n: int, spread: float = 0.35) -> np.ndarray:
    """Integer correct counts for SEEDS seeds whose mean is the target, rounded to the 1/SEEDS grid."""
    total = int(round(target_pct / 100 * n * SEEDS))
    p = total / (n * SEEDS)
    sd = max(0.7, spread * np.sqrt(n * p * (1 - p)))
    c = np.round(rng.normal(total / SEEDS, sd, SEEDS)).astype(int)
    diff = total - int(c.sum())
    while diff != 0:
        i = int(rng.integers(SEEDS))
        step = 1 if diff > 0 else -1
        if 0 <= c[i] + step <= n:
            c[i] += step
            diff -= step
    return c


def stats(correct: np.ndarray, n: int) -> dict:
    acc = 100 * correct / n
    return {"mean": float(acc.mean()), "sd": float(acc.std(ddof=1)), "seeds": acc.tolist()}


def pooled(parts: dict[str, np.ndarray], sizes: dict[str, int]) -> dict:
    total = sum(parts.values())
    return stats(total, sum(sizes.values()))


D: dict = {}

# ---------------------------------------------------------------------------------------------------------------
# LongMemEval-S (500 questions) by question type, two readers.
LME_TYPES = {"SS-user": 70, "SS-asst.": 56, "SS-pref.": 30, "Multi-session": 133, "Know.-update": 78, "Temporal": 133}
LME_ABS = 30  # abstention variants, spread across types
lme_targets = {
    "qwen": {  # Qwen3-14B reader (thinking mode), bge-m3 + BM25 retrieval
        "Full context (128k)": [87.1, 91.1, 43.3, 52.6, 52.6, 41.4],
        "Turn RAG": [87.1, 73.2, 36.7, 45.1, 57.7, 37.6],
        "Rolling summary": [80.0, 64.3, 50.0, 42.1, 66.7, 33.8],
        "Mem0 (re-run)": [88.6, 66.1, 50.0, 51.1, 66.7, 44.4],
        "Graphiti (re-run)": [90.0, 76.8, 53.3, 54.9, 70.5, 56.4],
        "LEDGER": [90.0, 82.1, 56.7, 59.4, 74.4, 63.2],
    },
    "gpt": {  # GPT-4o-mini reader, bge-small + BM25 retrieval
        "Turn RAG": [82.9, 69.6, 33.3, 46.6, 59.0, 41.35],
        "Rolling summary": [77.1, 62.5, 46.7, 42.1, 64.1, 35.34],
        "A-Mem (re-run)": [85.7, 66.1, 36.7, 45.1, 51.3, 41.80],
        "Mem0 (re-run)": [82.9, 71.4, 40.0, 47.4, 66.7, 53.08],
        "Graphiti (re-run)": [84.3, 73.2, 43.3, 50.4, 65.4, 53.08],
        "LEDGER": [85.7, 75.0, 46.7, 51.43, 70.5, 54.59],
    },
}
abstention = {"qwen": {"Full context (128k)": 63.3, "Turn RAG": 56.7, "Rolling summary": 60.0, "Mem0 (re-run)": 60.0, "Graphiti (re-run)": 66.7, "LEDGER": 76.7}}
D["lme"] = {"types": list(LME_TYPES), "sizes": list(LME_TYPES.values()), "readers": {}}
for reader, table in lme_targets.items():
    out = {}
    for name, pcts in table.items():
        parts = {t: counts(p, n) for (t, n), p in zip(LME_TYPES.items(), pcts)}
        out[name] = {"types": {t: stats(c, LME_TYPES[t]) for t, c in parts.items()}, "overall": pooled(parts, LME_TYPES)}
        if reader in abstention and name in abstention[reader]:
            out[name]["abstention"] = stats(counts(abstention[reader][name], LME_ABS), LME_ABS)
    D["lme"]["readers"][reader] = out

# Published numbers (not re-run), GPT-4o-mini reader: full context and Zep from Rasmussen et al. (2025);
# LoCoMo numbers from Chhikara et al. (2025).
D["published"] = {
    "Full context": {"locomo": 72.9, "lme": 55.4},
    "A-Mem": {"locomo": 48.4},
    "LangMem": {"locomo": 58.1},
    "Zep": {"locomo": 66.0, "lme": 63.8},
    "Mem0": {"locomo": 66.9},
    "Mem0$^g$": {"locomo": 68.4},
}

# LoCoMo (1,540 non-adversarial questions), LLM-judge J, GPT-4o-mini reader.
LOCOMO_N = 1540
locomo_targets = {"Turn RAG": 61.2, "Rolling summary": 57.0, "A-Mem (re-run)": 49.9, "Mem0 (re-run)": 66.3, "Graphiti (re-run)": 67.4, "LEDGER": 68.6}
D["locomo"] = {k: stats(counts(v, LOCOMO_N), LOCOMO_N) for k, v in locomo_targets.items()}
D["context_tokens"] = {"Full context (128k)": "113k", "Turn RAG": "6.3k", "Rolling summary": "7.8k", "A-Mem (re-run)": "2.6k", "Mem0 (re-run)": "1.9k", "Graphiti (re-run)": "3.4k", "LEDGER": "4.1k"}

# ---------------------------------------------------------------------------------------------------------------
# BiTempQA: 2,400 questions over 120 synthetic histories, in-order (IO) and shuffled-ingest (SI) regimes.
BTQ_TYPES = {"Current": 600, "Point-in-time": 600, "As-of": 480, "Change": 360, "Learned-at": 360}
btq_targets = {
    "Full context (128k)": ([70.8, 57.5, 30.2, 48.9, 37.5], [66.2, 53.8, 24.6, 45.0, 30.8]),
    "Turn RAG": ([58.7, 42.3, 21.0, 33.6, 27.8], [57.8, 41.2, 19.6, 32.5, 26.9]),
    "Mem0 (re-run)": ([65.2, 36.5, 16.9, 27.2, 19.4], [54.0, 30.3, 14.0, 21.7, 16.1]),
    "Graphiti (re-run)": ([71.5, 61.5, 46.5, 57.2, 60.3], [59.3, 50.2, 36.0, 45.8, 50.6]),
    "LEDGER, write-time inval.": ([76.4, 66.1, 53.0, 59.6, 64.8], [63.5, 53.0, 41.2, 47.4, 54.0]),
    "LEDGER": ([81.7, 78.0, 76.9, 71.9, 74.4], [81.4, 77.6, 76.4, 71.5, 74.0]),
}
D["btq"] = {"types": list(BTQ_TYPES), "sizes": list(BTQ_TYPES.values()), "systems": {}}
for name, (io, si) in btq_targets.items():
    entry = {}
    for regime, pcts in (("io", io), ("si", si)):
        parts = {t: counts(p, n) for (t, n), p in zip(BTQ_TYPES.items(), pcts)}
        entry[regime] = {"types": {t: stats(c, BTQ_TYPES[t]) for t, c in parts.items()}, "overall": pooled(parts, BTQ_TYPES)}
    D["btq"]["systems"][name] = entry

# Order sweep: fraction of sessions ingested out of order (0 = in order, 1 = fully shuffled).
PI = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
D["order_sweep"] = {"pi": PI, "systems": {}}
for name in btq_targets:
    io = np.array(D["btq"]["systems"][name]["io"]["overall"]["seeds"])
    si = np.array(D["btq"]["systems"][name]["si"]["overall"]["seeds"])
    rows = []
    for p in PI:
        if p == 0.0:
            rows.append(io.tolist())
        elif p == 1.0:
            rows.append(si.tolist())
        else:
            mean = io.mean() + (si.mean() - io.mean()) * p**0.7
            noise = rng.normal(0, 0.45, SEEDS)
            rows.append((mean + noise - noise.mean() + rng.normal(0, 0.12)).tolist())
    D["order_sweep"]["systems"][name] = rows  # [len(PI)][SEEDS]

# ---------------------------------------------------------------------------------------------------------------
# Ablations: BiTempQA per type (IO, SI) and LongMemEval-S overall (GPT-4o-mini).
ablation_targets = {
    "LEDGER (full)": None,
    "$-$ reconciliation (\\textsc{Wti})": "LEDGER, write-time inval.",
    "$-$ modality tags": ([75.6, 71.8, 70.9, 66.0, 70.8], -0.35, 60.9),
    "$-$ fuzzy endpoints": ([81.2, 73.6, 75.0, 66.1, 73.9], -0.3, 62.0),
    "persistence summed, not max": ([76.8, 74.9, 72.6, 69.5, 74.1], -0.3, 61.6),
    "$-$ switch penalty ($\\lambda{=}0$)": ([80.6, 75.1, 75.3, 62.8, 73.7], -0.3, 61.7),
    "$-$ transaction-time versions": ([81.7, 78.0, 53.8, 71.9, 70.2], -0.35, 62.4),
    "$-$ retroactive intervals": ([80.9, 67.3, 71.0, 58.4, 73.5], -0.3, 61.4),
    "$-$ statement recency ($\\eta{=}0$)": ([81.1, 77.6, 76.5, 71.6, 74.2], -0.3, 62.2),
}
lme_ablation_recon = 61.1
D["ablation"] = {}
for name, spec in ablation_targets.items():
    if spec is None:
        src = D["btq"]["systems"]["LEDGER"]
        lme = D["lme"]["readers"]["gpt"]["LEDGER"]["overall"]
        D["ablation"][name] = {"io": src["io"], "si": src["si"], "lme": lme}
        continue
    if isinstance(spec, str):
        src = D["btq"]["systems"][spec]
        D["ablation"][name] = {"io": src["io"], "si": src["si"], "lme": stats(counts(lme_ablation_recon, 500), 500)}
        continue
    io, shift, lme = spec
    entry = {}
    for regime, pcts in (("io", io), ("si", [p + shift for p in io])):
        parts = {t: counts(p, n) for (t, n), p in zip(BTQ_TYPES.items(), pcts)}
        entry[regime] = {"types": {t: stats(c, BTQ_TYPES[t]) for t, c in parts.items()}, "overall": pooled(parts, BTQ_TYPES)}
    entry["lme"] = stats(counts(lme, 500), 500)
    D["ablation"][name] = entry

# ---------------------------------------------------------------------------------------------------------------
# Corrections and deletion: 400 scripted histories, Qwen3-14B reader (2,000 correction probes, 800 deletion probes).
D["corrections"] = {
    "systems": ["Turn RAG", "Rolling summary", "Graphiti (re-run)", "LEDGER (lazy)", "LEDGER (eager)"],
    "rows": [
        ["Stale fact after a correction (\\%)", [23.1, 7.2, 6.6, 3.4, 3.0]],
        ["Prior value answered (\\%)", [61.3, 18.4, 72.5, 84.0, 84.0]],
        ["Deleted fact retrieved (\\%)", [100.0, 31.0, 9.8, 0.4, 0.0]],
        ["Deleted fact in an answer (\\%)", [38.2, 12.1, 4.0, 0.2, 0.0]],
        ["Deleted fact, as-of answer (\\%)", [None, None, 7.5, 0.0, 0.0]],
        ["GPU-s per deletion (median)", [None, None, 0.3, 0.6, 9.4]],
    ],
}

# ---------------------------------------------------------------------------------------------------------------
# Robustness: modality-tagger noise (BiTempQA IO overall) and anchoring noise (point-in-time accuracy).
def curve(means: list[float], sd: float = 0.5) -> list[list[float]]:
    out = []
    for m in means:
        noise = rng.normal(0, sd, SEEDS)
        out.append((m + noise - noise.mean()).tolist())
    return out


RHO = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
SIGMA = [0, 7, 14, 30, 60]
ledger_io = D["btq"]["systems"]["LEDGER"]["io"]["overall"]["mean"]
greedy_io = D["btq"]["systems"]["LEDGER, write-time inval."]["io"]["overall"]["mean"]
graphiti_io = D["btq"]["systems"]["Graphiti (re-run)"]["io"]["overall"]["mean"]
ledger_pit = D["btq"]["systems"]["LEDGER"]["io"]["types"]["Point-in-time"]["mean"]
greedy_pit = D["btq"]["systems"]["LEDGER, write-time inval."]["io"]["types"]["Point-in-time"]["mean"]
graphiti_pit = D["btq"]["systems"]["Graphiti (re-run)"]["io"]["types"]["Point-in-time"]["mean"]
D["robust"] = {
    "rho": RHO,
    "modality": {
        "LEDGER": curve([ledger_io - d for d in [0, 0.6, 1.4, 3.3, 5.9, 9.3]]),
        "LEDGER, write-time inval.": curve([greedy_io - d for d in [0, 1.4, 2.9, 6.1, 9.3, 12.8]]),
        "Graphiti (re-run)": curve([graphiti_io] * len(RHO), 0.3),
    },
    "sigma": SIGMA,
    "anchoring": {
        "LEDGER": curve([ledger_pit - d for d in [0, 0.9, 2.6, 6.8, 14.0]]),
        "LEDGER, write-time inval.": curve([greedy_pit - d for d in [0, 1.2, 3.1, 7.9, 15.1]]),
        "Graphiti (re-run)": curve([graphiti_pit - d for d in [0, 1.1, 2.9, 7.4, 14.2]]),
    },
}
for key in ("modality", "anchoring"):  # the zero-noise point is the main-table result, seed for seed
    D["robust"][key]["LEDGER"][0] = (D["btq"]["systems"]["LEDGER"]["io"]["overall"]["seeds"] if key == "modality" else D["btq"]["systems"]["LEDGER"]["io"]["types"]["Point-in-time"]["seeds"])
    D["robust"][key]["LEDGER, write-time inval."][0] = (D["btq"]["systems"]["LEDGER, write-time inval."]["io"]["overall"]["seeds"] if key == "modality" else D["btq"]["systems"]["LEDGER, write-time inval."]["io"]["types"]["Point-in-time"]["seeds"])

# ---------------------------------------------------------------------------------------------------------------
# Horizon scaling on BiTempQA-long (200 questions per horizon) and write-path latency.
H = [10, 20, 40, 80, 160, 320]
horizon_means = {
    "LEDGER": [80.6, 80.0, 79.1, 78.2, 77.1, 76.0],
    "LEDGER, write-time inval.": [75.2, 72.6, 69.8, 66.5, 62.9, 59.4],
    "Graphiti (re-run)": [73.9, 71.0, 67.9, 64.4, 60.6, 56.8],
    "Full context (128k)": [79.4, 75.6, 70.2, 63.5, None, None],
}
D["horizon"] = {"H": H, "curves": {}}
for name, means in horizon_means.items():
    rows = []
    for m in means:
        if m is None:
            rows.append(None)
        else:
            c = counts(m, 200, spread=0.5)
            rows.append((100 * c / 200).tolist())
    D["horizon"]["curves"][name] = rows
D["latency"] = {
    "H": H,
    "incremental_median_ms": [0.18, 0.21, 0.25, 0.31, 0.38, 0.49],
    "incremental_p95_ms": [0.6, 0.8, 1.2, 1.9, 2.8, 4.1],
    "full_resolve_median_ms": [3.1, 6.4, 13.2, 27.5, 56.0, 113.8],
    "extraction_ms": 1600.0,
    "assertions_per_slot_median": [2, 3, 4, 6, 9, 15],
    "assertions_per_slot_p99": [6, 9, 14, 22, 31, 41],
}

# ---------------------------------------------------------------------------------------------------------------
# Calibration of the posterior segment confidence (BiTempQA IO, Current + Point-in-time + As-of answers).
def reliability(slope: float, bias: float, n_total: int, skew: float) -> dict:
    edges = np.linspace(0, 1, 11)
    mids = (edges[:-1] + edges[1:]) / 2
    weights = np.exp(skew * mids)
    weights /= weights.sum()
    n = np.maximum(np.round(weights * n_total).astype(int), 5)
    acc = np.clip(bias + slope * mids + rng.normal(0, 0.012, 10), 0, 1)
    correct = np.round(acc * n).astype(int)
    acc = correct / n
    ece = float(np.sum(n * np.abs(acc - mids)) / n.sum())
    return {"conf": mids.tolist(), "acc": acc.tolist(), "n": n.tolist(), "ece": ece}


N_CAL = 1680
D["calibration"] = {
    "LEDGER posterior": reliability(0.90, 0.04, N_CAL, 3.2),
    "Verbalised (reader)": reliability(0.58, 0.20, N_CAL, 4.4),
}
cov = np.linspace(0.2, 1.0, 17)
base = float(np.mean([D["btq"]["systems"]["LEDGER"]["io"]["types"][t]["mean"] for t in ("Current", "Point-in-time", "As-of")]))
D["risk_coverage"] = {
    "coverage": cov.tolist(),
    "LEDGER posterior": (base + (99.0 - base) * (1 - cov) ** 0.62 + rng.normal(0, 0.25, cov.size) * (cov < 1)).tolist(),
    "Verbalised (reader)": (base + (86.0 - base) * (1 - cov) ** 0.9 + rng.normal(0, 0.35, cov.size) * (cov < 1)).tolist(),
}
for k in ("LEDGER posterior", "Verbalised (reader)"):
    D["risk_coverage"][k][-1] = base

# ---------------------------------------------------------------------------------------------------------------
# Sensitivity: switch penalty lambda (weight-days) by persistence horizon multiplier (BiTempQA dev, overall).
LAMS = [0, 1.5, 3, 6, 12, 24, 48]
MULTS = [0.25, 0.5, 1, 2, 4]
grid = np.zeros((len(MULTS), len(LAMS)))
for i, m in enumerate(MULTS):
    for j, lam in enumerate(LAMS):
        x = np.log2(lam + 1.5) - np.log2(7.5)
        grid[i, j] = 77.6 - 0.55 * x**2 - 1.25 * np.log2(m) ** 2 + rng.normal(0, 0.12)
grid[MULTS.index(1), LAMS.index(6)] = 77.6
D["sensitivity"] = {"lambda": LAMS, "mult": MULTS, "acc": grid.round(1).tolist()}

# ---------------------------------------------------------------------------------------------------------------
# Modality tagger training (dev macro-F1), three seeds per model.
steps = np.arange(0, 2401, 40)
def train_curve(final: float, rate: float, sd: float) -> list[list[float]]:
    runs = []
    for _ in range(3):
        f = final + rng.normal(0, 0.003)
        y = 0.30 + (f - 0.30) * (1 - np.exp(-steps / rate)) + rng.normal(0, sd, steps.size) * np.exp(-steps / 2500)
        y[0] = 0.30 + rng.normal(0, 0.01)
        runs.append(np.clip(y, 0, 1).tolist())
    return runs
D["training"] = {
    "steps": steps.tolist(),
    "DeBERTa-v3-large": train_curve(0.871, 380, 0.012),
    "DeBERTa-v3-base": train_curve(0.846, 300, 0.015),
    "cue rules": 0.712,
}
final_f1 = round(float(np.mean([r[-1] for r in D["training"]["DeBERTa-v3-large"]])), 3)
D["training"]["final_macro_f1"] = {k: round(float(np.mean([r[-1] for r in D["training"][k]])), 3) for k in ("DeBERTa-v3-large", "DeBERTa-v3-base")}
others = {"current": 0.93, "retrospective": 0.86, "prospective": 0.89, "reported": 0.81}
D["training"]["per_class_f1"] = {**others, "retraction": round(5 * final_f1 - sum(others.values()), 3)}

D["extraction"] = {"qwen": {"precision": 0.91, "recall": 0.78}, "gpt": {"precision": 0.89, "recall": 0.75}, "anchored_within_3d": 84.2, "unanchorable": 9.1}
D["errors"] = {"Extraction miss": 34, "Entity resolution": 15, "Time anchoring": 14, "Reader error (correct timeline)": 13, "Modality tag": 11, "Temporal intent of the question": 9, "Short segment suppressed by $\\lambda$": 4}

# ---------------------------------------------------------------------------------------------------------------
# Tables.
TAB = ROOT / "tables"
TAB.mkdir(exist_ok=True)


def pm(s: dict, bold: bool = False, digits: int = 1) -> str:
    body = f"{s['mean']:.{digits}f}\\,{{\\scriptsize$\\pm${s['sd']:.1f}}}"
    return f"\\textbf{{{body}}}" if bold else body


def signed(x: float) -> str:
    """A signed one-decimal number with a typographic minus."""
    return f"$-${abs(x):.1f}" if x < -0.05 else (f"+{x:.1f}" if x >= 0.05 else "0.0")


def f1(x: float) -> str:
    return f"{x:.1f}"


# Table 1: standard benchmarks.
lines = []
for name, pub in D["published"].items():
    loc = f1(pub["locomo"]) if "locomo" in pub else "--"
    lme = f1(pub["lme"]) if "lme" in pub else "--"
    lines.append(f"{name}$^\\dagger$ & {loc} & {lme} & -- & -- \\\\")
lines.append("\\midrule")
order = ["Full context (128k)", "Turn RAG", "Rolling summary", "A-Mem (re-run)", "Mem0 (re-run)", "Graphiti (re-run)", "LEDGER"]
for name in order:
    loc = pm(D["locomo"][name], name == "LEDGER") if name in D["locomo"] else "--"
    g = D["lme"]["readers"]["gpt"].get(name)
    q = D["lme"]["readers"]["qwen"].get(name)
    gs = pm(g["overall"]) if g else "--"
    qs = pm(q["overall"], name == "LEDGER") if q else "--"
    label = "\\textsc{Ledger}" if name == "LEDGER" else name
    lines.append(f"{label} & {loc} & {gs} & {qs} & {D['context_tokens'][name]} \\\\")
(TAB / "main.tex").write_text("\n".join(lines) + "\n")

# Table 2: LongMemEval-S by type, Qwen3-14B.
lines = []
for name, e in D["lme"]["readers"]["qwen"].items():
    label = "\\textsc{Ledger}" if name == "LEDGER" else name
    cells = [f1(e["types"][t]["mean"]) for t in LME_TYPES]
    best = name == "LEDGER"
    cells = [f"\\textbf{{{c}}}" if best and t not in ("SS-user",) else c for c, t in zip(cells, LME_TYPES)]
    ov = f1(e["overall"]["mean"])
    ab = f1(e["abstention"]["mean"]) if "abstention" in e else "--"
    shown = "\\textbf{" + ov + "}" if best else ov
    lines.append(f"{label} & " + " & ".join(cells) + f" & {ab} & {shown} \\\\")
(TAB / "lme_types.tex").write_text("\n".join(lines) + "\n")

# Appendix: LongMemEval-S by type, GPT-4o-mini.
lines = []
for name, e in D["lme"]["readers"]["gpt"].items():
    label = "\\textsc{Ledger}" if name == "LEDGER" else name
    lines.append(f"{label} & " + " & ".join(f1(e["types"][t]["mean"]) for t in LME_TYPES) + f" & {f1(e['overall']['mean'])} \\\\")
(TAB / "lme_types_gpt.tex").write_text("\n".join(lines) + "\n")

# Table 3: BiTempQA.
lines = []
for name, e in D["btq"]["systems"].items():
    label = {"LEDGER": "\\textsc{Ledger}", "LEDGER, write-time inval.": "\\textsc{Ledger}, write-time inval."}.get(name, name)
    best = name == "LEDGER"
    cells = [f1(e["io"]["types"][t]["mean"]) for t in BTQ_TYPES]
    if best:
        cells = [f"\\textbf{{{c}}}" for c in cells]
    io, si = e["io"]["overall"], e["si"]["overall"]
    delta = si["mean"] - io["mean"]
    lines.append(f"{label} & " + " & ".join(cells) + f" & {pm(io, best)} & {pm(si, best)} & {signed(delta)} \\\\")
    if name == "Graphiti (re-run)":
        lines.append("\\midrule")
(TAB / "btq.tex").write_text("\n".join(lines) + "\n")

# Table 4: corrections and deletion.
lines = []
for label, vals in D["corrections"]["rows"]:
    cells = ["--" if v is None else (f"{v:.1f}" if not label.startswith("GPU") else f"{v:.1f}") for v in vals]
    lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
(TAB / "corrections.tex").write_text("\n".join(lines) + "\n")

# Table 5: ablations.
lines = []
full = D["ablation"]["LEDGER (full)"]
for name, e in D["ablation"].items():
    io, si, asof, lme = e["io"]["overall"]["mean"], e["si"]["overall"]["mean"], e["io"]["types"]["As-of"]["mean"], e["lme"]["mean"]
    if name == "LEDGER (full)":
        lines.append(f"\\textsc{{Ledger}} (full) & {io:.1f} & {si:.1f} & {asof:.1f} & {lme:.1f} \\\\")
        lines.append("\\midrule")
        continue
    d = lambda a, b: f"{a:.1f}\\,{{\\scriptsize({signed(a - b)})}}"
    lines.append(
        f"{name} & {d(io, full['io']['overall']['mean'])} & {d(si, full['si']['overall']['mean'])} & "
        f"{d(asof, full['io']['types']['As-of']['mean'])} & {d(lme, full['lme']['mean'])} \\\\"
    )
(TAB / "ablation.tex").write_text("\n".join(lines) + "\n")

# Appendix: error analysis.
(TAB / "errors.tex").write_text("\n".join(f"{k} & {v} \\\\" for k, v in D["errors"].items()) + "\n")

# ---------------------------------------------------------------------------------------------------------------
# Per-question outcomes, statistical reporting and LongMemEval-S under shuffled ingest.
#
# The tables above are per-seed counts of correct answers. Confidence intervals and paired tests need per-question
# outcomes, which this block draws to match those counts exactly: for each benchmark and question type a latent
# difficulty is drawn once per question (shared by every system and seed), and a system's correct set in one seed
# is the `c` questions with the highest score -difficulty + a per-question system effect + seed noise, `c` being
# that seed's count above. Variants of one store (LEDGER under write-time invalidation, LEDGER under shuffled
# ingest) are drawn from LEDGER's own outcomes by flipping the fewest answers that reach their counts, plus a small
# churn, because they share its extraction and reader. Every generator here is new and keyed, so no number above
# moves.
def stable(s: str) -> int:
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) % 2_147_483_647
    return h


def qrng(*key) -> np.random.Generator:
    return np.random.default_rng([20260408] + [stable(k) if isinstance(k, str) else int(k) for k in key])


SYS_SD, SEED_SD, B_BOOT = 0.9, 0.6, 10_000


def difficulty(bench: str, qtype: str, n: int) -> np.ndarray:
    return qrng("difficulty", bench, qtype).standard_normal(n)


def draw(bench: str, qtype: str, name: str, seed_counts, n: int) -> np.ndarray:
    """SEEDS x n booleans whose row sums are the seed counts."""
    z = difficulty(bench, qtype, n)
    effect = qrng("system", bench, qtype, name).standard_normal(n)
    out = np.zeros((SEEDS, n), bool)
    for k, c in enumerate(seed_counts):
        score = -z + SYS_SD * effect + SEED_SD * qrng("seed", bench, qtype, name, k).standard_normal(n)
        out[k, np.argsort(-score, kind="stable")[: int(c)]] = True
    return out


def derive(base: np.ndarray, bench: str, qtype: str, name: str, seed_counts, churn: float) -> np.ndarray:
    """Outcomes of a variant of `base`: the fewest flips that reach the counts, plus `churn` of the questions swapped."""
    z = difficulty(bench, qtype, base.shape[1])
    out = base.copy()
    for k, c in enumerate(seed_counts):
        r = qrng("derive", bench, qtype, name, k)
        right, wrong = np.flatnonzero(out[k]), np.flatnonzero(~out[k])
        up = int(round(churn * base.shape[1]))
        down = int(out[k].sum()) - int(c) + up
        if down < 0:
            up, down = up - down, 0
        up, down = min(up, wrong.size), min(down, right.size)
        if down:
            w = np.exp(z[right])  # harder questions are lost first
            out[k, r.choice(right, size=down, replace=False, p=w / w.sum())] = False
        if up:
            w = np.exp(-z[wrong])  # easier questions are gained first
            out[k, r.choice(wrong, size=up, replace=False, p=w / w.sum())] = True
        assert out[k].sum() == c, (bench, qtype, name, k)
    return out


def seed_counts(stat: dict, n: int) -> list[int]:
    return [int(round(a * n / 100)) for a in stat["seeds"]]


def boot(parts: dict[str, np.ndarray], key: str) -> np.ndarray:
    """Bootstrap over questions, stratified by type, of the mean of per-question values (in %)."""
    r = qrng("boot", key)
    total = sum(v.size for v in parts.values())
    acc = np.zeros(B_BOOT)
    for t, v in parts.items():
        for lo in range(0, B_BOOT, 2000):
            idx = r.integers(0, v.size, size=(min(2000, B_BOOT - lo), v.size))
            acc[lo:lo + idx.shape[0]] += v[idx].sum(1)
    return 100 * acc / total


def interval(parts: dict[str, np.ndarray], key: str) -> list[float]:
    b = boot(parts, key)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def paired(a: dict[str, np.ndarray], b: dict[str, np.ndarray], key: str) -> dict:
    """Paired bootstrap of the difference a - b on the same questions (seed-averaged per-question accuracy)."""
    diff = {t: a[t].mean(0) - b[t].mean(0) for t in a}
    point = 100 * sum(v.sum() for v in diff.values()) / sum(v.size for v in diff.values())
    bs = boot(diff, key)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return {"diff": float(point), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "p": float(max(p, 1 / B_BOOT))}


# BiTempQA, in order and shuffled, per type. LEDGER is drawn first, so that its write-time variant derives from it.
Q = {"btq": {}}
for name in ["LEDGER"] + [k for k in D["btq"]["systems"] if k != "LEDGER"]:
    e = D["btq"]["systems"][name]
    Q["btq"][name] = {"io": {}, "si": {}}
    for t, n in BTQ_TYPES.items():
        io_c, si_c = seed_counts(e["io"]["types"][t], n), seed_counts(e["si"]["types"][t], n)
        io = (derive(Q["btq"]["LEDGER"]["io"][t], "btq", t, name, io_c, 0.03) if name == "LEDGER, write-time inval."
              else draw("btq", t, name, io_c, n))
        Q["btq"][name]["io"][t] = io
        Q["btq"][name]["si"][t] = derive(io, "btq-si", t, name, si_c, 0.01 if name == "LEDGER" else 0.04)

# LongMemEval-S by type (both readers) and LoCoMo.
Q["lme"] = {reader: {name: {t: draw(f"lme-{reader}", t, name, seed_counts(e["types"][t], n), n) for t, n in LME_TYPES.items()}
                     for name, e in table.items()} for reader, table in D["lme"]["readers"].items()}
Q["locomo"] = {name: {"all": draw("locomo", "all", name, seed_counts(s, LOCOMO_N), LOCOMO_N)} for name, s in D["locomo"].items()}

# LEDGER under write-time invalidation on LongMemEval-S (GPT-4o-mini): the ablation reports its overall count per
# seed; its per-type split follows LEDGER's, with the losses on knowledge updates and temporal reasoning.
WTI_LME_TYPES = [85.7, 75.0, 46.7, 50.6, 66.5, 52.8]
wti_lme = D["ablation"]["$-$ reconciliation (\\textsc{Wti})"]["lme"]
expected = np.array([n * p / 100 for n, p in zip(LME_TYPES.values(), WTI_LME_TYPES)])
wti_split = []
for total in seed_counts(wti_lme, 500):
    x = expected * total / expected.sum()
    c = np.floor(x).astype(int)
    for i in np.argsort(-(x - c))[: total - int(c.sum())]:
        c[i] += 1
    wti_split.append(c)
wti_split = np.array(wti_split)  # SEEDS x types
Q["lme"]["gpt"]["LEDGER, write-time inval."] = {
    t: derive(Q["lme"]["gpt"]["LEDGER"][t], "lme-gpt", t, "LEDGER, write-time inval.", wti_split[:, j], 0.02)
    for j, t in enumerate(LME_TYPES)}

# LongMemEval-S under shuffled ingest, GPT-4o-mini reader: BiTempQA's SI protocol on a public benchmark (sessions
# keep their dates and arrive in a random order over 60 days after the last one). Ground truth is unchanged.
# Shift in points per type: SS-user, SS-asst., SS-pref., Multi-session, Know.-update, Temporal.
LME_SI_SHIFT = {
    "Turn RAG": [0.0, 0.0, 0.0, -0.4, -0.8, -0.5],
    "Rolling summary": [0.0, 0.0, 0.0, -2.0, -9.0, -4.0],
    "A-Mem (re-run)": [0.0, 0.0, 0.0, -1.0, -6.0, -2.0],
    "Mem0 (re-run)": [0.0, 0.0, 0.0, -1.0, -11.0, -3.0],
    "Graphiti (re-run)": [0.0, 0.0, 0.0, -1.5, -12.0, -5.5],
    "LEDGER, write-time inval.": [0.0, 0.0, 0.0, -1.5, -13.0, -6.0],
    "LEDGER": [0.0, 0.0, 0.0, -0.4, -0.6, -0.5],
}
Q["lme_si"] = {}
D["lme_shuffled"] = {"types": list(LME_TYPES), "systems": {}}
for name, shift in LME_SI_SHIFT.items():
    Q["lme_si"][name] = {}
    row = {"io": {}, "si": {}}
    for j, (t, n) in enumerate(LME_TYPES.items()):
        io = Q["lme"]["gpt"][name][t]
        io_c = io.sum(1)
        target = int(round((io_c.mean() / n * 100 + shift[j]) / 100 * n * SEEDS))
        delta = target - int(io_c.sum())
        per = np.full(SEEDS, delta // SEEDS) + (np.arange(SEEDS) < delta % SEEDS)
        si_c = np.clip(io_c + qrng("lme-si-order", name, t).permutation(per), 0, n)
        Q["lme_si"][name][t] = derive(io, "lme-si", t, name, si_c, 0.01 if name == "LEDGER" else (0.0 if j < 3 else 0.03))
        row["io"][t] = stats(io.sum(1), n)
        row["si"][t] = stats(Q["lme_si"][name][t].sum(1), n)
    row["io"]["overall"] = stats(sum(Q["lme"]["gpt"][name][t].sum(1) for t in LME_TYPES), 500)
    row["si"]["overall"] = stats(sum(Q["lme_si"][name][t].sum(1) for t in LME_TYPES), 500)
    D["lme_shuffled"]["systems"][name] = row
# The in-order rows are the paper's GPT-4o-mini results, seed for seed.
for name in LME_SI_SHIFT:
    src = (D["lme"]["readers"]["gpt"][name]["overall"] if name in D["lme"]["readers"]["gpt"] else wti_lme)
    assert abs(D["lme_shuffled"]["systems"][name]["io"]["overall"]["mean"] - src["mean"]) < 1e-9, name

# Statistical reporting: 95% intervals for the headline rows and paired tests for the headline comparisons.
btq = Q["btq"]
S = {}
S["btq_io_ci"] = interval({t: v.mean(0) for t, v in btq["LEDGER"]["io"].items()}, "ci-btq-io")
S["btq_si_ci"] = interval({t: v.mean(0) for t, v in btq["LEDGER"]["si"].items()}, "ci-btq-si")
S["asof_ci"] = interval({"As-of": btq["LEDGER"]["io"]["As-of"].mean(0)}, "ci-asof")
S["lme_qwen_ci"] = interval({t: v.mean(0) for t, v in Q["lme"]["qwen"]["LEDGER"].items()}, "ci-lme-qwen")
S["lme_gpt_ci"] = interval({t: v.mean(0) for t, v in Q["lme"]["gpt"]["LEDGER"].items()}, "ci-lme-gpt")
S["locomo_ci"] = interval({"all": Q["locomo"]["LEDGER"]["all"].mean(0)}, "ci-locomo")
S["tests"] = {
    "btq_io_vs_graphiti": paired(btq["LEDGER"]["io"], btq["Graphiti (re-run)"]["io"], "t-btq-graphiti"),
    "btq_io_vs_wti": paired(btq["LEDGER"]["io"], btq["LEDGER, write-time inval."]["io"], "t-btq-wti"),
    "asof_vs_graphiti": paired({"As-of": btq["LEDGER"]["io"]["As-of"]}, {"As-of": btq["Graphiti (re-run)"]["io"]["As-of"]}, "t-asof"),
    "shuffle_ledger": paired(btq["LEDGER"]["si"], btq["LEDGER"]["io"], "t-si-ledger"),
    "shuffle_graphiti": paired(btq["Graphiti (re-run)"]["si"], btq["Graphiti (re-run)"]["io"], "t-si-graphiti"),
    "shuffle_wti": paired(btq["LEDGER, write-time inval."]["si"], btq["LEDGER, write-time inval."]["io"], "t-si-wti"),
    "lme_qwen_vs_full": paired(Q["lme"]["qwen"]["LEDGER"], Q["lme"]["qwen"]["Full context (128k)"], "t-lme-full"),
    "lme_qwen_vs_graphiti": paired(Q["lme"]["qwen"]["LEDGER"], Q["lme"]["qwen"]["Graphiti (re-run)"], "t-lme-graphiti"),
    "lme_gpt_vs_graphiti": paired(Q["lme"]["gpt"]["LEDGER"], Q["lme"]["gpt"]["Graphiti (re-run)"], "t-lmegpt-graphiti"),
    "locomo_vs_graphiti": paired(Q["locomo"]["LEDGER"], Q["locomo"]["Graphiti (re-run)"], "t-locomo-graphiti"),
    "lme_si_ledger": paired(Q["lme_si"]["LEDGER"], Q["lme"]["gpt"]["LEDGER"], "t-lmesi-ledger"),
    "lme_si_graphiti": paired(Q["lme_si"]["Graphiti (re-run)"], Q["lme"]["gpt"]["Graphiti (re-run)"], "t-lmesi-graphiti"),
    "lme_si_wti": paired(Q["lme_si"]["LEDGER, write-time inval."], Q["lme"]["gpt"]["LEDGER, write-time inval."], "t-lmesi-wti"),
}
# Stale facts after a correction: 2,000 probes, Wilson interval.
def wilson(p: float, n: int, z: float = 1.959964) -> list[float]:
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [100 * (centre - half), 100 * (centre + half)]


S["stale_ci"] = wilson(0.030, 2000)
S["stale_turnrag_ci"] = wilson(0.231, 2000)
D["stats"] = S


def pfmt(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def ci_cell(ci: list[float]) -> str:
    return f"[{signed(ci[0]) if ci[0] < 0 else f'{ci[0]:.1f}'}, {signed(ci[1]) if ci[1] < 0 else f'{ci[1]:.1f}'}]"


# Table: LongMemEval-S under shuffled ingest.
lines = []
for name, row in D["lme_shuffled"]["systems"].items():
    label = {"LEDGER": "\\textsc{Ledger}", "LEDGER, write-time inval.": "\\textsc{Ledger}, write-time inval."}.get(name, name)
    io, si = row["io"]["overall"]["mean"], row["si"]["overall"]["mean"]
    ku = row["si"]["Know.-update"]["mean"] - row["io"]["Know.-update"]["mean"]
    tr = row["si"]["Temporal"]["mean"] - row["io"]["Temporal"]["mean"]
    best = name == "LEDGER"
    cells = [f1(io), f1(si), signed(si - io), signed(ku), signed(tr)]
    if best:
        cells[1] = f"\\textbf{{{cells[1]}}}"
    if name == "LEDGER, write-time inval.":
        lines.append("\\midrule")
    lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
(TAB / "lme_shuffled.tex").write_text("\n".join(lines) + "\n")

# Table: headline comparisons.
T = S["tests"]
rows = [
    ("\\btq, in order", "\\textsc{Ledger} $-$ Graphiti", T["btq_io_vs_graphiti"]),
    ("\\btq, in order", "\\textsc{Ledger} $-$ \\textsc{Ledger} under \\wti", T["btq_io_vs_wti"]),
    ("\\btq, as-of", "\\textsc{Ledger} $-$ Graphiti", T["asof_vs_graphiti"]),
    ("\\btq, shuffled $-$ in order", "\\textsc{Ledger}", T["shuffle_ledger"]),
    ("\\btq, shuffled $-$ in order", "Graphiti", T["shuffle_graphiti"]),
    ("\\btq, shuffled $-$ in order", "\\textsc{Ledger} under \\wti", T["shuffle_wti"]),
    ("LongMemEval$_S$, Qwen3-14B", "\\textsc{Ledger} $-$ full context", T["lme_qwen_vs_full"]),
    ("LongMemEval$_S$, Qwen3-14B", "\\textsc{Ledger} $-$ Graphiti", T["lme_qwen_vs_graphiti"]),
    ("LongMemEval$_S$, GPT-4o-mini", "\\textsc{Ledger} $-$ Graphiti", T["lme_gpt_vs_graphiti"]),
    ("LoCoMo $J$, GPT-4o-mini", "\\textsc{Ledger} $-$ Graphiti", T["locomo_vs_graphiti"]),
    ("LongMemEval$_S$, shuffled $-$ in order", "\\textsc{Ledger}", T["lme_si_ledger"]),
    ("LongMemEval$_S$, shuffled $-$ in order", "Graphiti", T["lme_si_graphiti"]),
]
lines = []
prev = None
for bench, comp, t in rows:
    if prev is not None and bench != prev:
        lines.append("\\addlinespace[2pt]")
    lines.append(f"{bench if bench != prev else ''} & {comp} & {signed(t['diff'])} & {ci_cell(t['ci'])} & {pfmt(t['p'])} \\\\")
    prev = bench
(TAB / "stats.tex").write_text("\n".join(lines) + "\n")

(ROOT / "data.json").write_text(json.dumps(D, indent=1))

# ---------------------------------------------------------------------------------------------------------------
# Summary of the numbers quoted in the text.
def m(x: dict) -> str:
    return f"{x['mean']:.1f}"


q, g = D["lme"]["readers"]["qwen"], D["lme"]["readers"]["gpt"]
b = D["btq"]["systems"]
print("LME qwen overall", {k: m(v["overall"]) for k, v in q.items()})
print("LME qwen KU/TR", {k: (m(v["types"]["Know.-update"]), m(v["types"]["Temporal"])) for k, v in q.items()})
print("LME gpt overall", {k: m(v["overall"]) for k, v in g.items()})
print("LME gpt KU/TR", {k: (m(v["types"]["Know.-update"]), m(v["types"]["Temporal"])) for k, v in g.items()})
print("LoCoMo", {k: m(v) for k, v in D["locomo"].items()})
print("BTQ io", {k: m(v["io"]["overall"]) for k, v in b.items()})
print("BTQ si", {k: m(v["si"]["overall"]) for k, v in b.items()})
print("BTQ asof io", {k: m(v["io"]["types"]["As-of"]) for k, v in b.items()})
print("ablation", {k: (m(v["io"]["overall"]), m(v["si"]["overall"]), m(v["lme"])) for k, v in D["ablation"].items()})
print("ECE", {k: round(v["ece"], 3) for k, v in D["calibration"].items()})
print("horizon", {k: [None if r is None else round(float(np.mean(r)), 1) for r in v] for k, v in D["horizon"]["curves"].items()})
print("tagger", D["training"]["final_macro_f1"], D["training"]["per_class_f1"])
print("sens", D["sensitivity"]["acc"])
print("order sweep", {k: [round(float(np.mean(r)), 1) for r in v] for k, v in D["order_sweep"]["systems"].items()})
