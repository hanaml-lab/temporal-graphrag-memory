# LEDGER: order-invariant bitemporal reconciliation for knowledge-graph memory

[![ci](https://github.com/hanaml-lab/temporal-graphrag-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/hanaml-lab/temporal-graphrag-memory/actions/workflows/ci.yml)

Reference implementation of *LEDGER: Order-Invariant Bitemporal Reconciliation for Knowledge-Graph Memory in Long-Horizon LLM Agents* (Agarwal, hanaML technical report, 2026).

Temporal knowledge-graph memories record when a fact held (valid time) and when the agent learned it (transaction time). They maintain the valid-time axis by **write-time invalidation**: each incoming fact is compared once with the edges still open, and a contradicted edge is closed for good. The result depends on the order in which evidence arrives, and a misread assertion (a reminiscence, a plan, someone else's claim) closes a true edge permanently.

LEDGER keeps an append-only ledger of modality-tagged assertions and recomputes each slot's timeline from the whole evidence set:

- **Evidence model** (`ledger.reconcile`): explicit evidence from stated intervals and statements is summed; default persistence after a current statement is combined by a maximum; plans carry no valid-time evidence; undated reminiscences are provenance only; retractions are negative evidence.
- **Exact reconciliation** (`ledger.reconcile`): the timeline maximising evidence minus a penalty per switch, solved by Viterbi over the evidence breakpoints in O(nK), with forward-backward segment confidences.
- **Versioned store** (`ledger.store`): one version per transaction time, so `as_of(slot, t)` returns exactly the belief held at `t`; `redact(source)` removes an episode from every version and advances the slot's epoch.
- **Baseline** (`ledger.baseline`): write-time invalidation following Zep/Graphiti's edge-resolution rules, including the order-dependence counterexample from the report.
- **Time and modality** (`ledger.temporal`, `ledger.tagging`): statement-anchored resolution of time expressions into fuzzy intervals and a cue-based modality tagger (the report uses a fine-tuned DeBERTa-v3-large).
- **Toy setting** (`ledger.toy`): synthetic two-year life scripts, three ingest regimes and the five BiTempQA question types, end to end on CPU.

This release does not include the LLM harness (extraction prompts, hosted readers and judges) used for LongMemEval, LoCoMo and BiTempQA; `configs/btq_gpt4omini.yaml` records its settings.

## Install

```bash
pip install -e ".[dev]"           # numpy, pyyaml; pytest and ruff
pip install -e ".[paper]"         # + matplotlib, to regenerate the report's figures
```

## Quickstart (CPU, under a minute)

```bash
python -m ledger.demo             # or ledger-demo; settings in configs/toy.yaml
pytest -q
```

## Toy run (real output)

The demo has two parts. First, one history of 20 assertions about where a user lived and worked (with plans, an undated reminiscence, other people's claims, late disclosures and a retraction) is ingested after the fact in 100 arrival orders, the first chronological, and each store's final timeline is scored on probe days away from any change. Second, the BiTempQA-style toy scores 40 synthetic users on the five question types, with evidence ingested in order and shuffled. Output of `python -m ledger.demo` on a CPU (Python 3.12):

```
one history of 20 assertions about 2 slots, ingested in 100 random orders
store           distinct final timelines  correct, in order   mean  worst
ledger                                 1              1.000  1.000  1.000
wti                                   95              0.847  0.615  0.245
wti+modality                          89              0.969  0.638  0.235

BiTempQA-style toy, in-order ingest (40 users, seeds [0], 2422 questions), accuracy %
store           current      pit     asof   change  learned  overall
ledger             98.3     94.4     96.0     73.6     98.8     91.4
wti                68.3     69.2     73.5     63.0     82.7     72.1
wti+modality       90.8     82.5     92.5     74.7     98.8     87.8

BiTempQA-style toy, shuffled ingest (40 users, seeds [0], 2422 questions), accuracy %
store           current      pit     asof   change  learned  overall
ledger             98.3     94.4     96.5     73.6     98.3     91.3
wti                60.4     47.1     67.5     27.9     90.8     59.6
wti+modality       87.1     59.0     89.8     36.6     92.6     71.6

done in 40.1s
```

LEDGER ends with the same timeline in every arrival order, and its answers about the final state are identical in order and shuffled, as the report's order-invariance result requires. Write-time invalidation ends with a different timeline for most orders and loses 12.5 points overall when ingest is shuffled. On change questions in order, the switch penalty costs LEDGER some short segments. `paper/toy_run.md` records the environment and what each line shows.

## Reconciling your own assertions

`examples/reconcile_assertions.py` reads assertions in the order the store received them (slot, value, modality, the day it was said and the day it was learned, and an optional stated interval) and prints LEDGER's timeline for each slot, the timeline write-time invalidation builds from the same arrivals, and what LEDGER believed at chosen days.

```bash
python examples/reconcile_assertions.py examples/data/account_assertions.jsonl --as-of 100 170
```

```
acme.plan
  LEDGER                   starter [-30, 58) -> pro [58, 236) -> enterprise [236, 625)
  write-time invalidation  starter [0, 62) -> pro [62, 240) -> enterprise [240, ...)
  as of day 100, LEDGER held starter [-30, 385)
  as of day 170, LEDGER held starter [-30, 58) -> pro [58, 515)
acme.billing_city
  LEDGER                   Austin [-25, 100) -> Denver [100, 575)
  write-time invalidation  Austin [5, 130) -> Denver [130, ...)
  as of day 100, LEDGER held Austin [-25, 370)
  as of day 170, LEDGER held Austin [-25, 100) -> Denver [100, 495)
```

The store from Python:

```python
from ledger import Assertion, FuzzyInterval, LedgerStore, Modality

slot = ("user", "lives_in")
store = LedgerStore()
store.append(Assertion(slot, "Berlin", Modality.CURRENT, stmt_time=30, tx_time=30))
store.append(Assertion(slot, "Lisbon", Modality.CURRENT, stmt_time=150, tx_time=150, valid=FuzzyInterval.exact(60, 150)))
store.append(Assertion(slot, "Berlin", Modality.RETROSPECTIVE, stmt_time=210, tx_time=210))  # "back when I lived in Berlin"

store.current(slot).value_at(220)    # 'Lisbon', whatever the arrival order
store.as_of(slot, 100).value_at(90)  # 'Berlin': what the store believed on day 100
store.changes(slot, 0, 300)          # [(60, 'Berlin', 'Lisbon')]
```

## Results (from the report)

These numbers come from the report's seeded simulation and are illustrative. GPT-4o-mini reader unless noted; mean over 5 seeds:

| Method | BiTempQA, in order | BiTempQA, shuffled ingest | As-of questions | LongMemEval-S | LongMemEval-S (Qwen3-14B) | LoCoMo J |
|---|---|---|---|---|---|---|
| Full context (128k) | 51.1 | 46.3 | 30.2 | 55.4 (published) | 58.2 | 72.9 (published) |
| Mem0 (re-run) | 35.8 | 29.6 | 16.9 | 59.1 | 58.6 | 66.3 |
| Graphiti (re-run) | 60.2 | 49.0 | 46.5 | 60.3 | 65.0 | 67.4 |
| LEDGER store, write-time invalidation | 64.9 | 52.6 | 53.0 | 61.1 | | |
| **LEDGER** | **77.2** | **76.9** | **76.9** | **62.4** | **69.4** | **68.6** |

`bash scripts/reproduce_paper.sh` regenerates the tables and figures that the report's seeded simulation produces. On every push, CI reruns `paper/make_data.py` and checks the tables and results it writes against `paper/results.sha256` (`python paper/check_results.py`).

## Layout

```
src/ledger/       types.py  reconcile.py  store.py  baseline.py  temporal.py  tagging.py  toy.py  demo.py
configs/          toy.yaml  btq_gpt4omini.yaml
examples/         reconcile_assertions.py, data/account_assertions.jsonl
scripts/          run_toy.py  reproduce_paper.sh
paper/            make_data.py  figures.py  check_results.py  results.sha256  toy_run.md
tests/            pytest suite
```

## Citation

```bibtex
@techreport{agarwal2026ledger,
  title       = {{LEDGER}: Order-Invariant Bitemporal Reconciliation for Knowledge-Graph Memory in Long-Horizon {LLM} Agents},
  author      = {Agarwal, Manav},
  institution = {hanaML},
  year        = {2026},
  url         = {https://hanaml.com/research/papers/temporal-graphrag-memory}
}
```

## License

Apache-2.0. See `LICENSE`.
