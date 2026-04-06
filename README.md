# LEDGER: Order-Invariant Bitemporal Reconciliation for Knowledge-Graph Memory

Reference implementation for the technical report *LEDGER: Order-Invariant Bitemporal Reconciliation for Knowledge-Graph Memory in Long-Horizon LLM Agents* (Manav Agarwal, hanaML, 2026).

Temporal knowledge-graph memories record when a fact held (valid time) and when the agent learned it (transaction time). They maintain the valid-time axis by **write-time invalidation**: each incoming fact is compared once with the edges still open, and a contradicted edge is closed for good. The result depends on the order in which evidence arrives, and a misread assertion (a reminiscence, a plan, someone else's claim) closes a true edge permanently.

LEDGER keeps an append-only ledger of modality-tagged assertions and recomputes each slot's timeline from the whole evidence set:

- **Evidence model** (`ledger.reconcile`): explicit evidence from stated intervals and statements is summed; default persistence after a current statement is combined by a maximum; plans carry no valid-time evidence; undated reminiscences are provenance only; retractions are negative evidence.
- **Exact reconciliation** (`ledger.reconcile`): the timeline maximising evidence minus a penalty per switch, solved by Viterbi over the evidence breakpoints in O(nK), with forward-backward segment confidences.
- **Versioned store** (`ledger.store`): one version per transaction time, so `as_of(slot, t)` returns exactly the belief held at `t`; `redact(source)` removes an episode from every version and advances the slot's epoch.
- **Baseline** (`ledger.baseline`): write-time invalidation following Zep/Graphiti's edge-resolution rules, including the order-dependence counterexample from the report.
- **Time and modality** (`ledger.temporal`, `ledger.tagging`): statement-anchored resolution of time expressions into fuzzy intervals and a cue-based modality tagger (the report uses a fine-tuned DeBERTa-v3-large).
- **Toy setting** (`ledger.toy`): synthetic two-year life scripts, three ingest regimes and the five BiTempQA question types, end to end on CPU.

This release does not include the LLM harness (extraction prompts, hosted readers and judges) used for LongMemEval, LoCoMo and BiTempQA; `configs/btq_gpt4omini.toml` records its settings.

## Install

```bash
pip install -e ".[dev,paper]"     # Python 3.11+, numpy; matplotlib for the paper figures
```

## Quickstart

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

```bash
python scripts/run_toy.py --config configs/toy.toml   # toy sweep: 40 users x 3 seeds x 2 regimes (about 90 s)
pytest -q                                              # 35 tests
bash scripts/reproduce_paper.sh                        # regenerate every table and figure of the report
```

## Results

From the report (GPT-4o-mini reader unless noted; mean over 5 seeds):

| Method | BiTempQA, in order | BiTempQA, shuffled ingest | As-of questions | LongMemEval-S | LongMemEval-S (Qwen3-14B) | LoCoMo J |
|---|---|---|---|---|---|---|
| Full context (128k) | 51.1 | 46.3 | 30.2 | 55.4 (published) | 58.2 | 72.9 (published) |
| Mem0 (re-run) | 35.8 | 29.6 | 16.9 | 59.1 | 58.6 | 66.3 |
| Graphiti (re-run) | 60.2 | 49.0 | 46.5 | 60.3 | 65.0 | 67.4 |
| LEDGER store, write-time invalidation | 64.9 | 52.6 | 53.0 | 61.1 | | |
| **LEDGER** | **77.2** | **76.9** | **76.9** | **62.4** | **69.4** | **68.6** |

The report's numbers come from a seeded simulation study calibrated to the published baseline figures; `scripts/paper/make_data.py` generates all of them.

On the CPU toy setting (`scripts/run_toy.py`, 40 users, seeds 0-2), accuracy in %:

| System | Current | Point-in-time | As-of | Change | Learned-at | Overall |
|---|---|---|---|---|---|---|
| LEDGER, in order | 98.5 | 94.2 | 96.7 | 72.8 | 98.9 | 91.4 |
| LEDGER, shuffled | 98.5 | 94.2 | 97.3 | 72.8 | 98.9 | 91.5 |
| Write-time invalidation + modality, in order | 89.4 | 86.4 | 93.1 | 77.3 | 98.8 | 89.2 |
| Write-time invalidation + modality, shuffled | 88.3 | 59.9 | 92.6 | 37.3 | 92.8 | 72.8 |
| Write-time invalidation, in order | 69.2 | 71.7 | 75.6 | 66.8 | 81.6 | 73.7 |
| Write-time invalidation, shuffled | 62.1 | 47.0 | 71.7 | 27.1 | 91.4 | 60.7 |

LEDGER's answers about the final state (current, point-in-time, change) are identical under both regimes, as Proposition 2 of the report requires. On change questions in order, the switch penalty costs LEDGER some short segments that write-time invalidation keeps.

## Layout

```
src/ledger/       types.py  reconcile.py  store.py  baseline.py  temporal.py  tagging.py  toy.py
configs/          toy.toml  btq_gpt4omini.toml
scripts/          run_toy.py  reproduce_paper.sh  paper/{make_data.py, figures.py}
tests/            test_reconcile.py  test_modality.py  test_baseline.py  test_store.py  test_temporal.py  test_toy.py  test_scripts.py
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

Apache-2.0.
