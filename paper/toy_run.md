# Toy run (real output)

A real run of the reference implementation on its CPU toy setting. Unlike the report's tables, which come from a
seeded simulation and are illustrative, every number below was printed by the code. The toy is small by design: it
checks that the method works end to end on a problem with a known answer; it does not reproduce the report's
experiments.

- Command: `python -m ledger.demo` (settings in `configs/toy.yaml`)
- Code: commit `ac68144`
- Environment: Python 3.12.13, numpy 2.5.4, PyYAML 6.0.3; Linux x86_64, Intel(R) Core(TM) i9-14900KF
- Wall time: 41 s

## The toy setting

Part 1 is one history of 20 assertions about where a user lived (Berlin, then Lisbon from about day 124, then Porto
from about day 390) and worked (Quill, then Harbor from day 300), written in `ledger.demo.HISTORY`. It includes
plans, an undated reminiscence, other people's claims, two moves disclosed late with a stated date, and a
retraction. The history is ingested after the fact in 100 arrival orders, the first chronological and the rest
random, into LEDGER and into write-time invalidation with and without modality filtering. Each store's final
timeline is scored on 49 probe days per slot, at least 15 days from any change.

Part 2 is the BiTempQA-style toy (`ledger.toy`): 40 synthetic users with six functional slots over two years,
disclosed the way people do in conversation, and scored on the five question types (current state, value at a
point in time, belief as of a transaction time, when a change happened, when the store learned a value). Evidence
is ingested in order and shuffled.

## Output

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

## What it shows

- LEDGER ends with the same timeline in all 100 orders and is right on every probe day. Write-time invalidation
  ends with 95 different timelines (89 with modality filtering), and is right on 61.5% of probe days on average
  (63.8% with filtering, 24.5% and 23.5% in the worst orders).
- Even in chronological order, plain write-time invalidation is wrong on 15% of probe days: it takes a plan, a
  reminiscence and someone else's claim for moves. Modality filtering removes most of that (3% wrong) but not the
  dependence on order.
- On the toy, LEDGER's answers about the final state (current, point in time, change) are identical in order and
  shuffled (98.3, 94.4 and 73.6), as the report's order-invariance result requires; as-of and learned-at answers
  depend on arrival times by design. Write-time invalidation loses 12.5 points overall when ingest is shuffled
  (72.1 to 59.6), and 16.2 with modality filtering (87.8 to 71.6).
- On change questions in order, write-time invalidation with modality filtering does slightly better than LEDGER
  (74.7 against 73.6): the switch penalty costs LEDGER some short segments.
