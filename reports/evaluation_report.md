# Phase 6: System Evaluation & Benchmark Report

Evaluated with stand-in benchmark `evaluate.py`.

```text
========================================================
EMERGENCY RECALL (n=1213): 1.0000   <-- headline metric
  missed emergencies: 0
Safety score        : 0.4592
Mean cost / patient : 1.683  (all-level-3 baseline 3.112)
Plain accuracy      : 0.2930  (not what you're scored on)
Department accuracy : 0.7666
Abstention rate     : 0.0248   precision on should_abstain: 0.743   recall: 1.000
Over-triage rate    : 0.7070   Under-triage rate: 0.0000
--------------------------------------------------------
Wait time (min), 2 doctors:       FCFS     Triage   Reduction
  all patients               62.0     73.5    -18.5%
  true emergencies           62.1      6.4     89.6%
========================================================

```
