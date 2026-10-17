#!/usr/bin/env python3
"""
Stand-in evaluator (NOT the organizers' evaluate.py) for the synthetic OPD dataset.

predictions.csv columns: patient_id, triage_level (1-5), department, abstain (0/1, optional)
Usage: python evaluate.py predictions.csv hidden_labels.csv [arrival_log.csv]

Cost per patient (lower is better):
  abstain ("send to nurse now"): 0.3 if should_abstain else 1.5  (safe, but wasteful if unneeded)
  over-triage  (pred more urgent than truth): 1 per level
  under-triage (pred less urgent than truth): 4 per level; 20 per level if true emergency
  wrong department: +1 (+3 if true emergency)
Safety score = 1 - mean_cost / mean_cost(predict level 3 for everyone), floored at 0.
Emergency recall = share of true emergencies given level <=2 or abstained.
"""
import sys, heapq
import numpy as np
import pandas as pd


def costs(df):
    ab = df["abstain"].fillna(0).astype(int) == 1
    gap = df["pred_level"] - df["triage_level"]          # >0 => under-triage
    emerg = df["is_emergency"] == 1
    c = np.where(gap > 0, gap * np.where(emerg, 20, 4), -gap * 1.0)
    c = c + np.where(df["pred_dept"] != df["department"], np.where(emerg, 3, 1), 0)
    c = np.where(ab, np.where(df["should_abstain"] == 1, 0.3, 1.5), c)
    return c


def simulate(arr, order_key, doctors=2):
    """Per-day queue sim: doctors take the next waiting patient by order_key (lower = sooner)."""
    waits = pd.Series(0.0, index=arr.index)
    for _, g in arr.groupby("date"):
        g = g.sort_values("arrival_sec_of_day")
        free = [0.0] * doctors; heapq.heapify(free)
        pending = list(g.index); waiting = []; i = 0; t = 0.0
        arr_t = g["arrival_sec_of_day"] / 60.0
        while i < len(pending) or waiting:
            doc_free = heapq.heappop(free)
            t = max(t, doc_free)
            if not waiting and i < len(pending) and arr_t[pending[i]] > t:
                t = arr_t[pending[i]]
            while i < len(pending) and arr_t[pending[i]] <= t:
                idx = pending[i]; heapq.heappush(waiting, (order_key[idx], arr_t[idx], idx)); i += 1
            _, a_t, idx = heapq.heappop(waiting)
            waits[idx] = t - a_t
            heapq.heappush(free, t + g.at[idx, "consult_minutes"])
    return waits


def main():
    pred = pd.read_csv(sys.argv[1]); lab = pd.read_csv(sys.argv[2])
    df = lab.merge(pred, on="patient_id", how="left")
    if "abstain" not in df: df["abstain"] = 0
    df["abstain"] = df["abstain"].fillna(0)
    df["pred_level"] = df["triage_level_y"].fillna(3).astype(int) if "triage_level_y" in df else df["triage_level"]
    df["triage_level"] = df["triage_level_x"] if "triage_level_x" in df else df["triage_level"]
    df["pred_dept"] = df["department_y"] if "department_y" in df else df["department"]
    df["department"] = df["department_x"] if "department_x" in df else df["department"]

    df["cost"] = costs(df)
    base = df.assign(pred_level=3, abstain=0, pred_dept="General Medicine")
    base_cost = costs(base).mean()
    ab = df["abstain"] == 1
    em = df["is_emergency"] == 1
    caught = ((df["pred_level"] <= 2) | ab)[em]
    print("=" * 56)
    print(f"EMERGENCY RECALL (n={em.sum()}): {caught.mean():.4f}   <-- headline metric")
    print(f"  missed emergencies: {(~caught).sum()}")
    print(f"Safety score        : {max(0, 1 - df['cost'].mean() / base_cost):.4f}")
    print(f"Mean cost / patient : {df['cost'].mean():.3f}  (all-level-3 baseline {base_cost:.3f})")
    print(f"Plain accuracy      : {(df['pred_level'] == df['triage_level'])[~ab].mean():.4f}  (not what you're scored on)")
    print(f"Department accuracy : {(df['pred_dept'] == df['department'])[~ab].mean():.4f}")
    print(f"Abstention rate     : {ab.mean():.4f}   precision on should_abstain: "
          f"{(df['should_abstain'][ab] == 1).mean() if ab.any() else float('nan'):.3f}   "
          f"recall: {ab[df['should_abstain'] == 1].mean():.3f}")
    print(f"Over-triage rate    : {(df['pred_level'] < df['triage_level'])[~ab].mean():.4f}   "
          f"Under-triage rate: {(df['pred_level'] > df['triage_level'])[~ab].mean():.4f}")

    if len(sys.argv) > 3:
        arr = pd.read_csv(sys.argv[3]).merge(df[["patient_id", "pred_level", "abstain", "is_emergency"]], on="patient_id")
        arr["prio"] = np.where(arr["abstain"] == 1, 0, arr["pred_level"])   # abstain -> nurse now
        fcfs = simulate(arr, pd.Series(0, index=arr.index))
        tri = simulate(arr, arr["prio"])
        e = arr["is_emergency"] == 1
        print("-" * 56)
        print("Wait time (min), 2 doctors:       FCFS     Triage   Reduction")
        for name, m in [("all patients", slice(None)), ("true emergencies", e)]:
            a, b = fcfs[m].mean(), tri[m].mean()
            print(f"  {name:<22}{a:9.1f}{b:9.1f}{(1 - b / a) * 100:9.1f}%")
    print("=" * 56)


if __name__ == "__main__":
    main()
