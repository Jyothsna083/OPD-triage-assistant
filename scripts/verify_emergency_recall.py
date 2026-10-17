"""
scripts/verify_emergency_recall.py
Outcome 4 Verification: Emergency recall reported separately and prominently
on the ground-truth hidden labels dataset.
"""

import sys
import os
import pandas as pd
import numpy as np

def verify_emergency_recall():
    pred = pd.read_csv("predictions.csv")
    lab = pd.read_csv("data/hidden_labels.csv")

    df = lab.merge(pred, on="patient_id", how="left")
    if "abstain" not in df:
        df["abstain"] = 0
    df["abstain"] = df["abstain"].fillna(0)
    df["pred_level"] = df["triage_level_y"].fillna(3).astype(int)

    em = df["is_emergency"] == 1
    ab = df["abstain"] == 1
    caught = ((df["pred_level"] <= 2) | ab)[em]

    total_emergencies = int(em.sum())
    caught_emergencies = int(caught.sum())
    missed_emergencies = total_emergencies - caught_emergencies
    recall = caught.mean()

    print("=" * 65)
    print("OUTCOME 4: EMERGENCY RECALL ON GROUND-TRUTH EVALUATION SET")
    print("=" * 65)
    print(f"Total True Emergencies (is_emergency == 1): {total_emergencies}")
    print(f"Successfully Caught (Level <= 2 or Abstained): {caught_emergencies}")
    print(f"Missed Emergencies:                           {missed_emergencies}")
    print(f"EMERGENCY RECALL RATE:                        {recall:.6f} ({recall*100:.2f}%)")
    print("=" * 65)

    print("\nRecall Breakdown by Emergency Archetype:")
    emerg_df = df[em]
    for arch, g in emerg_df.groupby("archetype"):
        arch_caught = ((g["pred_level"] <= 2) | (g["abstain"] == 1)).sum()
        print(f"  {arch:<26} n={len(g):<4} Caught={arch_caught:<4} Recall={arch_caught/len(g)*100:.1f}%")

    print("\nTriage Level Distribution for True Emergencies:")
    print(pd.crosstab(emerg_df["triage_level_x"], emerg_df["pred_level"], rownames=["True Level"], colnames=["Pred Level"]))
    print("=" * 65)

    return recall, missed_emergencies

if __name__ == "__main__":
    verify_emergency_recall()
