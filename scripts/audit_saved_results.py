"""Print descriptive summaries of the checked-in study and control CSVs."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "results/Krishs-MacBook-Air/all/study_single.csv",
    "results/Krishs-MacBook-Air/all/study_h3.csv",
    "results/athena/controls/control_cap_noreg.csv",
    "results/athena/controls/control_cap_nmin1.csv",
)
METRICS = ("W", "kills_per_ep", "self_bankr_per_ep", "survivors", "firm0_profit")

for name in FILES:
    df = pd.read_csv(ROOT / name)
    grouped = df.groupby("tag", sort=True)
    summary = grouped[list(METRICS)].mean()
    summary.insert(0, "seeds", grouped["seed"].nunique())
    print(f"\n{name}\n{summary.to_string(float_format=lambda x: f'{x:.4f}')}")
