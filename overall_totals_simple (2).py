#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple total consumption per meter (robust; tolerant headers/encodings):
- Reads all CSVs from ABSOLUTE_DATA_DIR below.
- Uses ';' as separator.
- Skips empty CSVs gracefully.
- Accepts slightly different header names (whitespace/BOM/extra text).
- For each meter: total = last_cum - first_cum across ALL files.
- If a reset makes this negative, falls back to sum of positive diffs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import glob
from pandas.errors import EmptyDataError

# --- CONFIG ---
ABSOLUTE_DATA_DIR = r"C:\Users\user\PycharmProjects\gigahack_BITenergy\cod\data"
SEP = ";"
DAYFIRST = True

def _norm(s: str) -> str:
    if s is None:
        return s
    s = str(s).replace("\ufeff", "")  # drop BOM
    s = s.strip()
    s = re.sub(r"\s+", " ", s)  # collapse spaces/tabs
    return s

def find_col(df, patterns):
    # normalize header names once
    df.columns = [_norm(c) for c in df.columns]
    for pat in patterns:
        rx = re.compile(pat, flags=re.I)
        for c in df.columns:
            if rx.search(c):
                return c
    return None

def try_read_csv(file):
    encodings = ["utf-8-sig", "utf-8", "cp1251", "latin1"]
    for enc in encodings:
        try:
            return pd.read_csv(file, sep=SEP, encoding=enc, engine="python")
        except EmptyDataError:
            print(f"[SKIP empty] {Path(file).name}")
            return None
        except Exception:
            continue
    print(f"[WARN] Could not read {Path(file).name} with tried encodings.")
    return None

def main():
    base = Path(ABSOLUTE_DATA_DIR)
    files = sorted(glob.glob(str(base / "*.csv")))
    if not files:
        print(f"No CSVs found in {base}")
        return

    frames = []
    for f in files:
        df = try_read_csv(f)
        if df is None or df.empty:
            continue

        meter_col = find_col(df, [r"\bmeter\b"])
        time_col  = find_col(df, [r"\bclock\b", r"\bdate\b", r"\btime\b", r"\btimestamp\b"])
        imp_col   = find_col(df, [r"active\s*energy\s*import", r"\b1\.8\.0\b", r"\bimport\b"])

        if not (meter_col and time_col and imp_col):
            print(f"[WARN] Skipping {Path(f).name}: can't find needed columns; columns = {list(df.columns)}")
            continue

        w = df[[meter_col, time_col, imp_col]].copy()
        w.columns = ["meter","ts","imp_cum"]
        w["ts"] = pd.to_datetime(w["ts"], dayfirst=DAYFIRST, errors="coerce")
        w["imp_cum"] = pd.to_numeric(w["imp_cum"], errors="coerce")
        w = w.dropna(subset=["meter","ts","imp_cum"])
        if w.empty:
            print(f"[WARN] Skipping {Path(f).name}: no valid rows after cleanup")
            continue
        w["meter"] = w["meter"].astype(str)
        frames.append(w)

    if not frames:
        print("No usable data found across CSVs.")
        return

    all_df = pd.concat(frames, ignore_index=True).sort_values(["meter","ts"])

    firsts = all_df.groupby("meter").first(numeric_only=True)["imp_cum"]
    lasts  = all_df.groupby("meter").last(numeric_only=True)["imp_cum"]
    simple_total = (lasts - firsts).rename("overall_by_range")

    robust = (all_df.groupby("meter")["imp_cum"]
                    .apply(lambda s: s.diff().clip(lower=0).sum())
                    .rename("robust_sum_diffs"))

    out = pd.concat([firsts.rename("first_cum"),
                     lasts.rename("last_cum"),
                     simple_total, robust], axis=1).reset_index()

    out["used_total"] = np.where(out["overall_by_range"] >= 0,
                                 out["overall_by_range"], out["robust_sum_diffs"])

    out = (out.merge(all_df.groupby("meter")["ts"].first().rename("first_ts"), on="meter")
              .merge(all_df.groupby("meter")["ts"].last().rename("last_ts"), on="meter"))

    out = out[["meter","first_ts","last_ts","first_cum","last_cum",
               "overall_by_range","robust_sum_diffs","used_total"]]

    out_file = Path.cwd() / "overall_totals_simple.csv"
    out.to_csv(out_file, index=False)
    print(f"Wrote {out_file} with {len(out)} meters.")

if __name__ == "__main__":
    main()
