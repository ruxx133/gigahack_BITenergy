#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Analysis & Forecasting from cumulative smart meters (15-min)
----------------------------------------------------------------
- Reads CSVs with cumulative 1.8.x (import) and 2.8.x (export) registers.
- Builds 15-min Wh by differencing cumulative counters (handles resets).
- For EXPORT: detects *all* 2.8.x columns (2.8.0, 2.8.1, …) and sums their interval diffs.
- Converts to W with ×4 (avg power over 15-min). No kWh inflation.
- Cleans (dedupe, 15-min grid, <=1 step interpolation, outlier caps).
- Features (no leakage): calendar, lags (1,2,96,97,672), rolling mean/std (4,8,96),
  export/daylight/reset/outage flags, meter id.
- Optional rolling-origin CV (LightGBM preferred, RF fallback).
- Diagnostics print what export columns were detected and whether they actually vary.

Outputs: ./out/interval_15m_W.csv, ./out/daily_totals_kWh.csv, plots, and (if enabled) metrics/predictions.
"""

import glob, re, warnings
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rmse_compat(y_true, y_pred):
    # works with any sklearn version
    from sklearn.metrics import mean_squared_error
    import numpy as np
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# ================= CONFIG =================
ABSOLUTE_DATA_DIR = r"C:\Users\user\PycharmProjects\gigahack_BITenergy\cod\data"
PATTERN           = "*.csv"
SEP               = ";"
DAYFIRST          = True

# Modeling target units
TARGET_UNITS      = "W"   # "W" (avg over 15-min) or "kWh_15m"

# OBIS / headers
FORCE_EXPORT_COL  = "Active Energy Export (3:1-0:2.8.0*255:2)"  # your exact header (keep)
FORCE_IMPORT_COL  = None  # e.g., "Active Energy Import (3:1-0:1.8.0*255:2)" if you want to force

# Apply transformer coefficient if you KNOW it's needed
APPLY_TRANS_COEF  = False

# Reset threshold in Wh: big negative diff => reset
RESET_NEG_THRESH_WH = 20000.0  # ~ -20 kWh

# Cap outliers per meter at percentile
CAP_PCTL          = 0.999

# Fast/verbose
VERBOSE           = True
LIMIT_FILES: Optional[int]  = None
LIMIT_METERS: Optional[int] = None

# Modeling knobs
SKIP_MODEL        = False               # set False to train
SKIP_QUANTILES    = True
REDUCE_ESTIMATORS = True
N_SPLITS          = 2
# ==========================================

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=FutureWarning)
ENCODINGS = ["utf-8-sig","utf-8","cp1251","latin1"]

# ---------------- helpers ----------------
def _norm(s):
    if s is None: return s
    s = str(s).replace("\ufeff","").strip()
    return re.sub(r"\s+", " ", s)

def find_col(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    df.columns = [_norm(c) for c in df.columns]
    for pat in patterns:
        rx = re.compile(pat, re.I)
        for c in df.columns:
            if rx.search(c): return c
    return None

def find_all_cols(df: pd.DataFrame, patterns: List[str]) -> List[str]:
    """Return all column names that match ANY of the regex patterns."""
    df.columns = [_norm(c) for c in df.columns]
    out = []
    for c in df.columns:
        for pat in patterns:
            if re.search(pat, c, flags=re.I):
                out.append(c); break
    return out

def try_read_csv(path: str) -> Optional[pd.DataFrame]:
    from pandas.errors import EmptyDataError
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, sep=SEP, encoding=enc, engine="python")
        except EmptyDataError:
            print(f"[SKIP empty] {Path(path).name}")
            return None
        except Exception:
            continue
    print(f"[WARN] Could not read {Path(path).name} with tried encodings.")
    return None

def make_grid_15min(s: pd.Series):
    start = s.min().floor("15min"); end = s.max().ceil("15min")
    return pd.date_range(start, end, freq="15min")

def cap_by_percentile_per_meter(df, col, p=CAP_PCTL):
    def cap(s):
        q = s.quantile(p)
        return np.clip(s, 0, q) if np.isfinite(q) and q>0 else s
    return df.groupby("meter")[col].transform(cap)

def wMAPE(y_true, y_pred, eps=1e-9):
    denom = float(np.sum(np.abs(y_true)) + eps)
    return float(np.sum(np.abs(y_true - y_pred)) / denom)

# ------------- loader (handles multi export) -------------
def load_cumulative_all():
    base = Path(ABSOLUTE_DATA_DIR)
    files = sorted(glob.glob(str(base / PATTERN)))
    if not files:
        raise SystemExit(f"No CSVs found in: {ABSOLUTE_DATA_DIR}\\{PATTERN}")
    if LIMIT_FILES:
        files = files[:int(LIMIT_FILES)]
    if VERBOSE:
        print(f"[load] scanning {len(files)} files from {ABSOLUTE_DATA_DIR}")

    frames = []
    for i, f in enumerate(files, 1):
        if VERBOSE and (i % 5 == 0 or i == 1):
            print(f"  - reading file {i}/{len(files)}: {Path(f).name}")
        df = try_read_csv(f)
        if df is None or df.empty:
            continue

        df.columns = [_norm(c) for c in df.columns]

        meter_col = find_col(df, [r"\bmeter\b"])
        time_col  = find_col(df, [r"\bclock\b", r"\bts\b", r"\bdate\b", r"\btime\b", r"\btimestamp\b"])

        # IMPORT detection
        if FORCE_IMPORT_COL and FORCE_IMPORT_COL in map(str, df.columns):
            imp_col = FORCE_IMPORT_COL
        else:
            imp_col = find_col(df, [r"^active\s*energy\s*import\b",
                                    r"\b1[.:,]8[.:,]\d\b",   # 1.8.x
                                    r"\bimport\b"])

        # EXPORT detection:
        # 1) exact forced header
        exp_cols = []
        if FORCE_EXPORT_COL and FORCE_EXPORT_COL in map(str, df.columns):
            exp_cols = [FORCE_EXPORT_COL]

        # 2) collect ALL 2.8.x variants (2.8.0 .. 2.8.9), or generic "Active Energy Export"
        exp_cols += [c for c in find_all_cols(df, [
            r"^active\s*energy\s*export\b",
            r"\b2[.:,]8[.:,][0-9]\b",                # 2.8.0..9
            r"active\s*energy\s*export.*2[.:,]8[.:,][0-9]",
            r"\bA-\b", r"\bdelivered\b", r"\bto\s*grid\b", r"\bexport\b"
        ]) if c not in exp_cols]

        if not (meter_col and time_col and imp_col):
            if VERBOSE:
                print(f"    [warn] {Path(f).name}: missing meter/time/import; saw: {list(df.columns)[:8]}")
            continue

        # Normalize decimals for numeric-looking text columns
        numeric_candidates = [imp_col] + exp_cols
        for c in numeric_candidates:
            if c and c in df.columns and df[c].dtype == object:
                # replace decimal comma with dot and strip spaces
                s = df[c].astype(str).str.replace(" ", "", regex=False)
                s = s.str.replace(",", ".", regex=False)
                df[c] = s

        # Build frame w
        keep = [meter_col, time_col, imp_col]
        w = df[keep].copy()
        w.columns = ["meter","ts","imp_cum"]

        # attach ALL export cumulatives (rename to exp_cum_*)
        for k, c in enumerate(exp_cols, 1):
            w[f"exp_cum_{k}"] = df[c]

        # Transformer coefficient?
        all_num_cols = ["imp_cum"] + [c for c in w.columns if c.startswith("exp_cum_")]
        if APPLY_TRANS_COEF:
            coef_col = find_col(df, [r"\bTransFullCoef\b", r"\bCT|VT|coef|coefficient\b"])
            if coef_col and coef_col in df.columns:
                coef = pd.to_numeric(df[coef_col], errors="coerce").fillna(1.0)
                for c in all_num_cols:
                    w[c] = pd.to_numeric(w[c], errors="coerce") * coef
            else:
                for c in all_num_cols:
                    w[c] = pd.to_numeric(w[c], errors="coerce")
        else:
            for c in all_num_cols:
                w[c] = pd.to_numeric(w[c], errors="coerce")

        # Parse time & cleanup
        w["ts"] = pd.to_datetime(w["ts"], dayfirst=DAYFIRST, errors="coerce")
        w = w.dropna(subset=["meter","ts","imp_cum"])
        w["meter"] = w["meter"].astype(str)

        if VERBOSE:
            exp_info = [c for c in w.columns if c.startswith("exp_cum_")]
            print(f"    [ok] import='{imp_col}'  exports_detected={len(exp_info)} -> {exp_info[:3]}{'...' if len(exp_info)>3 else ''}")

        frames.append(w)

    if not frames:
        raise SystemExit("No usable data across CSVs.")

    all_df = (pd.concat(frames, ignore_index=True)
                .sort_values(["meter","ts"])
                .drop_duplicates(subset=["meter","ts"]))

    if LIMIT_METERS:
        top_meters = all_df["meter"].value_counts().head(int(LIMIT_METERS)).index.tolist()
        all_df = all_df[all_df["meter"].isin(top_meters)]
        if VERBOSE:
            print(f"[load] limited to {len(top_meters)} meters (most rows)")

    if VERBOSE:
        print(f"[load] rows={len(all_df):,}, meters={all_df['meter'].nunique()}")
    return all_df

# ----------- 15-min Wh from cumulative (multi-export aware) -----------
def per_interval_wh_from_cumulative(df: pd.DataFrame):
    """Compute 15-min Wh by differencing cumulative counters; handle resets.
       For export, sum diffs across ALL exp_cum_* columns (tariff splits)."""
    df = df.copy()

    def diffs_series(s: pd.Series) -> pd.Series:
        d = s.diff()
        d = d.mask(d < -abs(RESET_NEG_THRESH_WH), np.nan)  # big negative => reset
        return d.clip(lower=0)                              # tiny negatives => 0

    # group once (avoid include-groups deprecation noise)
    g = df.groupby("meter", group_keys=False)

    # Import
    imp_diff = g["imp_cum"].apply(diffs_series)
    df["imp_15m_Wh"] = imp_diff.to_numpy()

    # Export: sum interval diffs across all exp_cum_* columns (if any)
    exp_cols = [c for c in df.columns if c.startswith("exp_cum_")]
    if exp_cols:
        parts = []
        for c in exp_cols:
            parts.append(g[c].apply(diffs_series).rename(c))
        # align by index, then row-sum
        exp_diff_df = pd.concat(parts, axis=1)
        df["exp_15m_Wh"] = exp_diff_df.sum(axis=1).to_numpy()
    else:
        df["exp_15m_Wh"] = 0.0

    df["reset_flag"] = pd.isna(df["imp_15m_Wh"]).astype(int)
    return df


    def diffs(series):
        d = series.diff()
        d = d.mask(d < -abs(RESET_NEG_THRESH_WH), np.nan)  # big negative => reset
        d = d.clip(lower=0)                                # tiny negatives => 0
        return d

    # Import
    df["imp_15m_Wh"] = df.groupby("meter")["imp_cum"].apply(diffs).values

    # Export: collect all exp_cum_* columns
    exp_cols = [c for c in df.columns if c.startswith("exp_cum_")]
    if exp_cols:
        # Sum interval diffs across all export counters per meter
        exp_parts = []
        for c in exp_cols:
            part = df.groupby("meter")[c].apply(diffs)
            exp_parts.append(part)
        exp_sum = np.sum(exp_parts, axis=0)
        df["exp_15m_Wh"] = exp_sum.values
    else:
        df["exp_15m_Wh"] = 0.0

    # Flags
    df["reset_flag"] = df["imp_15m_Wh"].isna().astype(int)
    return df

# ----------------- clean & features -----------------
def clean_and_features(df15: pd.DataFrame):
    frames = []
    for m, g in df15.groupby("meter"):
        grid = make_grid_15min(g["ts"])
        gg = g.set_index("ts").reindex(grid)
        gg.index.name = "ts"
        gg["meter"] = m

        gg["was_missing"] = gg["imp_15m_Wh"].isna().astype(int)
        gg["reset_flag"]  = gg["reset_flag"].fillna(0).astype(int)

        gg["imp_15m_Wh"] = gg["imp_15m_Wh"].interpolate(limit=1)
        gg["exp_15m_Wh"] = gg["exp_15m_Wh"].interpolate(limit=1)

        frames.append(gg.reset_index())

    df = pd.concat(frames, ignore_index=True)

    # clip & cap
    for col in ["imp_15m_Wh","exp_15m_Wh"]:
        df[col] = df[col].clip(lower=0)
        df[col] = cap_by_percentile_per_meter(df, col)

    # target
    if TARGET_UNITS.upper() == "W":
        df["imp_TGT"] = df["imp_15m_Wh"] * 4.0  # Wh / 0.25 h
    else:
        df["imp_TGT"] = df["imp_15m_Wh"] / 1000.0

    # derived
    df["net_15m_Wh"] = df["imp_15m_Wh"] - df["exp_15m_Wh"]
    df["exporting"]  = (df["exp_15m_Wh"] > 0).astype(int)

    # calendar/daylight
    df = df.sort_values(["meter","ts"])
    df["hour"]    = df["ts"].dt.hour
    df["dow"]     = df["ts"].dt.dayofweek
    df["month"]   = df["ts"].dt.month
    df["weekend"] = (df["dow"]>=5).astype(int)
    df["daylight_flag"] = ((df["hour"]>=6) & (df["hour"]<=20)).astype(int)

    def zero_streak(s):
        z = (s <= 1e-9).astype(int)
        return (z.rolling(4, min_periods=1).sum() >= 4)
    df["outage_flag"] = df.groupby("meter")["imp_TGT"].apply(zero_streak).reset_index(level=0, drop=True).astype(int)

    # lags/rolling
    for L in [1, 2, 96, 97, 672]:
        df[f"lag_{L}"] = df.groupby("meter")["imp_TGT"].shift(L)
    for W in [4, 8, 96]:
        df[f"roll_mean_{W}"] = df.groupby("meter")["imp_TGT"].shift(1).rolling(W).mean()
        df[f"roll_std_{W}"]  = df.groupby("meter")["imp_TGT"].shift(1).rolling(W).std()

    df["meter_id"] = df["meter"].astype("category").cat.codes
    return df

# ----------------- diagnostics -----------------
def export_diagnostics(raw: pd.DataFrame, outdir: Path):
    exp_cols = [c for c in raw.columns if c.startswith("exp_cum_")]
    print("\n[diag] export columns detected:", exp_cols if exp_cols else "(none)")
    if not exp_cols:
        return
    present_ratio = raw[exp_cols].notna().any(axis=1).mean()
    print(f"[diag] rows with ANY export present: {present_ratio:.1%}")
    # measure movement across *all* export counters
    def movement(g):
        mv = 0.0
        for c in exp_cols:
            if c in g:
                s = g[c].dropna()
                if len(s) > 1:
                    mv += float(s.diff().abs().sum())
        return mv
    mv_per_meter = raw.groupby("meter", group_keys=False).apply(movement)
    moving = mv_per_meter[mv_per_meter > 0]
    print(f"[diag] meters with non-zero export movement: {len(moving)} / {raw['meter'].nunique()}")
    top = moving.sort_values(ascending=False).head(10)
    if not top.empty:
        print("[diag] top meters by export movement (sum of |Δ cumulative|):")
        print(top.to_string())

# ----------------- modeling -----------------
def main_modeling(df: pd.DataFrame, outdir: Path):
    if SKIP_MODEL:
        print("[model] SKIP_MODEL=True — not training (set to False when ready).")
        return

    target = "imp_TGT"
    feats  = [c for c in df.columns if c.startswith("lag_") or c.startswith("roll_")]
    feats += ["hour","dow","month","weekend","exporting","daylight_flag","reset_flag","outage_flag","meter_id"]

    if REDUCE_ESTIMATORS:
        if HAS_LGBM:
            def make_lgbm():
                return LGBMRegressor(
                    n_estimators=400, learning_rate=0.05,
                    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                    min_child_samples=60, random_state=42
                )
        else:
            def make_rf():
                return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        if HAS_LGBM:
            def make_lgbm():
                return LGBMRegressor(
                    n_estimators=1200, learning_rate=0.03,
                    num_leaves=64, subsample=0.8, colsample_bytree=0.8,
                    min_child_samples=80, random_state=42
                )
        else:
            def make_rf():
                return RandomForestRegressor(n_estimators=600, random_state=42, n_jobs=-1)

    work = df.dropna(subset=feats+[target]).sort_values("ts").reset_index(drop=True)
    X = work[feats].values
    y = work[target].values
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    fold_metrics = []
    last_pred_df = None
    for i, (tr, va) in enumerate(tscv.split(X), 1):
        model = make_lgbm() if HAS_LGBM else make_rf()
        if VERBOSE:
            print(f"[cv] fold {i}/{N_SPLITS}: train={tr[0]}..{tr[-1]}  val={va[0]}..{va[-1]}")
        model.fit(X[tr], y[tr])
        pred = model.predict(X[va])
        mae  = mean_absolute_error(y[va], pred)
        rmse = rmse_compat(y[va], pred)
        wm   = wMAPE(y[va], pred)
        fold_metrics.append((mae, rmse, wm))
        last_pred_df = work.iloc[va][["ts","meter"]].copy()
        last_pred_df["y_true"] = y[va]
        last_pred_df["y_pred"] = pred

    metrics = {
        "MAE": float(np.mean([m[0] for m in fold_metrics])),
        "RMSE": float(np.mean([m[1] for m in fold_metrics])),
        "wMAPE": float(np.mean([m[2] for m in fold_metrics])),
        "folds": [{"MAE": float(m[0]), "RMSE": float(m[1]), "wMAPE": float(m[2])} for m in fold_metrics],
        "model": "LightGBM" if HAS_LGBM else "RandomForest",
    }

    with open(outdir / "metrics.txt","w", encoding="utf-8") as f:
        f.write(f"Model: {metrics['model']}\n")
        f.write(f"CV MAE ({TARGET_UNITS}): {metrics['MAE']:.3f}\n")
        f.write(f"CV RMSE ({TARGET_UNITS}): {metrics['RMSE']:.3f}\n")
        f.write(f"CV wMAPE: {metrics['wMAPE']:.4f}\n")
        for j, m in enumerate(metrics["folds"], 1):
            f.write(f"Fold {j}: MAE={m['MAE']:.3f}, RMSE={m['RMSE']:.3f}, wMAPE={m['wMAPE']:.4f}\n")

    if last_pred_df is not None:
        last_pred_df.to_csv(outdir / f"predictions_last_fold_{TARGET_UNITS}.csv", index=False)

    if HAS_LGBM and not SKIP_QUANTILES:
        if VERBOSE: print("[quantiles] training P10/P50/P90 …")
        def fit_q(alpha):
            q = LGBMRegressor(
                objective="quantile", alpha=alpha,
                n_estimators=400 if REDUCE_ESTIMATORS else 1200,
                learning_rate=0.05 if REDUCE_ESTIMATORS else 0.03,
                num_leaves=31 if REDUCE_ESTIMATORS else 64,
                subsample=0.8, colsample_bytree=0.8,
                min_child_samples=60 if REDUCE_ESTIMATORS else 80,
                random_state=42
            )
            q.fit(X, y); return q
        p10 = fit_q(0.10).predict(X); p50 = fit_q(0.50).predict(X); p90 = fit_q(0.90).predict(X)
        if last_pred_df is not None:
            idx = last_pred_df.index
            last_pred_df["p10"] = p10[idx]; last_pred_df["p50"] = p50[idx]; last_pred_df["p90"] = p90[idx]
            last_pred_df.to_csv(outdir / f"predictions_last_fold_{TARGET_UNITS}.csv", index=False)
        print("[quantiles] saved.")

# ----------------- plots -----------------
def make_plots(df_feat: pd.DataFrame, outdir: Path):
    meters = list(df_feat["meter"].dropna().unique())[: (3 if LIMIT_METERS is None else min(3, LIMIT_METERS))]
    unit = TARGET_UNITS.upper()
    for m in meters:
        sub = df_feat[df_feat["meter"]==m].copy()
        prof = (sub.assign(mins=sub["ts"].dt.hour*60 + sub["ts"].dt.minute)
                  .groupby("mins", as_index=False)["imp_TGT"].mean())
        plt.figure()
        plt.plot(prof["mins"], prof["imp_TGT"])
        plt.title(f"Average daily profile – meter {m}")
        plt.xlabel("Minutes since 00:00"); plt.ylabel(f"Import ({unit})")
        plt.tight_layout(); plt.savefig(outdir / f"profile_avg_import_{unit}_{m}.png"); plt.close()

        ldc = sub["imp_TGT"].dropna().sort_values(ascending=False).reset_index(drop=True)
        plt.figure()
        plt.plot(ldc.index/len(ldc), ldc.values)
        plt.title(f"Load duration curve – meter {m}")
        plt.xlabel("Fraction of time"); plt.ylabel(f"Import ({unit})")
        plt.tight_layout(); plt.savefig(outdir / f"load_duration_import_{unit}_{m}.png"); plt.close()

# ----------------- main -----------------
def main():
    outdir = Path("out"); outdir.mkdir(exist_ok=True)

    raw = load_cumulative_all()
    export_diagnostics(raw, outdir)

    df15 = per_interval_wh_from_cumulative(raw)
    print("[build] computed 15-min Wh (import/export) from cumulative")

    df = clean_and_features(df15)
    print("[clean] features built")

    unit = TARGET_UNITS.upper()
    df.to_csv(outdir / f"interval_15m_{unit}.csv", index=False)

    daily = (df.assign(date=lambda x: x["ts"].dt.date)
               .groupby(["meter","date"], as_index=False)
               .agg(import_kWh=("imp_15m_Wh", lambda s: float(s.sum()/1000.0)),
                    export_kWh=("exp_15m_Wh", lambda s: float(s.sum()/1000.0))))
    daily["net_kWh"] = daily["import_kWh"] - daily["export_kWh"]
    daily.to_csv(outdir / "daily_totals_kWh.csv", index=False)

    print(f"[write] {outdir / f'interval_15m_{unit}.csv'}")
    print(f"[write] {outdir / 'daily_totals_kWh.csv'}")

    main_modeling(df, outdir)
    make_plots(df, outdir)

    print("Done. Outputs in ./out")
    for p in sorted(outdir.glob("*")):
        print(" -", p.name)

if __name__ == "__main__":
    main()
