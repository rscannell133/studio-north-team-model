"""Chronological backtest harness.

Rebuilds the cohort table from raw records as of each cutoff (no later data),
forecasts the following months, and scores with the instructor's exact rule
(cohort_lab.challenge.score_predictions). Run inside the lab's uv environment
so the released sBG baseline (cohort_lab.model) is importable.

usage: python backtest.py DATA_DIR LAB_ROOT OUT_DIR [model ...]
"""
from __future__ import annotations
import json, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import load_tables  # noqa: E402
from features import cohort_age_summary, grid_from_summary, CELL  # noqa: E402

CUTOFFS = {'dec24': '2025-01-01', 'jun25': '2025-07-01'}
KEYS = CELL + ['calendar_month', 'age_month']


def setup(data_dir: Path):
    tables = load_tables(data_dir, json.loads((ROOT / 'submission-contract.json').read_text()))
    full = cohort_age_summary(tables, '2026-01-01')
    folds = {}
    for name, cut in CUTOFFS.items():
        hist = cohort_age_summary(tables, cut)
        grid = grid_from_summary(hist)
        grid = grid[grid.calendar_month < '2026-01-01'].reset_index(drop=True)
        truth = grid.merge(full[CELL + ['calendar_month', 'n_retained', 'net_revenue', 'contribution']],
                           on=CELL + ['calendar_month'], how='left', validate='one_to_one')
        assert truth.n_retained.notna().all()
        folds[name] = dict(hist=hist, grid=grid, truth=truth)
    folds['final'] = dict(hist=full, grid=grid_from_summary(full), truth=None)
    return folds


def baseline_sbg(hist, grid, seed, cache: Path):
    """Released hierarchical sBG + Gamma, refit on the cutoff history."""
    from cohort_lab import model as M
    frame = M._validate(hist.copy())
    if cache.exists():
        draws = pickle.loads(cache.read_bytes())
    else:
        t = time.time()
        draws, diag = M._posterior(frame, seed, 3000)
        cache.write_bytes(pickle.dumps(draws))
        print(f'  sBG fit {time.time()-t:.0f}s  max_rhat={diag["max_r_hat"]:.3f} div={diag["divergences"]}', flush=True)
    rows = []
    for key, g in frame.groupby(CELL, sort=True):
        last = g.iloc[-1]
        fut = grid[(grid.cohort_month == key[0]) & (grid.channel == key[1]) & (grid.primary_need == key[2])].copy()
        ages = fut.age_month.astype(int).tolist()
        retained, gross = M._future(draws, last, ages, float(last.n_retained))
        nr, cr = M._economic_ratios(frame, last, ages)
        fut['predicted_retained'] = retained.mean(0)
        fut['predicted_net_revenue'] = (gross * nr).mean(0)
        fut['predicted_contribution'] = (gross * cr).mean(0)
        rows.append(fut)
    return pd.concat(rows, ignore_index=True), draws


def score(pred, fold):
    from cohort_lab.challenge import score_predictions
    s = score_predictions(pred, fold['truth'], fold['grid'])
    return {k: s[k] for k in ['contribution_wape', 'portfolio_contribution_bias', 'retention_cell_wape']}


def cohort_errors(pred, fold):
    t = fold['truth'].groupby('cohort_month').contribution.sum()
    p = pred.groupby('cohort_month').predicted_contribution.sum()
    return pd.DataFrame({'actual': t, 'pred': p, 'pct_err': (p - t) / t})
