"""Studio North holdout challenge entry: blended retention x trending member value.

Retention is a 50/50 average of the released hierarchical sBG and a
tenure-dependent hazard GLM; value per retained member follows tenure and
calendar-time trends. Chosen by chronological backtests (see README).
"""
from __future__ import annotations
import os
import sys
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', 'true')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

KEYS = ['cohort_month','channel','primary_need','calendar_month','age_month']
EXPOSURES = ['n_acquired','origin_retained']
PREDICTIONS = ['predicted_retained','predicted_net_revenue','predicted_contribution']


def predict(tables: dict[str,pd.DataFrame], prediction_grid: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Blended retention x trending value per member. See README for assumptions.

    1. Rebuild the cohort table from the observed records (features.py).
    2. Retained members: equal-weight average of the released hierarchical sBG
       (refit here with `seed`) and a tenure-dependent logistic hazard GLM.
    3. Net revenue and contribution per retained member: weighted log-linear /
       logit models in tenure, calendar time, channel and need.
    """
    import sbg_reference as SBG
    from features import cohort_age_summary
    import models as MD

    grid = prediction_grid.copy(deep=True)
    origin = pd.Timestamp(grid.calendar_month.min())
    hist = cohort_age_summary(tables, origin.strftime('%Y-%m-%d'))
    frame = SBG._validate(hist.copy())

    draws, _ = SBG._posterior(frame, seed, 3000)
    sbg = []
    for key, g in frame.groupby(KEYS[:3], sort=True):
        last = g.iloc[-1]
        fut = grid[(grid.cohort_month == key[0]) & (grid.channel == key[1]) & (grid.primary_need == key[2])]
        retained, _ = SBG._future(draws, last, fut.age_month.astype(int).tolist(), float(last.n_retained))
        sbg.append(pd.Series(retained.mean(axis=0), index=fut.index))
    sbg = pd.concat(sbg).reindex(grid.index).to_numpy()
    if np.isnan(sbg).any():
        raise ValueError('Grid cells missing from the observed cohort table')

    glm = MD.predict_retained(MD.fit_hazard(hist), grid)
    retained = 0.5 * sbg + 0.5 * glm

    import tiers
    tier_model = tiers.fit(tables, origin.strftime('%Y-%m-%d'), value_trend=True)
    tier_retained, tier_net, tier_contribution = tiers.predict(tier_model, grid)
    alive = tier_retained > 0
    if (alive != (grid.origin_retained.to_numpy() > 0)).any():
        raise ValueError('Tier model membership disagrees with origin survivors')
    net_pm = np.divide(tier_net, tier_retained, out=np.zeros(len(grid)), where=alive)
    contribution_pm = np.divide(tier_contribution, tier_retained, out=np.zeros(len(grid)), where=alive)
    grid['predicted_retained'] = retained
    grid['predicted_net_revenue'] = retained * net_pm
    grid['predicted_contribution'] = retained * contribution_pm
    return grid[KEYS+EXPOSURES+PREDICTIONS]


def load_tables(data_dir: Path, contract: dict) -> dict[str,pd.DataFrame]:
    expected = contract['input_tables']
    if {p.name for p in data_dir.glob('*.csv')} != {name+'.csv' for name in expected}:
        raise ValueError('Supply exactly the eight permitted observed CSVs')
    tables = {}
    for name, fields in expected.items():
        dtypes = {field:('string' if kind in {'VARCHAR','DATE','TIMESTAMP'} else
                         'Int64' if kind=='INTEGER' else 'float64') for field,kind in fields.items()}
        table = pd.read_csv(data_dir/(name+'.csv'),dtype=dtypes)
        if list(table.columns)!=list(fields):
            raise ValueError('Unexpected source schema: '+name)
        tables[name] = table
    return tables


def validate_output(predictions: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(predictions,pd.DataFrame) or set(predictions)!=set(KEYS+EXPOSURES+PREDICTIONS):
        raise ValueError('Return exactly the required grid and prediction columns')
    if predictions.duplicated(KEYS).any():
        raise ValueError('Duplicate prediction keys')
    ordered = predictions.sort_values(KEYS).reset_index(drop=True)
    expected = grid.sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(ordered[KEYS+EXPOSURES],expected[KEYS+EXPOSURES],check_dtype=False)
    values = ordered[PREDICTIONS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values<0).any():
        raise ValueError('Predictions must be finite and nonnegative')
    if (ordered.predicted_retained>ordered.origin_retained+1e-8).any():
        raise ValueError('Predicted retained count exceeds origin survivors')
    if (ordered.groupby(KEYS[:3]).predicted_retained.diff().dropna()>1e-8).any():
        raise ValueError('Expected membership must not increase')
    if ((ordered.predicted_retained==0) & ((ordered.predicted_net_revenue>0)|(ordered.predicted_contribution>0))).any():
        raise ValueError('Zero expected members must have zero expected receipts')
    if (ordered.predicted_contribution>ordered.predicted_net_revenue+1e-8).any():
        raise ValueError('Contribution exceeds net revenue')
    return ordered[KEYS+EXPOSURES+PREDICTIONS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--prediction-grid',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261116)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.receipt.json').exists():
        parser.error('Use a new output path; existing forecasts are preserved')
    contract_path = Path(__file__).with_name('submission-contract.json')
    contract = json.loads(contract_path.read_text())
    tables = load_tables(args.data_dir,contract)
    grid = pd.read_csv(args.prediction_grid)
    started = time.monotonic()
    predictions = validate_output(predict(tables,grid.copy(deep=True),seed=args.seed),grid)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    predictions.to_csv(args.output,index=False,float_format='%.10f',lineterminator='\n')
    sha = lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    receipt = {'model':'50/50 sBG + tenure-GLM retention x calendar-trend value per member',
               'rows':len(predictions),'seed':args.seed,'elapsed_seconds':time.monotonic()-started,
               'source_sha256':sha(__file__),'uv_lock_sha256':sha(Path(__file__).with_name('uv.lock')),
               'contract_sha256':sha(contract_path),'prediction_grid_sha256':sha(args.prediction_grid),
               'input_sha256':{name:sha(args.data_dir/(name+'.csv')) for name in tables},
               'prediction_sha256':sha(args.output),'future_outcomes_read':False}
    args.output.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'validated','rows':len(predictions),'output':str(args.output)}))


if __name__ == '__main__':
    main()
