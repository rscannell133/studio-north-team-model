import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import load_tables
from features import cohort_age_summary, grid_from_summary

data, ref = Path(sys.argv[1]), Path(sys.argv[2])
tables = load_tables(data, json.loads((ROOT / 'submission-contract.json').read_text()))
mine = cohort_age_summary(tables, '2026-01-01')
theirs = pd.read_csv(ref / 'tables/cohort_age_summary.csv')
k = ['cohort_month', 'channel', 'primary_need', 'age_month']
j = mine.merge(theirs, on=k, suffixes=('', '_ref'), how='outer', indicator=True)
print('rows', len(mine), len(theirs), j._merge.value_counts().to_dict())
for c in ['cohort_index','n_acquired', 'n_retained', 'gross_payment_revenue','net_revenue', 'contribution']:
    print(c, 'max abs diff', float(np.abs(j[c] - j[c + '_ref']).max()))
g = grid_from_summary(mine)
pg = pd.read_csv(ROOT / 'prediction_grid.csv')
print('grid equal', g.sort_values(k[:3] + ['calendar_month']).reset_index(drop=True)
      .equals(pg.sort_values(k[:3] + ['calendar_month']).reset_index(drop=True)[g.columns]))
