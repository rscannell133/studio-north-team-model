import sys, pickle, itertools
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest import score
import models as MD

BT = Path(sys.argv[1])
folds = pickle.load(open(BT / 'folds.pkl', 'rb'))
K = ['cohort_month', 'channel', 'primary_need', 'calendar_month']


def components(name):
    f = folds[name]; hist, grid = f['hist'], f['grid'].copy()
    base = pd.read_csv(BT / f'baseline_{name}.csv')
    grid = grid.merge(base[K + ['predicted_retained', 'predicted_net_revenue', 'predicted_contribution']], on=K)
    R = {'sbg': grid.predicted_retained.to_numpy()}
    R['glm'] = MD.predict_retained(MD.fit_hazard(hist), grid)
    R['glm_ct'] = MD.predict_retained(MD.fit_hazard(hist, cohort_trend=True), grid)
    R['avg'] = (R['sbg'] + R['glm']) / 2
    r = np.where(R['sbg'] > 0, R['sbg'], np.nan)
    V = {'base': (grid.predicted_net_revenue.to_numpy() / r, grid.predicted_contribution.to_numpy() / r)}
    vt = MD.fit_value(hist, calendar_trend=True)
    V['trend'] = MD.predict_value(vt, grid)
    V['trend_half'] = MD.predict_value(vt, grid, trend_damp=0.5)
    V['flat'] = MD.predict_value(MD.fit_value(hist, calendar_trend=False), grid)
    return grid, R, V


rows = []
for name in ['dec24', 'jun25']:
    grid, R, V = components(name)
    for (rn, ret), (vn, (npm, cpm)) in itertools.product(R.items(), V.items()):
        p = grid[K + ['age_month', 'n_acquired', 'origin_retained']].copy()
        p['predicted_retained'] = ret
        p['predicted_net_revenue'] = np.nan_to_num(ret * npm)
        p['predicted_contribution'] = np.nan_to_num(ret * cpm)
        s = score(p, folds[name])
        rows.append({'fold': name, 'retention': rn, 'value': vn, **s})
res = pd.DataFrame(rows)
tab = res.pivot_table(index=['retention', 'value'], columns='fold',
                      values=['contribution_wape', 'portfolio_contribution_bias'])
tab[('mean', 'wape')] = tab['contribution_wape'].mean(axis=1)
print(tab.sort_values(('mean', 'wape')).round(4).to_string())
res.to_csv(BT / 'variants.csv', index=False)
