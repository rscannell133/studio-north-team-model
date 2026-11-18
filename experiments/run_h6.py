import sys, json, itertools
from pathlib import Path
import numpy as np, pandas as pd
sys.argv = sys.argv[:2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_variants import components, folds, K
from backtest import score
import main, tiers

tables = main.load_tables(Path('/home/claude/src/lab/data'),
                          json.loads((Path(__file__).resolve().parents[1] / 'submission-contract.json').read_text()))
cut = {'dec24': '2025-01-01', 'jun25': '2025-07-01'}
rows = []
for name in ['dec24', 'jun25']:
    grid, R, V = components(name)
    m6 = tiers.fit(tables, cut[name], value_trend=True)
    r6, n6, c6 = tiers.predict(m6, grid)
    m6f = tiers.fit(tables, cut[name], value_trend=False)
    _, n6f, c6f = tiers.predict(m6f, grid)
    assert np.allclose(r6[grid.age_month == grid.groupby(K[:3]).age_month.transform('min')] * 0 + 1, 1)
    R2 = {'sbg': R['sbg'], 'glm': R['glm'], 'avg': R['avg'], 'h6': r6,
          'avg3': (R['sbg'] + R['glm'] + r6) / 3, 'sbg_h6': (R['sbg'] + r6) / 2}
    V2 = {'trend': V['trend'], 'h6_trend': (n6 / r6, c6 / r6), 'h6_flat': (n6f / r6, c6f / r6),
          'h6_half': tuple(x / r6 for x in tiers.predict(m6, grid, trend_damp=.5)[1:])}
    for (rn, ret), (vn, (npm, cpm)) in itertools.product(R2.items(), V2.items()):
        p = grid[K + ['age_month', 'n_acquired', 'origin_retained']].copy()
        p['predicted_retained'] = ret
        p['predicted_net_revenue'] = np.nan_to_num(ret * npm)
        p['predicted_contribution'] = np.nan_to_num(ret * cpm)
        rows.append({'fold': name, 'retention': rn, 'value': vn, **score(p, folds[name])})
    t = folds[name]['truth'].set_index(K).reindex(grid.set_index(K).index).n_retained.to_numpy()
    print(name, 'retention WAPE', {k: round(np.abs(v - t).sum() / t.sum(), 4) for k, v in R2.items()})
res = pd.DataFrame(rows)
tab = res.pivot_table(index=['retention', 'value'], columns='fold', values=['contribution_wape', 'portfolio_contribution_bias'])
tab[('mean', 'wape')] = tab['contribution_wape'].mean(axis=1)
print(tab.sort_values(('mean', 'wape')).round(4).head(14).to_string())
