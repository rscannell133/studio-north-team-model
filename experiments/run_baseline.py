import sys, pickle
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import setup, baseline_sbg, score, cohort_errors

data, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
folds = setup(data)
pickle.dump(folds, open(out / 'folds.pkl', 'wb'))
for name in sys.argv[3:]:
    f = folds[name]
    pred, _ = baseline_sbg(f['hist'], f['grid'], 20261116, out / f'sbg_draws_{name}.pkl')
    pred.to_csv(out / f'baseline_{name}.csv', index=False)
    if f['truth'] is not None:
        print(name, score(pred, f))
        print(cohort_errors(pred, f).round(3).to_string())
    else:
        print(name, 'total 2026 contribution', round(pred.predicted_contribution.sum()))
