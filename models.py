"""Alternative model components (numpy only).

Retention ("tenure GLM"): discrete-time logistic hazard of cancellation at each
renewal, logit h = b0 + b1*log(age) + b2*log(age)^2 + channel + need
[+ cohort trend], fit by binomial maximum likelihood on survivor counts.
Duration dependence enters directly through log(age) (cf. the beta-discrete-
Weibull extension of sBG, Fader et al. 2018) instead of only through
unobserved heterogeneity.

Value per retained member: log(net revenue per member) and logit(contribution
margin) are weighted linear models in log1p(age), calendar time, channel and
need. Calendar time captures the observed rise in price/margin that the
released baseline holds fixed. Margin is bounded in (0,1), so contribution
never exceeds net revenue by construction.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

CELL = ['cohort_month', 'channel', 'primary_need']
T0 = pd.Period('2024-01', 'M').ordinal


def _month_index(dates) -> np.ndarray:
    return np.array([pd.Period(d, 'M').ordinal - T0 for d in dates], dtype=float)


def _dummies(df):
    return np.column_stack([(df.primary_need == 'build_skill'), (df.channel == 'partner'),
                            (df.channel == 'organic')]).astype(float)


# ---------------- retention -------------------------------------------------
def _hazard_x(df, cohort_trend: bool):
    la = np.log(df.age_month.to_numpy(float))
    cols = [np.ones(len(df)), la, la ** 2, _dummies(df)]
    if cohort_trend:
        cols.append(((df.cohort_index.to_numpy(float) - 11.5) / 11.5)[:, None])
    return np.column_stack(cols)


def fit_hazard(hist: pd.DataFrame, cohort_trend: bool = False, ridge: float = 1e-3):
    f = hist.sort_values(CELL + ['age_month']).copy()
    f['prev'] = f.groupby(CELL).n_retained.shift(1)
    h = f[(f.age_month > 0) & (f.prev > 0)]
    x = _hazard_x(h, cohort_trend)
    n = h.prev.to_numpy(float)
    y = (h.prev - h.n_retained).to_numpy(float)  # cancellations
    b = np.zeros(x.shape[1]); b[0] = -2.0
    for _ in range(100):  # Newton-Raphson / IRLS, tiny ridge for stability
        p = 1 / (1 + np.exp(-(x @ b)))
        g = x.T @ (y - n * p) - ridge * b
        H = (x * (n * p * (1 - p))[:, None]).T @ x + ridge * np.eye(len(b))
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-10:
            break
    return {'beta': b, 'cohort_trend': cohort_trend}


def predict_retained(model, grid: pd.DataFrame) -> np.ndarray:
    g = grid.sort_values(CELL + ['age_month'])
    if 'cohort_index' not in g:
        g = g.assign(cohort_index=_month_index(g.cohort_month))
    h = 1 / (1 + np.exp(-(_hazard_x(g, model['cohort_trend']) @ model['beta'])))
    surv = pd.Series(1 - h, index=g.index).groupby([g[c] for c in CELL]).cumprod()
    return (g.origin_retained * surv).reindex(grid.index).to_numpy()


# ---------------- value per retained member ---------------------------------
def _value_x(df, calendar_trend: bool):
    cols = [np.ones(len(df)), np.log1p(df.age_month.to_numpy(float)), _dummies(df)]
    if calendar_trend:
        cols.append((_month_index(df.calendar_month) / 12)[:, None])
    return np.column_stack(cols)


def _wls(x, y, w, ridge=1e-6):
    xw = x * w[:, None]
    return np.linalg.solve(x.T @ xw + ridge * np.eye(x.shape[1]), xw.T @ y)


def fit_value(hist: pd.DataFrame, calendar_trend: bool = True, min_age: int = 1):
    v = hist[(hist.age_month >= min_age) & (hist.n_retained > 0) &
             (hist.net_revenue > 0) & (hist.contribution > 0)]
    x = _value_x(v, calendar_trend)
    w = v.n_retained.to_numpy(float)
    net_pm = np.log(v.net_revenue / v.n_retained).to_numpy()
    m = (v.contribution / v.net_revenue).clip(1e-3, 1 - 1e-3).to_numpy()
    b_net = _wls(x, net_pm, w)
    b_mar = _wls(x, np.log(m / (1 - m)), w)
    # smearing factor so exp(mean log) is unbiased for the mean
    resid = net_pm - x @ b_net
    smear = float(np.sum(w * np.exp(resid)) / w.sum())
    return {'b_net': b_net, 'b_mar': b_mar, 'smear': smear, 'calendar_trend': calendar_trend}


def predict_value(model, grid: pd.DataFrame, trend_damp: float = 1.0):
    g = grid.copy()
    x = _value_x(g, model['calendar_trend'])
    b_net, b_mar = model['b_net'].copy(), model['b_mar'].copy()
    if model['calendar_trend'] and trend_damp != 1.0:
        # damp only the extrapolated part beyond the forecast origin
        origin = _month_index(g.calendar_month).min() - 1
        x = x.copy(); t = x[:, -1] * 12
        x[:, -1] = (origin + trend_damp * (t - origin)) / 12
    net_pm = np.exp(x @ b_net) * model['smear']
    margin = 1 / (1 + np.exp(-(x @ b_mar)))
    return net_pm, net_pm * margin
