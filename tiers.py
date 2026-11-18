"""H6: tier-aware member-state model (numpy only, deterministic).

Each retained member is in one of two plan tiers (standard, premium). Month to
month a member either renews or cancels (absorbing), with a cancellation
hazard that depends on tier, tenure, channel and need; renewers may switch
tier with tenure-dependent probabilities. Value per retained member-month is
modelled separately by tier. The survivors' actual tier mix at the forecast
origin is used, and expected counts are propagated exactly (no simulation).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from features import acquired, _transactions

CELL = ['cohort_month', 'channel', 'primary_need']
T0 = pd.Period('2024-01', 'M').ordinal


def _logit_fit(x, n, y, ridge=1e-3):
    """Grouped binomial logistic regression: y successes of n trials."""
    b = np.zeros(x.shape[1])
    b[0] = np.log((y.sum() + .5) / (n.sum() - y.sum() + .5))
    for _ in range(200):
        p = 1 / (1 + np.exp(-(x @ b)))
        g = x.T @ (y - n * p) - ridge * b
        H = (x * (n * p * (1 - p))[:, None]).T @ x + ridge * np.eye(len(b))
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-10:
            break
    return b


def _sig(z):
    return 1 / (1 + np.exp(-z))


def panel(tables, as_of: str) -> pd.DataFrame:
    """One row per retained person-month: tier, age, cell, cash."""
    cutoff = pd.Timestamp(as_of)
    acq = acquired(tables, as_of)
    sp = tables['subscription_periods']
    sp = sp[(pd.to_datetime(sp.renewal_due_at) < cutoff) & (sp.renewal_outcome == 'paid')].copy()
    sp['month'] = pd.to_datetime(sp.period_start).dt.to_period('M')
    sp['premium'] = (sp.plan_tier == 'premium').astype(int)
    pm = sp.groupby(['person_id', 'month'], as_index=False).premium.max()
    pm = pm.merge(acq, on='person_id')
    pm = pm[pm.month >= pm.cohort]
    pm['age'] = (pm.month.map(lambda p: p.ordinal) - pm.cohort.map(lambda p: p.ordinal)).astype(int)
    tx = _transactions(tables, cutoff)
    tx['month'] = tx.occurred_at.dt.to_period('M')
    tx['net'] = tx.gross_amount - tx.discount_amount - tx.refund_amount
    tx['contribution'] = tx.net - tx.variable_cost
    fin = tx.groupby(['person_id', 'month'], as_index=False)[['net', 'contribution']].sum()
    pm = pm.merge(fin, on=['person_id', 'month'], how='left').fillna({'net': 0.0, 'contribution': 0.0})
    pm['t'] = pm.month.map(lambda p: p.ordinal - T0)
    return pm


def _dum(df):
    return np.column_stack([(df.primary_need == 'build_skill'), (df.channel == 'partner'),
                            (df.channel == 'organic')]).astype(float)


def _hx(age, prem, df):
    la = np.log(age)
    return np.column_stack([np.ones(len(la)), la, la ** 2, prem, prem * la, _dum(df)])


def _sx(age, df):
    la = np.log(age)
    return np.column_stack([np.ones(len(la)), la, _dum(df)])


def _vx(age, t, df, trend):
    cols = [np.ones(len(age)), np.log1p(age), _dum(df)]
    if trend:
        cols.append((t / 12)[:, None])
    return np.column_stack(cols)


def fit(tables, as_of: str, value_trend: bool = True, min_value_age: int = 1):
    pm = panel(tables, as_of)
    last_t = pd.Period(pd.Timestamp(as_of) - pd.DateOffset(months=1), 'M').ordinal - T0
    nxt = pm[['person_id', 't', 'premium']].rename(columns={'premium': 'next_premium'})
    nxt['t'] -= 1
    tr = pm[pm.t < last_t].merge(nxt, on=['person_id', 't'], how='left')
    tr['renewed'] = tr.next_premium.notna().astype(int)
    tr['age_next'] = tr.age + 1
    # cancellation hazard, grouped
    gk = ['age_next', 'premium', 'channel', 'primary_need']
    g = tr.groupby(gk, as_index=False).agg(n=('renewed', 'size'), r=('renewed', 'sum'))
    xh = _hx(g.age_next.to_numpy(float), g.premium.to_numpy(float), g)
    b_h = _logit_fit(xh, g.n.to_numpy(float), (g.n - g.r).to_numpy(float))
    # tier switching among renewers, separately by current tier
    b_s = {}
    rn = tr[tr.renewed == 1]
    for tier in (0, 1):
        s = rn[rn.premium == tier].groupby(['age_next', 'channel', 'primary_need'], as_index=False).agg(
            n=('next_premium', 'size'), up=('next_premium', 'sum'))
        b_s[tier] = _logit_fit(_sx(s.age_next.to_numpy(float), s), s.n.to_numpy(float), s.up.to_numpy(float))
    # value per retained member-month, by tier (cell-month aggregates)
    v = pm[pm.age >= min_value_age].groupby(['premium', 'age', 't', 'channel', 'primary_need'], as_index=False).agg(
        n=('person_id', 'size'), net=('net', 'sum'), contribution=('contribution', 'sum'))
    b_v = {}
    for tier in (0, 1):
        d = v[(v.premium == tier) & (v.net > 0) & (v.contribution > 0)]
        x = _vx(d.age.to_numpy(float), d.t.to_numpy(float), d, value_trend)
        w = d.n.to_numpy(float)
        y = np.log(d.net / d.n).to_numpy()
        m = (d.contribution / d.net).clip(1e-3, 1 - 1e-3).to_numpy()
        xw = x * w[:, None]
        A = x.T @ xw + 1e-6 * np.eye(x.shape[1])
        bn = np.linalg.solve(A, xw.T @ y)
        bm = np.linalg.solve(A, xw.T @ np.log(m / (1 - m)))
        smear = float(np.sum(w * np.exp(y - x @ bn)) / w.sum())
        b_v[tier] = (bn, bm, smear)
    origin_mix = (pm[pm.t == last_t].groupby(CELL[1:] + ['cohort'], as_index=False)
                  .agg(n=('premium', 'size'), prem=('premium', 'sum')))
    origin_mix['cohort_month'] = origin_mix.cohort.dt.to_timestamp().dt.strftime('%Y-%m-%d')
    return {'b_h': b_h, 'b_s': b_s, 'b_v': b_v, 'value_trend': value_trend,
            'origin_mix': origin_mix.set_index(CELL)[['n', 'prem']]}


def predict(model, grid: pd.DataFrame, trend_damp: float = 1.0):
    """Returns retained, net revenue, contribution arrays aligned to grid rows."""
    out = np.zeros((len(grid), 3))
    g = grid.sort_values(CELL + ['age_month'])
    pos = {ix: i for i, ix in enumerate(grid.index)}
    origin_t = pd.Period(g.calendar_month.min(), 'M').ordinal - T0 - 1
    for key, c in g.groupby(CELL, sort=False):
        n0 = float(c.origin_retained.iloc[0])
        mix = model['origin_mix'].loc[key] if key in model['origin_mix'].index else None
        p0 = float(mix.prem / mix.n) if mix is not None and mix.n > 0 else 0.5
        alive = np.array([n0 * (1 - p0), n0 * p0])
        one = c.iloc[[0]]
        for ix, row in c.iterrows():
            a = float(row.age_month)
            t = pd.Period(row.calendar_month, 'M').ordinal - T0
            new = np.zeros(2)
            for s in (0, 1):
                h = _sig(_hx(np.array([a]), np.array([float(s)]), one) @ model['b_h'])[0]
                up = _sig(_sx(np.array([a]), one) @ model['b_s'][s])[0]
                stay = alive[s] * (1 - h)
                new += stay * np.array([1 - up, up])
            alive = new
            te = origin_t + trend_damp * (t - origin_t)
            net = con = 0.0
            for s in (0, 1):
                bn, bm, smear = model['b_v'][s]
                x = _vx(np.array([a]), np.array([te]), one, model['value_trend'])
                npm = np.exp(x @ bn)[0] * smear
                net += alive[s] * npm
                con += alive[s] * npm * _sig(x @ bm)[0]
            out[pos[ix]] = (alive.sum(), net, con)
    return out[:, 0], out[:, 1], out[:, 2]
