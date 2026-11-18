"""Rebuild the lab's cohort_age_summary from raw tables, as of any cutoff.

Mirrors sql/01_views.sql (first_paid -> acquired_customers -> customer_month ->
cohort_age_summary). `as_of` is exclusive: only records before it are used, so
the same function serves the final forecast (as_of=2026-01-01) and the
chronological backtests (as_of=2025-01-01, 2025-07-01).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CELL = ['cohort_month', 'channel', 'primary_need']


def _transactions(tables, cutoff):
    tx = tables['transactions'].copy()
    tx['occurred_at'] = pd.to_datetime(tx.occurred_at)
    tx = tx[tx.occurred_at < cutoff]
    for c in ['gross_amount', 'discount_amount', 'refund_amount', 'variable_cost']:
        tx[c] = tx[c].astype(float).fillna(0.0)
    return tx


def acquired(tables: dict[str, pd.DataFrame], as_of: str) -> pd.DataFrame:
    """person_id, cohort (Period), channel, primary_need for customers acquired before as_of."""
    tx = _transactions(tables, pd.Timestamp(as_of))
    pay = tx[(tx.transaction_type == 'payment') & (tx.gross_amount - tx.discount_amount > 0)]
    first = pay.sort_values(['person_id', 'occurred_at', 'transaction_id']).drop_duplicates('person_id')
    first = first[['person_id', 'occurred_at']]

    campaigns = (tables['campaign_spend'].groupby('campaign_id', as_index=False)
                 .agg(channel=('channel', 'min')))
    attr = tables['acquisition_attribution'][['person_id', 'credited_campaign_id', 'source_type']]
    acq = (first.merge(tables['people'][['person_id', 'primary_need']], on='person_id')
           .merge(attr, on='person_id')
           .merge(campaigns, left_on='credited_campaign_id', right_on='campaign_id', how='left'))
    acq['channel'] = acq.channel.fillna(acq.source_type)
    acq['cohort'] = acq.occurred_at.dt.to_period('M')
    return acq[['person_id', 'cohort', 'channel', 'primary_need']]


def cohort_age_summary(tables: dict[str, pd.DataFrame], as_of: str) -> pd.DataFrame:
    cutoff = pd.Timestamp(as_of)
    tx = _transactions(tables, cutoff)
    acq = acquired(tables, as_of)

    last_month = (cutoff - pd.DateOffset(months=1)).to_period('M')
    span = (last_month.ordinal - acq.cohort.map(lambda p: p.ordinal)).to_numpy() + 1
    cm = acq.loc[acq.index.repeat(span), ['person_id', 'cohort', 'channel', 'primary_need']].copy()
    cm['age_month'] = cm.groupby('person_id').cumcount()
    cm['month'] = [c + a for c, a in zip(cm.cohort, cm.age_month)]

    sp = tables['subscription_periods'].copy()
    sp = sp[pd.to_datetime(sp.renewal_due_at) < cutoff]
    sp['month'] = pd.to_datetime(sp.period_start).dt.to_period('M')
    sp['retained'] = (sp.renewal_outcome == 'paid').astype(int)
    ret = sp.groupby(['person_id', 'month'], as_index=False).retained.max()

    tx['month'] = tx.occurred_at.dt.to_period('M')
    tx['net_revenue'] = tx.gross_amount - tx.discount_amount - tx.refund_amount
    tx['contribution'] = tx.net_revenue - tx.variable_cost
    tx['gross_payment_revenue'] = np.where(tx.transaction_type == 'payment',
                                           tx.gross_amount - tx.discount_amount, 0.0)
    fin = tx.groupby(['person_id', 'month'], as_index=False)[
        ['gross_payment_revenue', 'net_revenue', 'variable_cost', 'contribution']].sum()

    cm = cm.merge(ret, on=['person_id', 'month'], how='left').merge(fin, on=['person_id', 'month'], how='left')
    cm = cm.fillna({'retained': 0, 'gross_payment_revenue': 0.0, 'net_revenue': 0.0, 'variable_cost': 0.0, 'contribution': 0.0})
    out = (cm.groupby(['cohort', 'channel', 'primary_need', 'age_month', 'month'], as_index=False)
           .agg(n_acquired=('person_id', 'size'), n_retained=('retained', 'sum'),
                gross_payment_revenue=('gross_payment_revenue', 'sum'),
                net_revenue=('net_revenue', 'sum'), variable_cost=('variable_cost', 'sum'),
                contribution=('contribution', 'sum')))
    out['cohort_month'] = out.cohort.dt.to_timestamp().dt.strftime('%Y-%m-%d')
    out['calendar_month'] = out.month.map(lambda p: p.to_timestamp().strftime('%Y-%m-%d'))
    out['n_retained'] = out.n_retained.astype(int)
    out['cohort_index'] = out.cohort.map(lambda p: p.ordinal - pd.Period('2024-01', 'M').ordinal)
    for c in ['gross_payment_revenue', 'net_revenue', 'variable_cost', 'contribution']:
        out[c] = out[c].round(2)
    cols = CELL + ['cohort_index', 'age_month', 'calendar_month', 'n_acquired', 'n_retained',
                   'gross_payment_revenue', 'net_revenue', 'variable_cost', 'contribution']
    return out[cols].sort_values(CELL + ['age_month']).reset_index(drop=True)


def grid_from_summary(summary: pd.DataFrame, months: int = 12) -> pd.DataFrame:
    """Same construction as cohort_lab.challenge.prediction_grid."""
    origin = pd.to_datetime(summary.calendar_month).max()
    rows = []
    for key, g in summary.groupby(CELL, sort=True):
        last = g.sort_values('age_month').iloc[-1]
        for step in range(1, months + 1):
            rows.append(dict(zip(CELL, key)) | {
                'calendar_month': (origin + pd.DateOffset(months=step)).strftime('%Y-%m-%d'),
                'age_month': int(last.age_month) + step, 'n_acquired': int(last.n_acquired),
                'origin_retained': int(last.n_retained)})
    return pd.DataFrame(rows)
