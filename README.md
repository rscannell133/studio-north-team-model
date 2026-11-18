# Studio North holdout challenge: blended retention × tier-aware member value

**Path B (FAQ 8) entry.** It forecasts January–December 2026 retained members, net revenue and contribution for the 24 existing cohorts (1,152 rows). It uses only records observed through 31 December 2025.

## Run

```text
uv sync --frozen
uv run --frozen python main.py --data-dir /path/to/released/data --prediction-grid prediction_grid.csv --output outputs/predictions.csv --seed 20261116
```

It takes about 2 minutes on 4 CPU cores and needs no network or GPU. It fits from scratch on every run. Two runs with the declared seed give byte-identical output (SHA-256 `bf47b4d2…a359`). The output passes the instructor's `validate_predictions`.

## What the baseline misses

The baseline is the hierarchical shifted-Beta-Geometric (sBG) retention model plus a Gamma revenue model. I reproduced it exactly: $1,007,693 of 2026 contribution, identical to the lab's `baseline_predictions.csv`. Then I split its backtest errors into retained members and value per retained member.

1. **Value per member was the main miss.** Baseline contribution per member ran 5–12% low at every cutoff, and the gap widened with horizon.
2. **The cause is upgrades and add-ons, not prices.** List prices never changed ($50 standard, $75 premium). But:
   1. The premium share of retained members rose from 12% (Jan 2024) to 53% (Oct 2025).
   2. By tenure, premium share runs from 20% at acquisition to about 78% by month 18. About 25% of subscriptions switch tier.
   3. Premium members renew more often (95% vs 92%), so survivors tilt toward premium.
   4. Add-on revenue per member roughly doubled, and discounts faded.

   The baseline's revenue model has no tier state and no calendar term, so it holds 2026 value at historical levels.
3. **Retention errors flip direction.** sBG over-forecast survivors from the Dec-2024 cutoff (+7%) and was nearly unbiased from Jun-2025.

## The model

1. **Retained members** are the equal-weight average of two models:
   1. The released hierarchical sBG, vendored unchanged in `sbg_reference.py` (MIT, Longxiu Tian) and refit with the seed.
   2. A **tenure-dependent logistic hazard** (`models.py`): logit h = b0 + b1·log(age) + b2·log(age)² + need + channel, fit by binomial maximum likelihood on the cancellations at each renewal. Duration dependence enters directly, in the spirit of Fader, Hardie et al. (2018), *"How to Project Customer Retention" Revisited: The Role of Duration Dependence*.

   Both are projected forward from the observed survivors at the origin.
2. **Value per retained member** comes from a tier-aware member-state model (`tiers.py`):
   1. **State:** each retained member is standard or premium. The model starts from each cell's actual tier mix in December 2025.
   2. **Transitions** are logistic in tenure, channel and need:
      1. Cancellation hazard, by tier, with a tier × log(age) interaction.
      2. Upgrade or downgrade probability for renewers, separately by current tier.

      Expected counts are propagated exactly, with no simulation, so the tier mix evolves and levels off as cohorts age.
   3. **Value by tier:** log(net revenue per member-month) and logit(contribution ÷ net) are weighted linear models in log1p(age), calendar time, need and channel. The calendar term captures add-on growth and fading discounts. Margin is bounded in (0, 1), so contribution never exceeds net revenue, by construction.
3. Forecast = retained (part 1) × the tier model's expected value per retained member (part 2).

**Assumptions:**
1. Cancellation depends on tenure, tier, need and channel.
2. Tier switching follows its 2024–2025 tenure pattern.
3. Add-on and discount trends continue for 12 months.
4. No new cohorts.
5. Acquisition cost is excluded, because it predates the window.

## Chronological validation

At each cutoff the cohort table and the member panel are rebuilt from raw records dated before that cutoff (`features.py`, `tiers.py`). Then the next months are forecast and scored with the instructor's `score_predictions`: cohort × month contribution WAPE. All models are refit at each cutoff.

| Forecast origin (data ends) | Months | Cohorts | Baseline | Version 1 (pooled value trend) | **Final (tier-aware value)** |
|---|---:|---:|---:|---:|---:|
| Dec 2024 | 12 | 12 | 6.27% | 5.30% | **4.91%** |
| Jun 2025 | 6 | 18 | 7.11% | 4.14% | **3.85%** |
| Sep 2024 † | 12 | 9 | 8.83% | 6.52% | 6.83% |
| Mar 2025 † | 9 | 15 | 7.25% | 4.99% | **4.66%** |

**How the model was selected:** components were chosen on the Dec-2024 and Jun-2025 cutoffs only. The rule was fixed in advance: a change stays only if it beats the incumbent at both cutoffs by at least 3% relative. † The Sep-2024 and Mar-2025 cutoffs are confirmation only. The final model beats the baseline at all four. Against Version 1 it is better at three of four. It is slightly worse at Sep 2024, the cutoff with the least history (9 months) for estimating tier transitions.

**Ablation** (contribution WAPE, Dec-2024 / Jun-2025):

| Retained members | Value per member | Dec-24 | Jun-25 |
|---|---|---:|---:|
| sBG (baseline) | baseline | 6.27% | 7.11% |
| sBG | tier-aware | 6.72% | 3.36% |
| 50/50 sBG + GLM | pooled calendar trend | 5.30% | 4.14% |
| **50/50 sBG + GLM** | **tier-aware** | **4.91%** | **3.85%** |
| 50/50 sBG + GLM | tier-aware, trend damped 50% | 4.99% | 4.33% |
| sBG | tier-aware, no calendar trend | 5.24% | 5.15% |
| tier model's own retention | tier-aware | 5.13% | 4.95% |
| 1/3 each sBG, GLM, tier | tier-aware | 4.99% | 4.13% |

Two changes that were tested and dropped:
1. **Dampening the newest cohorts' optimism.** Done through cohort-trend terms, it did not win both cutoffs.
2. **Engagement signals** (product events, support contacts). They moved renewal rates by only about 1 point, so they were not added.

## Extrapolation beyond observed ages (Jan-2024 cohort, ages 24–35)

No customer older than 23 months has been observed. The sBG hazard declines smoothly as α/(α+β+t−1), and the GLM and tier hazards are smooth in log(age), so none produces a cliff. Upgrade probabilities are logistic in log(age), so premium share approaches a plateau instead of growing without bound. This cohort is about 1.4% of scored contribution.

## Forecast and main risk

2026 contribution is **$1.049M, versus $1.008M for the baseline (+4.1%)**:

| Cohorts | 2026 contribution |
|---|---:|
| 2024 | $0.34M |
| Jan–Jun 2025 | $0.29M |
| Jul–Dec 2025 | $0.42M |

The main risk is that upgrade or add-on behaviour changes in 2026, for example through a pricing or packaging change. The model assumes the 2024–2025 patterns continue. The backtests support that at every cutoff but cannot reveal a regime change.

## Files

1. `main.py`: entry point and `predict()`.
2. `features.py`: rebuilds the lab's `cohort_age_summary` from raw tables as of any cutoff. It matches the lab's SQL output exactly.
3. `models.py`: the tenure hazard GLM and the pooled value model used in Version 1.
4. `tiers.py`: the tier-aware member-state model.
5. `sbg_reference.py`: the released baseline, vendored unchanged.
6. `experiments/`: the backtest harness and variant runs. These need the lab repository on `PYTHONPATH` for the instructor's scorer.
