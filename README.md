# Studio North holdout challenge: blended retention × trending member value

**Path B (FAQ 8) entry.** It forecasts January–December 2026 retained members, net revenue and contribution for the 24 existing cohorts (1,152 rows). It uses only records observed through 31 December 2025.

## Run

```text
uv sync --frozen
uv run --frozen python main.py --data-dir /path/to/released/data --prediction-grid prediction_grid.csv --output outputs/predictions.csv --seed 20261116
```

It takes about 2 minutes on 4 CPU cores and needs no network or GPU. It fits from scratch on every run. Two runs with the declared seed give byte-identical output (SHA-256 `9e965df3…5e40`). The output passes the instructor's `validate_predictions`.

## What changed from the released sBG baseline, and why

The baseline is the hierarchical shifted-Beta-Geometric (sBG) retention model plus a Gamma revenue model. I reproduced it exactly: $1,007,693 of 2026 contribution, identical to the lab's `baseline_predictions.csv`. Then I split its backtest errors into two parts: how many members are retained, and how much each retained member is worth.

1. **Value per member was the main miss.** Baseline contribution per retained member ran **5–12% low** at every backtest cutoff, and the gap widened with forecast horizon. The cause is in the data. For customers of the same tenure, net revenue and contribution margin per member have risen steadily with calendar time, at about 6% per half-year. Margin also rose from 0.41 to 0.47 over 2024–2025. The baseline's revenue model has tenure and cohort terms but no calendar-time term, and its net-to-gross ratios pool all history. So it holds 2026 value at past levels.
2. **Retention errors flip direction.** sBG over-forecast survivors from the Dec-2024 cutoff (+7%) and was nearly unbiased from Jun-2025.

## The model

1. **Retention** is the equal-weight average of two models:
   1. The released hierarchical sBG, vendored unchanged in `sbg_reference.py` (MIT, Longxiu Tian) and refit here with the seed.
   2. A **tenure-dependent logistic hazard** (`models.py`), fit by binomial maximum likelihood on the cancellations at each renewal: logit h = b0 + b1·log(age) + b2·log(age)² + need + channel. Duration dependence enters directly through age, in the spirit of Fader, Hardie et al. (2018), *"How to Project Customer Retention" Revisited: The Role of Duration Dependence*. In sBG, duration dependence comes only from unobserved heterogeneity.

   Each model is projected forward from the observed survivors at the origin (`origin_retained`). Expected membership never rises.
2. **Value per retained member** comes from two weighted models, with weights equal to retained members:
   1. log(net revenue per member) ~ log1p(age) + calendar time + need + channel, with a smearing correction.
   2. logit(contribution ÷ net revenue) on the same terms.

   Margin is bounded in (0, 1), so contribution can never exceed net revenue, by construction rather than by clipping.
3. Forecast = retained × value per member, per cell and month.

**Assumptions:**
1. Cancellation odds depend on tenure, need and channel, plus the sBG's cohort effects in that half of the average.
2. The price and margin trend of 2024–2025 continues for 12 more months.
3. No new cohorts.
4. Acquisition cost is excluded, because it predates the window.

**Why a 50/50 average with fixed weights:** the two retention models erred in opposite directions across cutoffs. Equal weights avoid estimating combination weights, and the forecast-combination literature finds that equal weights usually beat estimated ones.

## Chronological validation

At each cutoff the cohort table is rebuilt from raw records dated before that cutoff (`features.py`). Then the next months are forecast and scored with the instructor's `score_predictions`: cohort × month contribution WAPE. The baseline is refit at each cutoff.

| Forecast origin (data ends) | Months scored | Cohorts | Baseline WAPE | This model | Baseline bias | This model bias |
|---|---:|---:|---:|---:|---:|---:|
| Dec 2024 | 12 | 12 | 6.27% | **5.30%** | −1.9% | +2.4% |
| Jun 2025 | 6 | 18 | 7.11% | **4.14%** | −6.5% | −2.1% |
| Sep 2024 † | 12 | 9 | 8.83% | **6.52%** | −8.3% | −4.4% |
| Mar 2025 † | 9 | 15 | 7.25% | **4.99%** | −5.8% | −1.3% |

**How the model was selected:** the specification was chosen on the Dec-2024 and Jun-2025 cutoffs. The rule was fixed in advance: a change stays only if it beats the baseline at both cutoffs by at least 3% relative. † The Sep-2024 and Mar-2025 cutoffs were run afterwards as confirmation only and were not used for selection.

**Ablation** (contribution WAPE, Dec-2024 / Jun-2025):

| Retention | Value per member | Dec-24 | Jun-25 |
|---|---|---:|---:|
| sBG (baseline) | baseline | 6.27% | 7.11% |
| sBG | calendar trend | 8.20% | 3.90% |
| tenure GLM | calendar trend | 5.29% | 5.00% |
| **50/50 sBG + GLM** | **calendar trend** | **5.30%** | **4.14%** |
| 50/50 sBG + GLM | trend damped 50% | 5.19% | 5.07% |
| sBG + GLM with cohort trend | calendar trend | 7.31% | 3.93% |

Dampening the newest cohorts' optimism (H1) was tested through the cohort-trend terms. It did not win both cutoffs and was dropped.

## Extrapolation beyond observed ages (Jan-2024 cohort, ages 24–35)

No customer older than 23 months has been observed. The sBG hazard declines smoothly as α/(α+β+t−1), and the GLM hazard is quadratic in log(age). Over ages 24–35 both move slowly, so neither produces a cliff. Value per member grows with log1p(age), which flattens at high tenure. This cohort is about 1.4% of the scored contribution.

## Main risk

The 2026 forecast is **$1.067M of contribution, versus $1.008M for the baseline (+5.9%)**. Almost all of the difference comes from the price and margin trend continuing. If Studio North's price increases stop in 2026, this entry will over-forecast. The backtests support the trend at every cutoff, but they cannot show a regime change.

## Files

1. `main.py`: entry point and `predict()`.
2. `features.py`: rebuilds the lab's `cohort_age_summary` from raw tables as of any cutoff. It matches the lab's SQL output exactly.
3. `models.py`: the tenure hazard GLM and the value-per-member models.
4. `sbg_reference.py`: the released baseline, vendored unchanged.
5. `experiments/`: the backtest harness and variant runs. These need the lab repository on `PYTHONPATH` for the instructor's scorer.
