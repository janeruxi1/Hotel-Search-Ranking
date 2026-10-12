# Metric framework

Unit of analysis: the **search** (`srch_id`), which is also the unit of randomization.
Intervals are bootstrapped over searches, except for choice-model coefficients and the dollar
values derived from them, which use the delta method on the estimated covariance.

## Primary metric

| Metric | Definition | Why |
|---|---|---|
| Bookings per search | Share of searches ending in a booking (at most one booking per search) | The marketplace's conversion outcome; the policy comparison is made on it |

## Secondary metrics

| Metric | Definition | Use |
|---|---|---|
| Click-through by position | Clicks / impressions at each position | Position-effect curves |
| Booking rate by position | Bookings / impressions at each position | Position effect on the outcome that pays |
| Expected booking value per search | Σ P(book hotel i) × nightly price of hotel i × nights | Revenue side of the trade-off (gross booking value, not margin; margin is not in the data) |
| Position effect ratio | Rate at position 1 / rate at position k, naive vs randomized | One-number summary of the bias |

## Guardrails for a ranking change

| Guardrail | Threshold (fixed before phase 4, in `reports/decision_rule.md`) |
|---|---|
| Expected booking value per search | Not more than 2% below the baseline ranking |
| Share of bookings at hotels with no reviews | Not above the baseline share (avoid promoting unknown quality) |
| Price of booked hotels | Report the change; a large drop signals revenue risk |

## Definitions and rules

- **Arms:** `random_bool = 1` (random order) and `0` (Expedia's algorithm). Booking rates are
  never compared between arms, because the arms were sampled differently.
- **Relative price:** log(price) minus the mean log price of the hotels in the same search.
- **Price trimming:** searches with any price ≤ 0 or above $5,000 (the cap fixed in the audit)
  are excluded, as are searches the user sorted by price; the counts are reported in notebook 01.
- **Positions:** slots 1–40 as logged; slots 5, 11, 17 and 23 are nearly empty (reserved slots) and
  are left out of position curves and choice sets.
- **Population:** searches with at least one click (the only searches in the data).

## Where each guardrail is reported

All three are evaluated in notebook 04 for every candidate ranking: booking value in the
decision rule (section 4), the no-review booking share and the booked hotel's nightly price in
section 5. In the A/B test (notebook 05) booking value per search is the formal guardrail.
