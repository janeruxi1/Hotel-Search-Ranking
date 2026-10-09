# Metric framework

Unit of analysis: the **search** (`srch_id`), which is also the unit of randomization.
All intervals are bootstrapped over searches.

## Primary metric

| Metric | Definition | Why |
|---|---|---|
| Bookings per search | Share of searches ending in a booking (at most one booking per search) | The marketplace's conversion outcome; the policy comparison is made on it |

## Secondary metrics

| Metric | Definition | Use |
|---|---|---|
| Click-through by position | Clicks / impressions at each position | Position-effect curves |
| Booking rate by position | Bookings / impressions at each position | Position effect on the outcome that pays |
| Expected revenue per search | Σ P(book hotel i) × price of hotel i | Revenue side of the trade-off (gross booking value, not margin; margin is not in the data) |
| Position effect ratio | Rate at position 1 / rate at position k, naive vs randomized | One-number summary of the bias |

## Guardrails for a ranking change

| Guardrail | Threshold (fixed before Phase 5) |
|---|---|
| Expected revenue per search | Not more than 2% below the baseline ranking |
| Share of bookings at hotels with no reviews | Not above the baseline share (avoid promoting unknown quality) |
| Price of booked hotels | Report the change; a large drop signals revenue risk |

## Definitions and rules

- **Arms:** `random_bool = 1` (random order) and `0` (Expedia's algorithm). Booking rates are
  never compared between arms, because the arms were sampled differently.
- **Relative price:** log(price) minus the mean log price of the hotels in the same search.
- **Price trimming:** impressions with price ≤ 0 or above a cap fixed in the audit are excluded,
  and the share excluded is reported.
- **Positions:** slots 1–40 as logged; slot 5 is audited separately because it is nearly empty.
- **Population:** searches with at least one click (the only searches in the data).
