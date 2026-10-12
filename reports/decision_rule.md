# Pre-declared decision rule for phase 4 (ranking policies)

Status: **committed before any phase 4 evaluation code was written or run.** The git history
shows this file landing before `notebooks/04_policy_evaluation`. Changing it after seeing results
would defeat its purpose; any later change must be a new, dated section below, not an edit.

## What will be compared

Four ways to order the hotels in a search:

| Policy | Rule |
|---|---|
| B1 · Reviews first | Sort by review score, highest first (no-review hotels last) |
| B2 · Cheapest first | Sort by price, lowest first — the change the ranking team proposed |
| M1 · Most likely to be booked | Sort by the model's booking utility (attributes only, no position term) |
| M2 · Highest expected value | Sort by booking probability × price |

B1 and B2 are simple baselines; M1 and M2 come from the choice model.

## How they will be evaluated

- **Data.** Held-out random-order searches (search id divisible by 5, the same split as notebook
  03). Models are re-fitted on the other 80% only.
- **Model.** A booking model with an outside option ("clicked but did not book"), fitted on
  random-order searches. Because positions were assigned at random, searches where attractive
  hotels happened to land near the top show whether a better top of the page raises the chance
  that any booking happens, not only which hotel is booked.
- **Metrics per search.** Expected bookings (primary) and expected booking value = Σ P(book i) ×
  nightly price × nights (secondary).
- **Uncertainty.** 95% intervals from a bootstrap over held-out searches combined with draws
  of the model coefficients from their estimated covariance.

## The rule

Let **B\*** be whichever of B1 and B2 has more expected bookings per search, and **M\*** whichever
of M1 and M2 has more.

**Recommend an A/B test of M\*** if both hold:
1. M\*'s expected bookings per search exceed B\*'s, with the lower end of the 95% interval of the
   difference above zero; and
2. M\*'s expected booking value per search is no more than 2% below B\*'s (point estimate).

Otherwise **recommend B\*** and state which condition failed.

Separately, report whether **B2 (cheapest first)** beats B1 on expected bookings, because that is
the question the ranking team asked.

## What this rule does not cover

- Expedia's own ranking cannot be evaluated: the data does not say what order Expedia would have
  shown in the random-order searches. The comparison is between candidate rankings only.
- Offline estimates rely on the choice model. The recommendation is to *test*, not to ship; the
  A/B test sized in phase 5 is what decides.

## Changes

- 2026-10-11: phase numbers relabelled to match notebook numbers (the ranking-policy phase was
  "Phase 5" in the first commit of this file and is now phase 4; the memo phase moved from 6 to
  5). Wording only; the rule itself is unchanged.
