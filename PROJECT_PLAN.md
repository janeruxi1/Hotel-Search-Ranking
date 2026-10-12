# Project Plan & Design Decisions

Working document for project 04. It records the decisions made before building, so the
rationale survives past the conversation that produced it. Numbers quoted here come from
the data profile run on 2026-10-09; every number that reaches the README will be produced
by committed code and checked by CI, as in projects 01–03.

---

## What this project is

**How much does a hotel's position in search results cause bookings, what else drives
the choice within a search (price, promotions, reviews), and would a different ranking
earn more — measured on Expedia's own randomized traffic before anyone runs an A/B test?**

Data: Expedia's hotel-search logs released for the ICDM 2013 Kaggle competition
(`expedia-personalized-sort`): 9,917,530 hotel impressions from 399,344 searches,
November 2012 – June 2013. For **30.4% of searches Expedia showed the results in random
order**. That randomized slice is the project's lever: a real experiment inside real data.

## Positioning — causal inference and decision science first

Headline framing: *"Expedia's ranking makes position 1 look 17× better than position 30 for
clicks. Its own randomized traffic says about 5× (4.4× after adjusting for a residual tilt).
Here is what that bias costs, what actually drives bookings, and which ranking to test next."*
(Numbers from `reports/metrics/02_position_effects.json`, on searches that fill every slot
from 1 to 30. The profile's first pooled figures, 22× and 7×, mixed in a composition effect:
bottom positions exist only in long lists, where each hotel's click share is lower.)

- Leads with the identification problem (better hotels are ranked higher, so naive
  position effects are confounded), not with learning-to-rank leaderboard scores.
- Hero figure: click and booking rate by position, algorithm-ranked vs randomized.
- The decision is a ranking policy chosen by expected bookings and revenue, with
  uncertainty, plus the A/B test that would confirm it.
- Skills line: causal inference with a randomized benchmark · discrete-choice modelling ·
  off-policy evaluation · experiment design · working with 10M-row real data.

Audience: decision-science, product and marketplace data-science roles (Seattle employers
such as Expedia and Amazon, and remote roles).

## Portfolio arc this project completes

| Project | Data | Causal setting | What it demonstrates |
|---|---|---|---|
| 01 — A/B test | Synthetic | Randomized | Inference when you can experiment |
| 02 — Churn & retention | Synthetic | Predictive + policy | Turning predictions into budgeted decisions |
| 03 — RAG evaluation | Synthetic corpus | Evaluation | Measuring an AI system before trusting it |
| **04 — Hotel search (this)** | **Real, 10M rows** | **Observational, checked against a randomized slice** | **Separating selection from effect, then choosing a policy** |

## What the data allows — and what it does not

Facts from the 2026-10-09 profile, and the design consequence of each:

| Fact | Consequence |
|---|---|
| 30.4% of searches randomly ordered | Position effects are identified **within the random arm**. The algorithm arm gives the naive comparison. |
| Every search has at least one click; at most one booking | Results describe searches with engagement, not all searches. Stated in every headline. |
| 93.9% of algorithm-arm searches end in a booking vs 13.0% of random-arm searches | The two arms were **sampled differently**. Never compare booking rates *between* arms as an effect of randomization; compare *within* each arm. |
| Position 5 nearly empty (about 6% of the rows at positions 1–4) | A reserved slot; audit it and treat positions as slots actually shown. |
| Price: median $122, a few zeros, a maximum above $19 million; rises only mildly with length of stay | Mostly nightly prices. Use **price relative to the other hotels in the same search**, which removes units and currency issues; trim extreme values with a stated rule. |
| Price was **not** randomized | Price and promotion effects are observational within-search associations, adjusted for hotel quality. Never called causal. |
| Competitor-price fields 55–98% missing | Not used for headline results; one exploratory check at most. |

## Questions, in order

1. **Audit.** Does the data match the published description, and what are its traps?
2. **Position effect.** How steeply do clicks and bookings fall with position when position
   is random, and how much does the naive (algorithm-arm) curve overstate it?
3. **What drives choice within a search.** Conditional logit on the random arm (where
   position is independent of hotel attributes): price relative to the search, promotion,
   stars, review score, brand, location score. Express effects in dollars: what one position
   is worth as a price cut, what a promotion flag is worth.
4. **Which ranking to test.** Rank hotels by predicted booking probability, by expected
   revenue, by price, and by a blend. Estimate expected bookings and revenue per search for
   each policy **offline**, using position effects from Q2 and choice probabilities from Q3,
   evaluated on held-out random-arm searches. Bookings and revenue trade off; show the frontier.
5. **Pilot.** Size an A/B test that would confirm the chosen ranking.

## Pre-declared decision rule

Fixed in [`reports/decision_rule.md`](reports/decision_rule.md), committed with phase 3 and
before any phase 4 code: four candidate rankings (reviews first, cheapest first, two model
rankings); a model ranking is recommended for an A/B test only if its expected bookings per
search beat the better baseline with the 95% interval of the difference above zero, and its
expected booking value is no more than 2% lower.

## Staging rule

**Every phase ends with a working, committed artifact.** No README number is written
until the code producing it has run and saved its output to `reports/metrics/*.json`.

## Phases

Phase numbers match notebook numbers: notebook `0N_*` is phase N.

| # | Phase | Status | Artifact that must exist to call it done |
|---|---|---|---|
| 0 | Plan and data preparation | done | `reports/scenario_brief.md`, `reports/metrics_framework.md`, this file; `scripts/prepare_data.py` → `data/interim/train.parquet` with validated row, search and hotel counts; synthetic CI fixture with the same schema |
| 1 | Audit | done | `notebooks/01_audit` → `reports/metrics/01_audit.json`: arms, selection, reserved slots, price rules, randomization check |
| 2 | Position effects | done | `notebooks/02_position_effects` → naive vs randomized curves with search-clustered intervals; overstatement factor; comparison with Compiani et al. |
| 3 | Choice model | done | `notebooks/03_choice_model` → conditional logit on the random arm; dollar value of a position and of a promotion; hotel-price control; hold-out validation |
| 4 | Ranking policies | done | decision rule committed first; `notebooks/04_policy_evaluation` → expected bookings and booking value per search per policy with intervals |
| 5 | Decision memo & A/B test | done | `reports/decision_memo.md`, A/B sizing; memo numbers checked by CI against the records |
| 6 | Packaging | done | README, tests, CI on the synthetic fixture, Streamlit demo on derived results only |

## Methods notes

- **Uncertainty:** bootstrap over searches (the unit of randomization), never over rows.
- **Position model (as planned):** P(click) = examination(position) × attractiveness(hotel, search).
  **As built:** a conditional logit on the random arm with one dummy per displayed slot plus hotel
  attributes, which estimates both in one model (notebook 03); notebook 02 measures the
  position curves directly with a balanced panel and a within-search linear probability model.
- **Validation of the offline evaluator:** apply it to the random arm's *logged* order and check that the
  predicted booking rate matches the observed one (done: calibration by decile), test the mechanism
  directly (order-luck slope), and check that it does not flatter rankings built from the same model
  (cross-fitting on two halves of the training searches).
- **Benchmark:** Compiani, Lewis, Peng & Wang, "Trade-Offs Between Ranking Objectives" (arXiv 2210.16408)
  estimate position effects on the same randomized sessions; report how the estimates compare.

## Data terms and reproducibility

- The data are Kaggle competition data used under the competition rules. They are **never
  committed**; `.gitignore` excludes `data/raw/*`, `data/interim/`, `data/processed/`.
- `data/raw/README.md` explains how to obtain them. `data.zip` uses Deflate64 compression,
  which Python's `zipfile` cannot read: extract it with Windows Explorer, 7-Zip or `unzip`.
- CI runs every notebook on a small synthetic fixture with the same schema; the real-data
  results are recorded in `reports/metrics/` and checked against the memo.
- Any hosted demo shows derived, aggregated results only.

## Compute

About 10M rows × 54 columns. `prepare_data.py` keeps the columns used and downcasts
types so the full table fits in memory on a laptop; the choice model uses the random arm
(about 2.9M rows).
