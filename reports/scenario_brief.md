# Scenario brief

**Requester:** Director of Lodging Marketplace, an online travel agency (the data is Expedia's;
the scenario is framed the way an internal decision-science team would receive it).
**Analyst:** Xi Ru · **Status:** Phase 0, written before any modelling.

## The question as asked

> "Our ranking team wants to ship a new sort order that pushes cheaper hotels up. Before we
> spend an A/B test slot on it: how much does position actually matter, is price what drives
> bookings, and which ordering should we test?"

## Why the obvious answer is wrong

Top positions get far more clicks. But the current algorithm already puts the most attractive
hotels at the top, so part of the top slots' advantage belongs to the hotels, not the slots.
Reading position effects from normal traffic overstates them, and any ranking change sized
on that reading will overpromise.

The data contains a way out: in about 30% of searches the hotels were shown in random order.
In those searches, position is independent of hotel quality, so the drop in clicks by
position is the effect of position alone.

## What will be delivered

1. A position-effect curve from the randomized traffic, next to the naive curve, with the
   size of the bias stated in one number.
2. What drives choice within a search: relative price, promotion, stars, reviews, brand,
   location, with each effect expressed in dollars (for example, "one position up is worth
   a price cut of $X").
3. Expected bookings and revenue per search for four candidate rankings, with intervals,
   and the trade-off between them.
4. A recommendation under a decision rule fixed in advance, and the size of the A/B test
   that would confirm it.

## Out of scope

- Personalization by individual user history (95% missing).
- Competitor pricing (mostly missing).
- Claims about all searches: the data contains only searches with at least one click.

## What would change the recommendation

- If the randomized position effect is close to the naive one, ranking changes matter more
  than expected and the test should be prioritized.
- If price sensitivity is weak relative to reviews and location, a "cheapest first" sort is
  the wrong candidate.
- If the best policy wins on bookings but loses more than the agreed share of revenue, the
  recommendation is the blended ranking or no change.
