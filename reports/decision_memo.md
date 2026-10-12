# Decision memo: which hotel ranking to A/B test

**To:** Director of Lodging Marketplace · **From:** Xi Ru, Decision Science · **Date:** 2026-10-11
**Re:** "Should we spend an A/B test slot on putting cheaper hotels first?"

## Bottom line

1. **Don't test cheapest first.** Offline it loses to a plain reviews-first sort on both
   bookings (13.69 against 13.84 per 100 searches) and booking value ($35.44 against $46.79
   per search, -24%).
2. **Test two candidates against the current ranking:** reviews first (the pick of the rule we
   agreed before the analysis) and a blend of the two model rankings (better than reviews first on
   both metrics offline, but found after looking, so it has to earn its place in the test).
3. **The test needs about 1.85 million clicked searches** (616,042 per arm), and that size is set
   by the 2% booking-value guardrail, not by the booking lift. Relaxing the guardrail to 5% would
   cut it to about 114,000 per arm. That trade-off is your call; see "Decisions needed".

## What the analysis found

**Position matters, but far less than the dashboards suggest.** In Expedia's ranked traffic,
position 1 gets 16.8× the clicks of position 30. In the 30% of searches shown in random order it
gets 4.9×. Reading position effects from ranked traffic overstates them about 3.4-fold
(95% interval 3.2–3.7), because the ranking already puts the best hotels on top. Any business
case sized on the naive curve will overpromise.

**Price matters, and so does quality.** Comparing a hotel with itself at different prices, a 10%
higher price cuts clicks by about 15%. Moving a hotel from position 10 to position 1 is worth the
same as a $56 price cut for clicks and $40 for bookings. Stars, reviews and promotions are worth
$10–21 each. Price is one lever among several, which is why "cheapest first" underperforms.

**Offline comparison of rankings** (23,165 held-out random-order searches):

| Ranking | Bookings per 100 searches | Booking value per search |
|---|---|---|
| Reviews first | 13.84 | $46.79 |
| Cheapest first (the proposal) | 13.69 | $35.44 |
| Model: most likely to be booked | 15.26 | $44.41 |
| Model: highest expected value | 14.23 | $50.41 |
| Model blend (exploratory) | 15.14 | $47.04 |

The rule we fixed in advance tests the strongest model ranking only if it does not give up more
than 2% of booking value. "Most likely to be booked" gains +10.3% bookings but its booking value changes
by -5.1%, beyond the 2% allowed, so the rule's answer is reviews first.

**Two further checks hold up.** The other guardrails in our metric framework are met by the
blend: 1.23% of its bookings go to hotels with no reviews (1.31% under reviews first; cheapest
first more than doubles it, to 2.81%), and its booked hotel costs $138 a night against $151, so
its value gain comes from more bookings, not dearer ones. And the evaluator does not flatter
rankings built from its own model: scored by a model fitted on different searches, the strongest
model ranking gives 15.25 against 15.26 bookings per 100 searches when it scores itself.

**The main uncertainty.** All gains above assume that a better top of the page makes a booking
more likely at all, not just shifts which hotel gets booked. The model is well calibrated
overall (13.33% predicted against 13.18% observed), but the direct check of that mechanism is
inconclusive: realized effects ran at about 0.37 of the model's prediction, with a 95% interval
from -0.28 to 1.02. Offline analysis can choose what to test; only the test can say what it is
worth.

## Recommended test

| | |
|---|---|
| Arms | C: current ranking · T1: reviews first · T2: model blend (exploratory), equal thirds |
| Unit | Visitor (searches analysed, with visitor-clustered standard errors) |
| Primary metric | Booking per search |
| Guardrail | Booking value per search (capped at the 99th percentile) no more than 2% below control |
| Comparisons | T1 vs C, T2 vs C, T2 vs T1, sized at α = 0.0167 each (Bonferroni; Holm in the analysis), power 80% |
| Size | 616,042 clicked searches per arm, 1,848,126 in total; minimum two full weeks |
| Detectable lift | 1.5% relative on a 13.3% booking rate |

The booking comparison alone would need 114,487 per arm: it is sized on the gap between the two
candidates (+9.4% offline), discounted to +3.5% because of the inconclusive mechanism check.

**What we do with the result** (fixed now, before any data):

- Ship T2 if it beats C on bookings and passes the guardrail. Ship T1 under the same conditions
  if T2 does not. If both pass, ship whichever wins the T2 vs T1 comparison, and T1 if that
  comparison is inconclusive, since it is the simpler ranking.
- Keep the current ranking if neither candidate beats it.
- Stop early only for a guardrail breach or a sample-ratio mismatch (checked daily); no early
  stopping for a win.

## Decisions needed

1. **Guardrail margin.** At 2% the test needs about 1.85 million clicked searches; at 5% it needs
   about 0.3 million, and the booking comparison sets the size instead.
2. **Traffic share.** At 100,000 clicked searches a day entering the test, the 2% version takes
   19 days.

## Limitations

- The data contains only searches with at least one click. In production the test must be
  analysed on every search that shows results, since a ranking changes who clicks; the required
  number of searches is then larger by roughly one over the click-through rate.
- Searches are treated as independent in the sizing. A one-week A/A period will estimate the
  design effect from repeat visitors before the test starts.
- Variance reduction (CUPED on each visitor's pre-test bookings) would cut the guardrail
  requirement, possibly by a lot; the data has no visitor history to estimate how much, so the
  sizes above do not assume it.
- Booking value is nightly price × nights; a minority of sites appear to report prices per stay,
  which adds noise to value but not to bookings.
- Expedia's current ranking cannot be scored offline, so how much either candidate beats it is
  unknown until the test runs.

*Analysis: notebooks 01–05 in this repository; the key numbers above are checked against
`reports/metrics/` in CI.*
