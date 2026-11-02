# Notebooks: reading guide

Read them in order. Each notebook opens with a short table (question, answer, next) and ends
with a **Verdict**; every figure has a "how to read it" note under it.

| # | Notebook | The question in plain words | Main figure |
|---|---|---|---|
| 01 | [Data audit](01_audit.ipynb) | Does the data match its description, and where are the traps? | Reserved slots; star-rating tilt by arm |
| 02 | [Position effects](02_position_effects.ipynb) | How much does position itself cause clicks and bookings? | Naive vs randomized position curves |
| 03 | [Choice model](03_choice_model.ipynb) | What drives the choice, in dollars of price cut? | Position worth in dollars; calibration |
| 04 | [Policy evaluation](04_policy_evaluation.ipynb) | Which ranking should go into the A/B test? | Bookings and value by ranking |
| 05 | [A/B test design](05_ab_test_design.ipynb) | How big does the test need to be? | Power curves |
| 06 | [Learning to rank](06_learning_to_rank.ipynb) (exploratory) | Would a LambdaMART ranker change the answer? | NDCG by ranker; feature importance |

**Running them.** Each `.ipynb` is generated from the `.py` file beside it (the `.py` is the
source); see the main [README](../README.md) for the data and the build command.
