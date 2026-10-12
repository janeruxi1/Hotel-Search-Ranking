-- Search-level tables for the audit (DuckDB dialect). Notebook 01 builds the per-search flags
-- that define the analysis sample for every later notebook from these definitions.
-- Expects a table or view called `impressions` with the columns of data/interim/train.parquet,
-- for example:  CREATE VIEW impressions AS SELECT * FROM read_parquet('data/interim/train.parquet');
-- tests/test_sql.py checks every rule against an independent pandas implementation.

CREATE OR REPLACE TEMP TABLE params AS
SELECT CAST(5000 AS DOUBLE) AS price_cap,       -- data.PRICE_CAP: prices outside (0, cap] are data errors
       CAST(0.95 AS DOUBLE) AS price_sort_rho;   -- data.PRICE_SORT_RHO: rank correlation that marks a price-sorted search

CREATE OR REPLACE TEMP TABLE reserved_slots AS
SELECT * FROM (VALUES (5), (11), (17), (23)) AS t(position);   -- data.RESERVED_SLOTS

-- One row per search.
CREATE OR REPLACE TEMP VIEW searches AS
SELECT srch_id,
       CASE WHEN BOOL_OR(CAST(random_bool AS BOOLEAN)) THEN 'random' ELSE 'algorithm' END AS arm,
       COUNT(*)                                AS n_results,
       MAX(position)                           AS max_position,
       SUM(CAST(click_bool AS INTEGER))        AS clicks,
       SUM(CAST(booking_bool AS INTEGER))      AS bookings
FROM impressions
GROUP BY srch_id;

-- Per-search data-quality flags (audit.price_flags):
--   bad_price     any price outside (0, cap]
--   price_rho     Pearson correlation of the average ranks of position and price, computed on the
--                 valid prices of searches with at least 5 of them; NULL when price is constant
--   price_sorted  price_rho >= the threshold (less 1e-9: small lists give exactly 0.95, and
--                 floating-point rounding must not decide them): the user sorted by price
CREATE OR REPLACE TEMP TABLE search_flags AS
WITH valid AS (
    SELECT i.srch_id, i.position, i.price_usd
    FROM impressions i, params p
    WHERE i.price_usd > 0 AND i.price_usd <= p.price_cap
),
ranked AS (
    SELECT srch_id,
           -- average rank for ties, as pandas' rank() does
           RANK() OVER (PARTITION BY srch_id ORDER BY position)
             + (COUNT(*) OVER (PARTITION BY srch_id, position) - 1) / 2.0 AS rank_position,
           RANK() OVER (PARTITION BY srch_id ORDER BY price_usd)
             + (COUNT(*) OVER (PARTITION BY srch_id, price_usd) - 1) / 2.0 AS rank_price
    FROM valid
),
rho AS (
    SELECT srch_id,
           CASE WHEN COUNT(*) >= 5 AND VAR_POP(rank_price) > 1e-12
                THEN CORR(rank_position, rank_price) END AS price_rho
    FROM ranked
    GROUP BY srch_id
),
bad AS (
    SELECT i.srch_id, BOOL_OR(i.price_usd <= 0 OR i.price_usd > p.price_cap) AS bad_price
    FROM impressions i, params p
    GROUP BY i.srch_id
)
SELECT s.srch_id, s.arm, s.arm = 'random' AS random,
       s.n_results AS n, s.max_position, s.clicks, s.bookings,
       b.bad_price, r.price_rho,
       COALESCE(r.price_rho >= p.price_sort_rho - 1e-9, FALSE) AS price_sorted,
       NOT b.bad_price AND NOT COALESCE(r.price_rho >= p.price_sort_rho - 1e-9, FALSE) AS keep
FROM searches s
JOIN bad b USING (srch_id)
LEFT JOIN rho r USING (srch_id)
CROSS JOIN params p;

-- The analysis sample (position.analysis_sample): clean searches, reserved slots dropped.
CREATE OR REPLACE TEMP VIEW clean_impressions AS
SELECT i.*, f.arm
FROM impressions i
JOIN search_flags f USING (srch_id)
WHERE f.keep AND i.position NOT IN (SELECT position FROM reserved_slots);
