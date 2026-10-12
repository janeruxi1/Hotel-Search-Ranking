-- Notebook 01, section 5: which searches enter the analysis, and why the others do not.
SELECT
    SUM(CASE WHEN bad_price THEN 1 ELSE 0 END)                         AS searches_bad_price,
    SUM(CASE WHEN price_sorted AND NOT bad_price THEN 1 ELSE 0 END)    AS searches_price_sorted,
    SUM(CASE WHEN keep AND arm = 'algorithm' THEN 1 ELSE 0 END)        AS kept_algorithm,
    SUM(CASE WHEN keep AND arm = 'random' THEN 1 ELSE 0 END)           AS kept_random,
    (SELECT COUNT(*) FROM impressions JOIN search_flags USING (srch_id) WHERE keep) AS impressions_kept
FROM search_flags;
