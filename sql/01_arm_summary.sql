-- Notebook 01, section 1: searches, impressions and engagement in each arm.
SELECT arm,
       COUNT(*)                                                   AS searches,
       SUM(n_results)                                             AS impressions,
       ROUND(COUNT(*) / SUM(COUNT(*)) OVER (), 4)                 AS share_of_searches,
       ROUND(AVG(CASE WHEN clicks > 0 THEN 1.0 ELSE 0.0 END), 4)  AS searches_with_click,
       ROUND(AVG(CASE WHEN bookings > 0 THEN 1.0 ELSE 0.0 END), 4) AS searches_with_booking,
       ROUND(AVG(clicks), 4)                                      AS clicks_per_search,
       MEDIAN(n_results)                                          AS median_results
FROM searches
GROUP BY arm
ORDER BY arm;
