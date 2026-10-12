-- Notebook 01, section 2: how full each reserved slot is relative to its two neighbours.
-- A slot with no impressions still gets a row (count 0), as in audit.slot_occupancy.
WITH counts AS (
    SELECT CASE WHEN CAST(random_bool AS BOOLEAN) THEN 'random' ELSE 'algorithm' END AS arm, position, COUNT(*) AS n
    FROM impressions
    GROUP BY ALL
),
grid AS (
    SELECT a.arm, g.position
    FROM (SELECT DISTINCT arm FROM counts) a,
         generate_series(1, (SELECT MAX(position) FROM impressions)) AS g(position)
),
filled AS (
    SELECT g.arm, g.position, COALESCE(c.n, 0) AS n
    FROM grid g LEFT JOIN counts c USING (arm, position)
),
ratios AS (
    SELECT arm, position, n,
           ROUND(n / ((LAG(n) OVER w + LEAD(n) OVER w) / 2.0), 4) AS vs_neighbours
    FROM filled
    WINDOW w AS (PARTITION BY arm ORDER BY position)
)
SELECT arm, position, n, vs_neighbours
FROM ratios
WHERE position IN (SELECT position FROM reserved_slots)
ORDER BY arm, position;
