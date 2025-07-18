SELECT *
FROM
    (
        SELECT
            cat,
            ( (count(x) * SUM(xy) - SUM(x) * SUM(y)) / (count(x) * SUM(xx) - SUM(x) * SUM(x)) )  AS slope,
            (SUM(y) - ((count(x) * SUM(xy) - SUM(x) * SUM(y)) / (count(x) * SUM(xx) - SUM(x)*SUM(x)) ) * SUM(x)) / count(x) AS intercept
        FROM
            (
                SELECT
                    i.i_category_id AS cat,
                    s.ss_sold_date_sk AS x,
                    CAST(SUM(s.ss_net_paid) AS FLOAT8) AS y,
                    CAST(s.ss_sold_date_sk * SUM(s.ss_net_paid) AS FLOAT8) AS xy,
                    CAST(s.ss_sold_date_sk * s.ss_sold_date_sk AS FLOAT8) AS xx
                FROM store_sales s
                         INNER JOIN item i ON s.ss_item_sk = i.i_item_sk
                         INNER JOIN date_dim d ON s.ss_sold_date_sk = d.d_date_sk
                WHERE s.ss_store_sk = 10
                  AND i.i_category_id IS NOT NULL
                  AND CAST(d.d_date AS DATE) >= DATE '2001-09-02'
                  AND   CAST(d.d_date AS DATE) <= DATE '2002-09-02'
                GROUP BY i.i_category_id, s.ss_sold_date_sk
            ) temp
        GROUP BY cat
    ) regression
WHERE slope <= 0.0
ORDER BY cat
;