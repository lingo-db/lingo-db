WITH p AS
         (
             SELECT
                 pr_item_sk,
                 count(pr_item_sk) AS r_count,
                 AVG( CAST(pr_review_rating AS FLOAT8) ) avg_rating
             FROM product_reviews
             WHERE pr_item_sk IS NOT NULL
             GROUP BY pr_item_sk
         ), s AS
         (
             SELECT
                 ws_item_sk
             FROM web_sales ws
                      INNER JOIN date_dim d ON ws.ws_sold_date_sk = d.d_date_sk
             WHERE ws_item_sk IS NOT null
               AND CAST(d.d_date AS DATE) >= DATE '2003-01-02'
               AND CAST(d.d_date AS DATE) <= DATE '2003-02-02'
             GROUP BY ws_item_sk
         )
SELECT p.r_count    AS x,
       p.avg_rating AS y
FROM s INNER JOIN p ON p.pr_item_sk = s.ws_item_sk
;