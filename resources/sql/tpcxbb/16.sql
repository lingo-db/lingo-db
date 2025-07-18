WITH date_keys AS (
    SELECT CAST(d_date AS DATE) AS d_date, d_date_sk
    FROM date_dim
    WHERE CAST(d_date AS DATE) IN (
                                   DATE '2001-02-14',
                                   DATE '2001-03-16',
                                   DATE '2001-04-15'
        )
),
date_bounds AS (
 SELECT
     MIN(CASE WHEN d_date = DATE '2001-02-14' THEN d_date_sk END) AS start_sk,
     MIN(CASE WHEN d_date = DATE '2001-03-16' THEN d_date_sk END) AS mid_sk,
     MIN(CASE WHEN d_date = DATE '2001-04-15' THEN d_date_sk END) AS end_sk
 FROM date_keys
)

SELECT w_state, i_item_id,
       SUM
       (
               CASE WHEN ws_sold_date_sk < date_bounds.mid_sk
                        THEN ws_sales_price - COALESCE(wr_refunded_cash,0)
                    ELSE 0.0 END
       ) AS sales_before,
       SUM
       (
               CASE WHEN ws_sold_date_sk >= date_bounds.mid_sk
                        THEN ws_sales_price - COALESCE(wr_refunded_cash,0)
                    ELSE 0.0 END
       ) AS sales_after
FROM
    date_bounds,
    (
        SELECT ws_item_sk,
               ws_warehouse_sk,
               ws_sold_date_sk,
               ws_sales_price,
               wr_refunded_cash
        FROM web_sales ws
                 LEFT OUTER JOIN web_returns wr ON
            (
                ws.ws_order_number = wr.wr_order_number
                    AND ws.ws_item_sk = wr.wr_item_sk
                )
        WHERE ws_sold_date_sk BETWEEN date_bounds.start_sk
          AND date_bounds.end_sk
    ) as a1
        JOIN item i ON a1.ws_item_sk = i.i_item_sk
        JOIN warehouse w ON a1.ws_warehouse_sk = w.w_warehouse_sk
GROUP BY w_state,i_item_id
ORDER BY w_state,i_item_id
    LIMIT 100
;