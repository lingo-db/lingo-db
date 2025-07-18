WITH temp_table1 as (
    SELECT
        ss.ss_customer_sk AS customer_sk,
        sum( case when (d_year = 2001) THEN ss_net_paid ELSE 0.0 END) first_year_total,
        sum( case when (d_year = 2002) THEN ss_net_paid ELSE 0.0 END) second_year_total
    FROM store_sales ss
             JOIN
         (
             SELECT d_date_sk, d_year
             FROM date_dim d
             WHERE d.d_year in (2001, 2002)
         ) dd on ( ss.ss_sold_date_sk = dd.d_date_sk )
    GROUP BY ss.ss_customer_sk
    HAVING sum( case when (d_year = 2001) THEN ss_net_paid ELSE 0.0 END) > 0.0
),
    temp_table2 as (
        SELECT
            ws.ws_bill_customer_sk AS customer_sk,
            sum( case when (d_year = 2001) THEN ws_net_paid ELSE 0.0 END) first_year_total,
            sum( case when (d_year = 2002) THEN ws_net_paid ELSE 0.0 END) second_year_total
        FROM web_sales ws
                 JOIN
             (
                 SELECT d_date_sk, d_year
                 FROM date_dim d
                 WHERE d.d_year in (2001, 2002)
             ) dd ON ( ws.ws_sold_date_sk = dd.d_date_sk )
        GROUP BY ws.ws_bill_customer_sk
        HAVING sum( case when (d_year = 2001) THEN ws_net_paid ELSE 0.0 END) > 0.0
)
SELECT
    CAST(c_customer_sk AS BIGINT) as c_customer_sk,
    c_first_name,
    c_last_name,
    (store.second_year_total / store.first_year_total) AS storeSalesIncreaseRatio,
    (web.second_year_total / web.first_year_total) AS webSalesIncreaseRatio
FROM temp_table1 store,
     temp_table2 web,
     customer c
WHERE store.customer_sk = web.customer_sk
  AND web.customer_sk = c_customer_sk
  AND (web.second_year_total / web.first_year_total) > (store.second_year_total / store.first_year_total)
ORDER BY webSalesIncreaseRatio DESC,
         c_customer_sk,
         c_first_name,
         c_last_name
    LIMIT 100
;