WITH temp_table_1 as
         (
             SELECT ss_customer_sk AS customer_sk,
                    sum( case when (d_year = 2001) THEN (((ss_ext_list_price-ss_ext_wholesale_cost-ss_ext_discount_amt)+ss_ext_sales_price)/2.0) ELSE 0.0 END)
                                   AS first_year_total,
                    sum( case when (d_year = 2002) THEN (((ss_ext_list_price-ss_ext_wholesale_cost-ss_ext_discount_amt)+ss_ext_sales_price)/2.0) ELSE 0.0 END)
                                   AS second_year_total
             FROM store_sales,
                  date_dim
             WHERE ss_sold_date_sk = d_date_sk
               AND   d_year BETWEEN 2001 AND 2002
             GROUP BY ss_customer_sk
             -- first_year_total is an aggregation, rewrite all sum () statement
             HAVING sum( case when (d_year = 2001) THEN (((ss_ext_list_price-ss_ext_wholesale_cost-ss_ext_discount_amt)+ss_ext_sales_price)/2.0) ELSE 0.0 END) > 0.0
         ),
     temp_table_2 AS
         (
             SELECT ws_bill_customer_sk AS customer_sk ,
                    sum( case when (d_year = 2001) THEN (((ws_ext_list_price-ws_ext_wholesale_cost-ws_ext_discount_amt)+ws_ext_sales_price)/2.0) ELSE 0.0 END)
                                        AS first_year_total,
                    sum( case when (d_year = 2002) THEN (((ws_ext_list_price-ws_ext_wholesale_cost-ws_ext_discount_amt)+ws_ext_sales_price)/2.0) ELSE 0.0 END)
                                        AS second_year_total
             FROM web_sales,
                  date_dim
             WHERE ws_sold_date_sk = d_date_sk
               AND   d_year BETWEEN 2001 AND 2002
             GROUP BY ws_bill_customer_sk
             -- required to avoid division by 0, because later we will divide by this value
             HAVING sum( case when (d_year = 2001) THEN (((ws_ext_list_price-ws_ext_wholesale_cost-ws_ext_discount_amt)+ws_ext_sales_price)/2.0)ELSE 0.0 END) > 0.0
         )
-- MAIN QUERY
SELECT
    CAST( (web.second_year_total / web.first_year_total) AS FLOAT8) AS web_sales_increase_ratio,
    c_customer_sk,
    c_first_name,
    c_last_name,
    c_preferred_cust_flag,
    c_birth_country,
    c_login,
    c_email_address
FROM temp_table_1 store,
     temp_table_2 web,
     customer c
WHERE store.customer_sk = web.customer_sk
  AND  web.customer_sk = c_customer_sk
  -- if customer has sales in first year for both store and websales,
  -- select him only if web second_year_total/first_year_total
  -- ratio is bigger then his store second_year_total/first_year_total ratio.
  AND (web.second_year_total / web.first_year_total) >
      (store.second_year_total / store.first_year_total)
ORDER BY
    web_sales_increase_ratio DESC,
    c_customer_sk,
    c_first_name,
    c_last_name,
    c_preferred_cust_flag,
    c_birth_country,
    c_login
    LIMIT 100
;
