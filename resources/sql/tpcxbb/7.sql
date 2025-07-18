WITH temp_table as
         (
             SELECT k.i_item_sk
             FROM item k,
                  (
                      SELECT i_category,
                             SUM(j.i_current_price) / COUNT(j.i_current_price) * 1.2 AS avg_price
                      FROM item j
                      GROUP BY j.i_category
                  ) avgCategoryPrice
             WHERE avgCategoryPrice.i_category = k.i_category
               AND k.i_current_price > avgCategoryPrice.avg_price
         )
SELECT ca_state, COUNT(*) AS cnt
FROM
    customer_address a,
    customer c,
    store_sales s,
    temp_table highPriceItems
WHERE a.ca_address_sk = c.c_current_addr_sk
  AND c.c_customer_sk = s.ss_customer_sk
  AND ca_state IS NOT NULL
  AND ss_item_sk = highPriceItems.i_item_sk
  AND s.ss_sold_date_sk IN
      (
          SELECT d_date_sk
          FROM date_dim
          WHERE d_year = 2004
            AND d_moy = 7
      )
GROUP BY ca_state
HAVING COUNT(*) >= 10
ORDER BY cnt DESC, ca_state
    LIMIT 10
;