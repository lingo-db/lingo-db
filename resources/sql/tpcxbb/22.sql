SELECT
    w_warehouse_name,
    i_item_id,
    SUM(CASE WHEN datediff('second', timestamp '2001-05-08 00:00:00', d_date)
                      / 1000000 < 0 THEN inv_quantity_on_hand ELSE 0 END) AS inv_before,
    SUM(CASE WHEN datediff('second', timestamp '2001-05-08 00:00:00', d_date)
                      / 1000000 >= 0 THEN inv_quantity_on_hand ELSE 0 END) AS inv_after
FROM
    inventory inv,
    item i,
    warehouse w,
    date_dim d
WHERE i_current_price BETWEEN 0.98 AND 1.5
  AND i_item_sk        = inv_item_sk
  AND inv_warehouse_sk = w_warehouse_sk
  AND inv_date_sk      = d_date_sk
  AND datediff('second', timestamp '2001-05-08 00:00:00', d_date) / 1000000 >= -30
  AND datediff('second', timestamp '2001-05-08 00:00:00', d_date) / 1000000 <= 30
GROUP BY w_warehouse_name, i_item_id
HAVING SUM(CASE WHEN datediff('second', timestamp '2001-05-08', d_date)
                         / 1000000 < 0 THEN inv_quantity_on_hand ELSE 0 END) > 0
   AND
    (
        CAST(
                SUM (CASE WHEN datediff('second', timestamp '2001-05-08 00:00:00', d_date) / 1000000 >= 0 THEN inv_quantity_on_hand ELSE 0 END) AS FLOAT8)
            / CAST( SUM(CASE WHEN datediff('second', timestamp '2001-05-08 00:00:00', d_date) / 1000000 < 0 THEN inv_quantity_on_hand ELSE 0 END)
            AS FLOAT8) >= 0.666667
        )
   AND
    (
        CAST(
                SUM(CASE WHEN datediff('second', timestamp '2001-05-08 00:00:00', d_date) / 1000000 >= 0 THEN inv_quantity_on_hand ELSE 0 END) AS FLOAT8)
            / CAST ( SUM(CASE WHEN datediff('second', timestamp '2001-05-08 00:00:00', d_date) / 1000000 < 0 THEN inv_quantity_on_hand ELSE 0 END)
            AS FLOAT8) <= 1.50
        )
ORDER BY w_warehouse_name, i_item_id
    LIMIT 100
;
