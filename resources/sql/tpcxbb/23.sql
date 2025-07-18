WITH inv_dates as (
    SELECT  inv_warehouse_sk,
            inv_item_sk,
            inv_quantity_on_hand,
            d_moy
    FROM inventory inv
    INNER JOIN date_dim d ON inv.inv_date_sk = d.d_date_sk
    AND d.d_year = 2001
    AND d_moy between 1 AND 2
),
    mean_df as (
        SELECT inv_warehouse_sk,
               inv_item_sk,
               d_moy,
               AVG(inv_quantity_on_hand) AS q_mean -- TODO this was casted to float first, but an error came out
        FROM inv_dates
        GROUP BY inv_warehouse_sk, inv_item_sk, d_moy
    ),
    pre_iteration as ( -- needed with current parser
        SELECT id.inv_warehouse_sk,
               id.inv_item_sk,
               id.d_moy,
               md.q_mean,
               SUM( (id.inv_quantity_on_hand - md.q_mean) * (id.inv_quantity_on_hand - md.q_mean)) as sm,
               COUNT(id.inv_quantity_on_hand) - 1.0 AS cnt
        FROM mean_df md
                 INNER JOIN inv_dates id ON id.inv_warehouse_sk = md.inv_warehouse_sk
            AND id.inv_item_sk = md.inv_item_sk
            AND id.d_moy = md.d_moy
            AND md.q_mean > 0.0
        GROUP BY id.inv_warehouse_sk, id.inv_item_sk, id.d_moy, md.q_mean
    ),
    iteration as (
        SELECT  inv_warehouse_sk,
                inv_item_sk,
                d_moy,
                q_mean,
                SQRT(sm::FLOAT8) / cnt as q_std -- Note: reference implementation takes the sqrt of the decimal, we convert to float
        FROM pre_iteration
    ),
    temp_table as (
        SELECT inv_warehouse_sk,
               inv_item_sk,
               d_moy,
               q_std / q_mean AS qty_cov
        FROM iteration
        WHERE (q_std / q_mean) >= 1.3
    )

SELECT inv1.inv_warehouse_sk,
       inv1.inv_item_sk,
       inv1.d_moy,
       inv1.qty_cov AS cov,
       inv2.d_moy AS inv2_d_moy,
       inv2.qty_cov AS inv2_cov
FROM temp_table inv1
         INNER JOIN temp_table inv2 ON inv1.inv_warehouse_sk = inv2.inv_warehouse_sk
    AND inv1.inv_item_sk = inv2.inv_item_sk
    AND inv1.d_moy = 1
    AND inv2.d_moy = 2
ORDER BY inv1.inv_warehouse_sk,
    inv1.inv_item_sk
;
