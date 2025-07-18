SELECT
    part_i.i_item_id AS i_item_id,
    part_i.i_item_desc AS i_item_desc,
    part_s.s_store_id AS s_store_id,
    part_s.s_store_name AS s_store_name,
    CAST(SUM(part_ss.ss_quantity) AS BIGINT) AS store_sales_quantity,
    CAST(SUM(part_sr.sr_return_quantity) AS BIGINT) AS store_returns_quantity,
    CAST(SUM(part_ws.ws_quantity) AS BIGINT) AS web_sales_quantity
FROM
    (
        SELECT
            sr_item_sk,
            sr_customer_sk,
            sr_ticket_number,
            sr_return_quantity
        FROM
            store_returns sr,
            date_dim d2
        WHERE d2.d_year = 2003
          AND d2.d_moy BETWEEN 1 AND 7 --which were returned in the next six months
          AND sr.sr_returned_date_sk = d2.d_date_sk
    ) part_sr
        INNER JOIN
    (
        SELECT
            ws_item_sk,
            ws_bill_customer_sk,
            ws_quantity
        FROM
            web_sales ws,
            date_dim d3
        -- in the following three years (re-purchased by the returning customer afterwards through the web sales channel)
        WHERE d3.d_year BETWEEN 2003 AND 2005
          AND ws.ws_sold_date_sk = d3.d_date_sk
    ) part_ws ON
        (
            part_sr.sr_item_sk = part_ws.ws_item_sk
                AND part_sr.sr_customer_sk = part_ws.ws_bill_customer_sk
            ) INNER JOIN
    (
        SELECT
            ss_item_sk,
            ss_store_sk,
            ss_customer_sk,
            ss_ticket_number,
            ss_quantity
        FROM
            store_sales ss,
            date_dim d1
        WHERE d1.d_year = 2003
          AND d1.d_moy = 1
          AND ss.ss_sold_date_sk = d1.d_date_sk
    ) part_ss ON
        (
            part_ss.ss_ticket_number = part_sr.sr_ticket_number
                AND part_ss.ss_item_sk = part_sr.sr_item_sk
                AND part_ss.ss_customer_sk = part_sr.sr_customer_sk
            )
        INNER JOIN store part_s ON
        (
            part_s.s_store_sk = part_ss.ss_store_sk
            )
        INNER JOIN item part_i ON
        (
            part_i.i_item_sk = part_ss.ss_item_sk
            )
GROUP BY
    part_i.i_item_id,
    part_i.i_item_desc,
    part_s.s_store_id,
    part_s.s_store_name
ORDER BY
    part_i.i_item_id,
    part_i.i_item_desc,
    part_s.s_store_id,
    part_s.s_store_name
    LIMIT 100
;