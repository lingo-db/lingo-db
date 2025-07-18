WITH dates as (
    select min(d_date_sk) as min_d_date_sk,
           max(d_date_sk) as max_d_date_sk
    from date_dim
    where d_year = 2001
      and d_moy = 12
)

SELECT sum(promotional) as promotional,
       sum(total) as total,
       CASE WHEN sum(total) > 0.0 THEN (100.0 * sum(promotional)) / sum(total)
            ELSE 0.0 END as promo_percent
FROM
    (
        SELECT p_channel_email,
               p_channel_dmail,
               p_channel_tv,
               SUM( CAST(ss_ext_sales_price AS FLOAT8) ) total,
               CASE WHEN (p_channel_dmail = 'Y' OR p_channel_email = 'Y' OR p_channel_tv = 'Y')
                        THEN SUM(CAST(ss_ext_sales_price AS FLOAT8)) ELSE 0 END as promotional
        FROM dates, store_sales ss
                 JOIN promotion p ON ss.ss_promo_sk = p.p_promo_sk
                 JOIN item i on ss.ss_item_sk = i.i_item_sk
                 JOIN store s on ss.ss_store_sk = s.s_store_sk
                 JOIN customer c on c.c_customer_sk = ss.ss_customer_sk
                 JOIN customer_address ca
                            on c.c_current_addr_sk = ca.ca_address_sk
        WHERE i.i_category IN ('Books', 'Music')
          AND s.s_gmt_offset = -5.0
          AND ca.ca_gmt_offset = -5.0
          AND ss.ss_sold_date_sk >= dates.min_d_date_sk
          AND ss.ss_sold_date_sk <= dates.max_d_date_sk
        GROUP BY p_channel_email, p_channel_dmail, p_channel_tv
    ) sum_promotional
-- we don't need a 'ON' join condition. result is just two numbers.
;