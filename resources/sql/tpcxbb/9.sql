SELECT SUM(ss1.ss_quantity)
FROM store_sales ss1,
     date_dim dd,customer_address ca1,
     store s,
     customer_demographics cd
-- select date range
WHERE ss1.ss_sold_date_sk = dd.d_date_sk
  AND dd.d_year = 2001
  AND ss1.ss_addr_sk = ca1.ca_address_sk
  AND s.s_store_sk = ss1.ss_store_sk
  AND cd.cd_demo_sk = ss1.ss_cdemo_sk
  AND
    (
        (
            cd.cd_marital_status = 'M'
                AND cd.cd_education_status = '4 yr Degree'
                AND 100 <= ss1.ss_sales_price
                AND ss1.ss_sales_price <= 150
            )
            OR
        (
            cd.cd_marital_status = 'M'
                AND cd.cd_education_status = '4 yr Degree'
                AND 50 <= ss1.ss_sales_price
                AND ss1.ss_sales_price <= 200
            )
            OR
        (
            cd.cd_marital_status = 'M'
                AND cd.cd_education_status = '4 yr Degree'
                AND 150 <= ss1.ss_sales_price
                AND ss1.ss_sales_price <= 200
            )
        )
  AND
    (
        (
            ca1.ca_country = 'United States'
                AND ca1.ca_state IN ('KY', 'GA', 'NM')
                AND 0 <= ss1.ss_net_profit
                AND ss1.ss_net_profit <= 2000
            )
            OR
        (
            ca1.ca_country = 'United States'
                AND ca1.ca_state IN ('MT', 'OR', 'IN')
                AND 150 <= ss1.ss_net_profit
                AND ss1.ss_net_profit <= 3000
            )
            OR
        (
            ca1.ca_country = 'United States'
                AND ca1.ca_state IN ('WI', 'MO', 'WV')
                AND 50 <= ss1.ss_net_profit
                AND ss1.ss_net_profit <= 25000
            )
        )
;
