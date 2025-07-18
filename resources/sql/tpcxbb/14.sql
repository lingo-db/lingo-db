SELECT CASE WHEN pmc > 0.0 THEN CAST (amc AS FLOAT8) / CAST (pmc AS FLOAT8) ELSE -1.0 END AS am_pm_ratio
FROM
    (
        SELECT SUM(amc1) AS amc, SUM(pmc1) AS pmc
        FROM
            (
                SELECT
                    CASE WHEN t_hour BETWEEN 7 AND 8 THEN COUNT(1) ELSE 0 END AS amc1,
                    CASE WHEN t_hour BETWEEN 19 AND 20 THEN COUNT(1) ELSE 0 END AS pmc1
                FROM web_sales ws
                         JOIN household_demographics hd ON (hd.hd_demo_sk = ws.ws_ship_hdemo_sk and hd.hd_dep_count = 5)
                         JOIN web_page wp ON (wp.wp_web_page_sk = ws.ws_web_page_sk and wp.wp_char_count BETWEEN 5000 AND 6000)
                         JOIN time_dim td ON (td.t_time_sk = ws.ws_sold_time_sk and td.t_hour IN (7,8,19,20))
                GROUP BY t_hour
            ) cnt_am_pm
    ) sum_am_pm
;