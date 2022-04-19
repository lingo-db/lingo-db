--//RUN: run-sql %s %S/../../../resources/data/tpch | FileCheck %s
--//CHECK: |                       p_brand  |                        p_type  |                        p_size  |                  supplier_cnt  |
--//CHECK: -------------------------------------------------------------------------------------------------------------------------------------
--//CHECK: |                    "Brand#14"  |       "SMALL ANODIZED NICKEL"  |                            45  |                            12  |
--//CHECK: |                    "Brand#22"  |       "SMALL BURNISHED BRASS"  |                            19  |                            12  |
--//CHECK: |                    "Brand#25"  |       "PROMO POLISHED COPPER"  |                            14  |                            12  |
--//CHECK: |                    "Brand#35"  |        "LARGE ANODIZED STEEL"  |                            45  |                            12  |
--//CHECK: |                    "Brand#35"  |        "PROMO BRUSHED COPPER"  |                             9  |                            12  |
--//CHECK: |                    "Brand#51"  |      "ECONOMY ANODIZED STEEL"  |                             9  |                            12  |
--//CHECK: |                    "Brand#53"  |        "LARGE BRUSHED NICKEL"  |                            45  |                            12  |
--//CHECK: |                    "Brand#11"  |     "ECONOMY POLISHED COPPER"  |                            14  |                             8  |
--//CHECK: |                    "Brand#11"  |          "LARGE PLATED STEEL"  |                            23  |                             8  |
--//CHECK: |                    "Brand#11"  |        "PROMO POLISHED STEEL"  |                            23  |                             8  |
--//CHECK: |                    "Brand#11"  |    "STANDARD ANODIZED COPPER"  |                             9  |                             8  |
--//CHECK: |                    "Brand#12"  |     "ECONOMY BURNISHED BRASS"  |                             9  |                             8  |
--//CHECK: |                    "Brand#12"  |        "LARGE ANODIZED BRASS"  |                            14  |                             8  |
--//CHECK: |                    "Brand#12"  |          "SMALL ANODIZED TIN"  |                            23  |                             8  |
--//CHECK: |                    "Brand#12"  |        "SMALL BRUSHED NICKEL"  |                            23  |                             8  |
--//CHECK: |                    "Brand#12"  |     "STANDARD ANODIZED BRASS"  |                             3  |                             8  |
--//CHECK: |                    "Brand#12"  |      "STANDARD BURNISHED TIN"  |                            23  |                             8  |
--//CHECK: |                    "Brand#13"  |      "ECONOMY POLISHED BRASS"  |                             9  |                             8  |
--//CHECK: |                    "Brand#13"  |      "LARGE BURNISHED COPPER"  |                            45  |                             8  |
--//CHECK: |                    "Brand#13"  |       "MEDIUM ANODIZED STEEL"  |                            23  |                             8  |
--//CHECK: |                    "Brand#13"  |        "MEDIUM PLATED NICKEL"  |                             3  |                             8  |
--//CHECK: |                    "Brand#13"  |       "PROMO BURNISHED BRASS"  |                             9  |                             8  |
--//CHECK: |                    "Brand#13"  |        "PROMO POLISHED BRASS"  |                             3  |                             8  |
--//CHECK: |                    "Brand#13"  |          "PROMO POLISHED TIN"  |                            36  |                             8  |
--//CHECK: |                    "Brand#13"  |       "SMALL BURNISHED STEEL"  |                            23  |                             8  |
--//CHECK: |                    "Brand#13"  |      "STANDARD BRUSHED STEEL"  |                             9  |                             8  |
--//CHECK: |                    "Brand#14"  |         "ECONOMY BRUSHED TIN"  |                             3  |                             8  |
--//CHECK: |                    "Brand#14"  |       "ECONOMY BURNISHED TIN"  |                            23  |                             8  |
--//CHECK: |                    "Brand#14"  |         "PROMO BRUSHED STEEL"  |                             9  |                             8  |
--//CHECK: |                    "Brand#14"  |            "PROMO PLATED TIN"  |                            45  |                             8  |
--//CHECK: |                    "Brand#15"  |          "ECONOMY PLATED TIN"  |                             9  |                             8  |
--//CHECK: |                    "Brand#15"  |     "STANDARD BRUSHED COPPER"  |                            14  |                             8  |
--//CHECK: |                    "Brand#15"  |         "STANDARD PLATED TIN"  |                             3  |                             8  |
--//CHECK: |                    "Brand#21"  |        "ECONOMY POLISHED TIN"  |                             3  |                             8  |
--//CHECK: |                    "Brand#21"  |       "PROMO POLISHED COPPER"  |                             9  |                             8  |
--//CHECK: |                    "Brand#21"  |          "PROMO POLISHED TIN"  |                            49  |                             8  |
--//CHECK: |                    "Brand#21"  |        "SMALL POLISHED STEEL"  |                             3  |                             8  |
--//CHECK: |                    "Brand#21"  |       "STANDARD PLATED BRASS"  |                            49  |                             8  |
--//CHECK: |                    "Brand#21"  |      "STANDARD PLATED NICKEL"  |                            49  |                             8  |
--//CHECK: |                    "Brand#22"  |        "ECONOMY ANODIZED TIN"  |                            49  |                             8  |
--//CHECK: |                    "Brand#22"  |       "ECONOMY BRUSHED BRASS"  |                            14  |                             8  |
--//CHECK: |                    "Brand#22"  |         "LARGE BURNISHED TIN"  |                            36  |                             8  |
--//CHECK: |                    "Brand#22"  |       "MEDIUM ANODIZED STEEL"  |                            36  |                             8  |
--//CHECK: |                    "Brand#22"  |         "MEDIUM PLATED STEEL"  |                             9  |                             8  |
--//CHECK: |                    "Brand#22"  |       "PROMO POLISHED NICKEL"  |                             9  |                             8  |
--//CHECK: |                    "Brand#22"  |        "SMALL ANODIZED STEEL"  |                            19  |                             8  |
--//CHECK: |                    "Brand#22"  |    "STANDARD ANODIZED COPPER"  |                            23  |                             8  |
--//CHECK: |                    "Brand#23"  |      "ECONOMY BRUSHED NICKEL"  |                            23  |                             8  |
--//CHECK: |                    "Brand#23"  |        "LARGE ANODIZED BRASS"  |                             9  |                             8  |
--//CHECK: |                    "Brand#23"  |        "LARGE ANODIZED STEEL"  |                            23  |                             8  |
--//CHECK: |                    "Brand#23"  |        "SMALL BRUSHED COPPER"  |                            23  |                             8  |
--//CHECK: |                    "Brand#23"  |        "STANDARD BRUSHED TIN"  |                             3  |                             8  |
--//CHECK: |                    "Brand#23"  |   "STANDARD BURNISHED NICKEL"  |                            49  |                             8  |
--//CHECK: |                    "Brand#23"  |      "STANDARD PLATED NICKEL"  |                            36  |                             8  |
--//CHECK: |                    "Brand#24"  |      "ECONOMY ANODIZED BRASS"  |                            19  |                             8  |
--//CHECK: |                    "Brand#24"  |      "ECONOMY POLISHED BRASS"  |                            36  |                             8  |
--//CHECK: |                    "Brand#24"  |       "LARGE BURNISHED STEEL"  |                            14  |                             8  |
--//CHECK: |                    "Brand#24"  |        "MEDIUM PLATED NICKEL"  |                            36  |                             8  |
--//CHECK: |                    "Brand#25"  |       "ECONOMY BRUSHED STEEL"  |                            49  |                             8  |
--//CHECK: |                    "Brand#25"  |        "MEDIUM BURNISHED TIN"  |                             3  |                             8  |
--//CHECK: |                    "Brand#25"  |          "PROMO ANODIZED TIN"  |                            36  |                             8  |
--//CHECK: |                    "Brand#25"  |         "PROMO PLATED NICKEL"  |                             3  |                             8  |
--//CHECK: |                    "Brand#25"  |       "SMALL BURNISHED BRASS"  |                             3  |                             8  |
--//CHECK: |                    "Brand#31"  |        "LARGE ANODIZED BRASS"  |                             3  |                             8  |
--//CHECK: |                    "Brand#31"  |       "SMALL ANODIZED COPPER"  |                             3  |                             8  |
--//CHECK: |                    "Brand#31"  |       "SMALL ANODIZED NICKEL"  |                             9  |                             8  |
--//CHECK: |                    "Brand#31"  |        "SMALL ANODIZED STEEL"  |                            14  |                             8  |
--//CHECK: |                    "Brand#32"  |       "MEDIUM ANODIZED STEEL"  |                            49  |                             8  |
--//CHECK: |                    "Brand#32"  |     "MEDIUM BURNISHED COPPER"  |                            19  |                             8  |
--//CHECK: |                    "Brand#32"  |       "SMALL BURNISHED STEEL"  |                            23  |                             8  |
--//CHECK: |                    "Brand#32"  |    "STANDARD BURNISHED STEEL"  |                            45  |                             8  |
--//CHECK: |                    "Brand#34"  |     "ECONOMY ANODIZED NICKEL"  |                            49  |                             8  |
--//CHECK: |                    "Brand#34"  |         "LARGE BURNISHED TIN"  |                            49  |                             8  |
--//CHECK: |                    "Brand#34"  |     "MEDIUM BURNISHED NICKEL"  |                             3  |                             8  |
--//CHECK: |                    "Brand#34"  |          "PROMO ANODIZED TIN"  |                             3  |                             8  |
--//CHECK: |                    "Brand#34"  |           "SMALL BRUSHED TIN"  |                             3  |                             8  |
--//CHECK: |                    "Brand#34"  |      "STANDARD BURNISHED TIN"  |                            23  |                             8  |
--//CHECK: |                    "Brand#35"  |        "MEDIUM BRUSHED STEEL"  |                            45  |                             8  |
--//CHECK: |                    "Brand#35"  |       "PROMO BURNISHED STEEL"  |                            14  |                             8  |
--//CHECK: |                    "Brand#35"  |       "SMALL BURNISHED STEEL"  |                            23  |                             8  |
--//CHECK: |                    "Brand#35"  |       "SMALL POLISHED COPPER"  |                            14  |                             8  |
--//CHECK: |                    "Brand#35"  |      "STANDARD PLATED COPPER"  |                             9  |                             8  |
--//CHECK: |                    "Brand#41"  |       "ECONOMY BRUSHED BRASS"  |                            23  |                             8  |
--//CHECK: |                    "Brand#41"  |       "LARGE BURNISHED STEEL"  |                            23  |                             8  |
--//CHECK: |                    "Brand#41"  |         "PROMO BURNISHED TIN"  |                            14  |                             8  |
--//CHECK: |                    "Brand#41"  |          "PROMO PLATED STEEL"  |                            36  |                             8  |
--//CHECK: |                    "Brand#41"  |          "PROMO POLISHED TIN"  |                            19  |                             8  |
--//CHECK: |                    "Brand#41"  |      "SMALL BURNISHED COPPER"  |                            23  |                             8  |
--//CHECK: |                    "Brand#42"  |          "LARGE POLISHED TIN"  |                            14  |                             8  |
--//CHECK: |                    "Brand#42"  |         "MEDIUM ANODIZED TIN"  |                            49  |                             8  |
--//CHECK: |                    "Brand#42"  |          "MEDIUM BRUSHED TIN"  |                            14  |                             8  |
--//CHECK: |                    "Brand#42"  |     "MEDIUM BURNISHED NICKEL"  |                            23  |                             8  |
--//CHECK: |                    "Brand#42"  |        "MEDIUM PLATED COPPER"  |                            45  |                             8  |
--//CHECK: |                    "Brand#42"  |           "MEDIUM PLATED TIN"  |                            45  |                             8  |
--//CHECK: |                    "Brand#42"  |         "SMALL PLATED COPPER"  |                            36  |                             8  |
--//CHECK: |                    "Brand#43"  |       "ECONOMY BRUSHED STEEL"  |                            45  |                             8  |
--//CHECK: |                    "Brand#43"  |        "LARGE BRUSHED COPPER"  |                            19  |                             8  |
--//CHECK: |                    "Brand#43"  |         "PROMO BRUSHED BRASS"  |                            36  |                             8  |
--//CHECK: |                    "Brand#43"  |         "SMALL BURNISHED TIN"  |                            45  |                             8  |
--//CHECK: |                    "Brand#43"  |         "SMALL PLATED COPPER"  |                            45  |                             8  |
--//CHECK: |                           ...  |                           ...  |                           ...  |                           ...  |
--//CHECK: |                    "Brand#55"  |     "ECONOMY POLISHED NICKEL"  |                             9  |                             4  |
--//CHECK: |                    "Brand#55"  |         "LARGE BRUSHED BRASS"  |                            23  |                             4  |
--//CHECK: |                    "Brand#55"  |         "LARGE BRUSHED BRASS"  |                            45  |                             4  |
--//CHECK: |                    "Brand#55"  |        "LARGE BRUSHED COPPER"  |                            49  |                             4  |
--//CHECK: |                    "Brand#55"  |        "LARGE BRUSHED NICKEL"  |                             9  |                             4  |
--//CHECK: |                    "Brand#55"  |        "LARGE BRUSHED NICKEL"  |                            14  |                             4  |
--//CHECK: |                    "Brand#55"  |       "LARGE BURNISHED BRASS"  |                             3  |                             4  |
--//CHECK: |                    "Brand#55"  |      "LARGE BURNISHED COPPER"  |                            14  |                             4  |
--//CHECK: |                    "Brand#55"  |      "LARGE BURNISHED COPPER"  |                            36  |                             4  |
--//CHECK: |                    "Brand#55"  |          "LARGE PLATED BRASS"  |                            45  |                             4  |
--//CHECK: |                    "Brand#55"  |         "LARGE PLATED COPPER"  |                            19  |                             4  |
--//CHECK: |                    "Brand#55"  |         "LARGE PLATED NICKEL"  |                             9  |                             4  |
--//CHECK: |                    "Brand#55"  |          "LARGE PLATED STEEL"  |                             9  |                             4  |
--//CHECK: |                    "Brand#55"  |            "LARGE PLATED TIN"  |                             9  |                             4  |
--//CHECK: |                    "Brand#55"  |            "LARGE PLATED TIN"  |                            14  |                             4  |
--//CHECK: |                    "Brand#55"  |            "LARGE PLATED TIN"  |                            23  |                             4  |
--//CHECK: |                    "Brand#55"  |       "LARGE POLISHED NICKEL"  |                             3  |                             4  |
--//CHECK: |                    "Brand#55"  |        "LARGE POLISHED STEEL"  |                            36  |                             4  |
--//CHECK: |                    "Brand#55"  |        "LARGE POLISHED STEEL"  |                            45  |                             4  |
--//CHECK: |                    "Brand#55"  |      "MEDIUM ANODIZED COPPER"  |                             9  |                             4  |
--//CHECK: |                    "Brand#55"  |        "MEDIUM BRUSHED BRASS"  |                             3  |                             4  |
--//CHECK: |                    "Brand#55"  |       "MEDIUM BRUSHED NICKEL"  |                            23  |                             4  |
--//CHECK: |                    "Brand#55"  |          "MEDIUM BRUSHED TIN"  |                            45  |                             4  |
--//CHECK: |                    "Brand#55"  |      "MEDIUM BURNISHED BRASS"  |                            23  |                             4  |
--//CHECK: |                    "Brand#55"  |     "MEDIUM BURNISHED COPPER"  |                            36  |                             4  |
--//CHECK: |                    "Brand#55"  |     "MEDIUM BURNISHED NICKEL"  |                             3  |                             4  |
--//CHECK: |                    "Brand#55"  |      "MEDIUM BURNISHED STEEL"  |                            14  |                             4  |
--//CHECK: |                    "Brand#55"  |      "MEDIUM BURNISHED STEEL"  |                            36  |                             4  |
--//CHECK: |                    "Brand#55"  |        "MEDIUM PLATED NICKEL"  |                            23  |                             4  |
--//CHECK: |                    "Brand#55"  |       "PROMO ANODIZED COPPER"  |                            14  |                             4  |
--//CHECK: |                    "Brand#55"  |       "PROMO ANODIZED COPPER"  |                            49  |                             4  |
--//CHECK: |                    "Brand#55"  |        "PROMO ANODIZED STEEL"  |                            36  |                             4  |
--//CHECK: |                    "Brand#55"  |          "PROMO ANODIZED TIN"  |                            23  |                             4  |
--//CHECK: |                    "Brand#55"  |        "PROMO BRUSHED NICKEL"  |                            36  |                             4  |
--//CHECK: |                    "Brand#55"  |         "PROMO BRUSHED STEEL"  |                             3  |                             4  |
--//CHECK: |                    "Brand#55"  |         "PROMO BRUSHED STEEL"  |                            36  |                             4  |
--//CHECK: |                    "Brand#55"  |           "PROMO BRUSHED TIN"  |                             9  |                             4  |
--//CHECK: |                    "Brand#55"  |      "PROMO BURNISHED COPPER"  |                             3  |                             4  |
--//CHECK: |                    "Brand#55"  |       "PROMO BURNISHED STEEL"  |                            14  |                             4  |
--//CHECK: |                    "Brand#55"  |         "PROMO BURNISHED TIN"  |                            23  |                             4  |
--//CHECK: |                    "Brand#55"  |         "PROMO BURNISHED TIN"  |                            49  |                             4  |
--//CHECK: |                    "Brand#55"  |         "PROMO PLATED COPPER"  |                             3  |                             4  |
--//CHECK: |                    "Brand#55"  |         "PROMO PLATED NICKEL"  |                             3  |                             4  |
--//CHECK: |                    "Brand#55"  |         "PROMO PLATED NICKEL"  |                            14  |                             4  |
--//CHECK: |                    "Brand#55"  |         "PROMO PLATED NICKEL"  |                            23  |                             4  |
--//CHECK: |                    "Brand#55"  |            "PROMO PLATED TIN"  |                             3  |                             4  |
--//CHECK: |                    "Brand#55"  |       "PROMO POLISHED COPPER"  |                             3  |                             4  |
--//CHECK: |                    "Brand#55"  |        "SMALL ANODIZED BRASS"  |                            19  |                             4  |
--//CHECK: |                    "Brand#55"  |       "SMALL ANODIZED NICKEL"  |                            45  |                             4  |
--//CHECK: |                    "Brand#55"  |        "SMALL BRUSHED COPPER"  |                            14  |                             4  |
--//CHECK: |                    "Brand#55"  |        "SMALL BRUSHED COPPER"  |                            45  |                             4  |
--//CHECK: |                    "Brand#55"  |       "SMALL BURNISHED BRASS"  |                            14  |                             4  |
--//CHECK: |                    "Brand#55"  |         "SMALL BURNISHED TIN"  |                             3  |                             4  |
--//CHECK: |                    "Brand#55"  |         "SMALL BURNISHED TIN"  |                            49  |                             4  |
--//CHECK: |                    "Brand#55"  |          "SMALL PLATED BRASS"  |                            45  |                             4  |
--//CHECK: |                    "Brand#55"  |         "SMALL PLATED COPPER"  |                            23  |                             4  |
--//CHECK: |                    "Brand#55"  |         "SMALL PLATED COPPER"  |                            36  |                             4  |
--//CHECK: |                    "Brand#55"  |         "SMALL PLATED COPPER"  |                            45  |                             4  |
--//CHECK: |                    "Brand#55"  |         "SMALL PLATED COPPER"  |                            49  |                             4  |
--//CHECK: |                    "Brand#55"  |         "SMALL PLATED NICKEL"  |                             9  |                             4  |
--//CHECK: |                    "Brand#55"  |          "SMALL PLATED STEEL"  |                             9  |                             4  |
--//CHECK: |                    "Brand#55"  |            "SMALL PLATED TIN"  |                            14  |                             4  |
--//CHECK: |                    "Brand#55"  |            "SMALL PLATED TIN"  |                            36  |                             4  |
--//CHECK: |                    "Brand#55"  |       "SMALL POLISHED NICKEL"  |                            45  |                             4  |
--//CHECK: |                    "Brand#55"  |        "SMALL POLISHED STEEL"  |                            19  |                             4  |
--//CHECK: |                    "Brand#55"  |          "SMALL POLISHED TIN"  |                            19  |                             4  |
--//CHECK: |                    "Brand#55"  |     "STANDARD ANODIZED BRASS"  |                            36  |                             4  |
--//CHECK: |                    "Brand#55"  |     "STANDARD ANODIZED BRASS"  |                            49  |                             4  |
--//CHECK: |                    "Brand#55"  |     "STANDARD ANODIZED STEEL"  |                            19  |                             4  |
--//CHECK: |                    "Brand#55"  |       "STANDARD ANODIZED TIN"  |                            36  |                             4  |
--//CHECK: |                    "Brand#55"  |       "STANDARD ANODIZED TIN"  |                            49  |                             4  |
--//CHECK: |                    "Brand#55"  |      "STANDARD BRUSHED BRASS"  |                            36  |                             4  |
--//CHECK: |                    "Brand#55"  |     "STANDARD BRUSHED COPPER"  |                             3  |                             4  |
--//CHECK: |                    "Brand#55"  |     "STANDARD BRUSHED COPPER"  |                             9  |                             4  |
--//CHECK: |                    "Brand#55"  |     "STANDARD BRUSHED COPPER"  |                            23  |                             4  |
--//CHECK: |                    "Brand#55"  |      "STANDARD BRUSHED STEEL"  |                            19  |                             4  |
--//CHECK: |                    "Brand#55"  |        "STANDARD BRUSHED TIN"  |                            23  |                             4  |
--//CHECK: |                    "Brand#55"  |        "STANDARD BRUSHED TIN"  |                            45  |                             4  |
--//CHECK: |                    "Brand#55"  |    "STANDARD BURNISHED BRASS"  |                            19  |                             4  |
--//CHECK: |                    "Brand#55"  |   "STANDARD BURNISHED NICKEL"  |                             3  |                             4  |
--//CHECK: |                    "Brand#55"  |   "STANDARD BURNISHED NICKEL"  |                            36  |                             4  |
--//CHECK: |                    "Brand#55"  |    "STANDARD BURNISHED STEEL"  |                            19  |                             4  |
--//CHECK: |                    "Brand#55"  |       "STANDARD PLATED BRASS"  |                            23  |                             4  |
--//CHECK: |                    "Brand#55"  |      "STANDARD PLATED NICKEL"  |                             9  |                             4  |
--//CHECK: |                    "Brand#55"  |         "STANDARD PLATED TIN"  |                            36  |                             4  |
--//CHECK: |                    "Brand#55"  |     "STANDARD POLISHED BRASS"  |                             3  |                             4  |
--//CHECK: |                    "Brand#55"  |     "STANDARD POLISHED BRASS"  |                            49  |                             4  |
--//CHECK: |                    "Brand#55"  |    "STANDARD POLISHED COPPER"  |                            19  |                             4  |
--//CHECK: |                    "Brand#55"  |    "STANDARD POLISHED COPPER"  |                            36  |                             4  |
--//CHECK: |                    "Brand#55"  |    "STANDARD POLISHED NICKEL"  |                            14  |                             4  |
--//CHECK: |                    "Brand#55"  |     "STANDARD POLISHED STEEL"  |                             9  |                             4  |
--//CHECK: |                    "Brand#55"  |     "STANDARD POLISHED STEEL"  |                            36  |                             4  |
--//CHECK: |                    "Brand#12"  |      "LARGE BURNISHED NICKEL"  |                            14  |                             3  |
--//CHECK: |                    "Brand#12"  |          "PROMO POLISHED TIN"  |                             3  |                             3  |
--//CHECK: |                    "Brand#21"  |         "MEDIUM ANODIZED TIN"  |                             9  |                             3  |
--//CHECK: |                    "Brand#22"  |         "PROMO BRUSHED BRASS"  |                            19  |                             3  |
--//CHECK: |                    "Brand#22"  |      "PROMO BURNISHED COPPER"  |                            14  |                             3  |
--//CHECK: |                    "Brand#43"  |      "STANDARD BRUSHED BRASS"  |                            23  |                             3  |
--//CHECK: |                    "Brand#44"  |      "MEDIUM ANODIZED NICKEL"  |                             9  |                             3  |
--//CHECK: |                    "Brand#53"  |      "MEDIUM BURNISHED BRASS"  |                            49  |                             3  |
-- TPC-H Query 16

select
        p_brand,
        p_type,
        p_size,
        count(distinct ps_suppkey) as supplier_cnt
from
        partsupp,
        part
where
        p_partkey = ps_partkey
        and p_brand <> 'Brand#45'
        and p_type not like 'MEDIUM POLISHED%'
        and p_size in (49, 14, 23, 45, 19, 3, 36, 9)
        and ps_suppkey not in (
                select
                        s_suppkey
                from
                        supplier
                where
                        s_comment like '%Customer%Complaints%'
        )
group by
        p_brand,
        p_type,
        p_size
order by
        supplier_cnt desc,
        p_brand,
        p_type,
        p_size

