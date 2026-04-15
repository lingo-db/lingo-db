-- TPC-H Query 4

select
    o_orderdate,
    count(*) as order_count
from
    orders
where
    o_orderdate >= date '1993-07-01'
  and o_orderdate < date '1993-10-01'
  and exists (
    select
        1
    from
        lineitem
    where
        l_orderkey = o_orderkey
      and l_commitdate < l_receiptdate
)
group by
    o_orderdate
order by
    o_orderdate
