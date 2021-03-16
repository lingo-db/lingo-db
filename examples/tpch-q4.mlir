module @tpchq4  {
  func @query() {
    %0 = relalg.basetable @orders { table_identifier = "orders" } columns:  {
    	o_orderkey => @o_orderkey ({ name="o_orderkey" , type=!db.int<64,nullable>}),
    	o_orderdate => @o_orderdate ({ name="o_orderdate" , type=!db.date<nullable>}),
    	o_orderpriority => @o_orderpriority ({ name="o_orderpriority" , type=!db.string<nullable> })
    }

    %1 = relalg.basetable @lineitem { table_identifier= "lineitem" } columns:  {
    	l_orderkey => @l_orderkey ({ name="l_orderkey" , type=!db.int<64> }),
    	l_commitdate => @l_commitdate ({ name="l_commitdate" , type=!db.date<nullable>}),
    	l_receiptdate => @l_receiptdate ({ name="l_receiptdate" , type=!db.date<nullable>})
    }

    %2 = relalg.selection %0 (%arg0: !relalg.tuple){
        %3 = relalg.getattr %arg0 @orders::@o_orderdate :!db.date<nullable>
        %21 = relalg.getattr %arg0 @orders::@o_orderkey : !db.int<64,nullable>
        %4 = db.constant("1993-07-01") : !db.date<nullable>
        %5 = db.constant("1993-10-01") : !db.date<nullable>
        %6 = db.compare "sge" %3 : !db.date<nullable> , %4 : !db.date<nullable> !db.bool<nullable>
        %7 = db.compare "slt" %3 : !db.date<nullable> , %5 : !db.date<nullable> !db.bool<nullable>
        %8 = relalg.selection %1 (%arg1: !relalg.tuple){
            %9 = relalg.getattr %arg1 @lineitem::@l_orderkey  : !db.int<64,nullable>
            %10 = db.compare "eq" %21 : !db.int<64,nullable>, %9  : !db.int<64,nullable> !db.bool<nullable>
            %11 = relalg.getattr %arg1 @orders::@o_orderkey  : !db.int<64,nullable>
            %12 = relalg.getattr %arg1 @lineitem::@l_orderkey : !db.int<64,nullable>
            %13 = db.compare "slt" %11 : !db.int<64,nullable>, %12 : !db.int<64,nullable> !db.bool<nullable>
            %14 = db.and %10 : !db.bool<nullable> , %13 : !db.bool<nullable>
            relalg.return %14 : !db.bool<nullable>
        }
        %15 = relalg.exists %8
        %16 = db.and %6 :!db.bool<nullable>, %7 :!db.bool<nullable>, %15:!db.bool
        relalg.return %16:!db.bool<nullable>
    }
    %17 = relalg.aggregation @agg1 %2 [@orders::@o_orderpriority] (%arg2: !relalg.relation){
        %18 = relalg.count %arg2
        relalg.addattr @order_count({type=!db.int<64>,name=""}) %18
        relalg.return
    }
    //%19 = relalg.sort %17 [@orders::@o_orderpriority]
    //%20 = relalg.materialize %17 [@orders::@o_orderpriority, @agg1::@order_count]
    return
  }
}