module @querymodule{
    func @main (%executionContext: !util.generic_memref<i8>)  -> !db.table{
        %1 = relalg.basetable @lineitem { table_identifier="lineitem", rows=600572 , pkey=["l_orderkey","l_linenumber"]} columns: {l_orderkey => @l_orderkey({type=!db.int<64>}),
            l_partkey => @l_partkey({type=!db.int<64>}),
            l_suppkey => @l_suppkey({type=!db.int<64>}),
            l_linenumber => @l_linenumber({type=!db.int<32>}),
            l_quantity => @l_quantity({type=!db.decimal<15,2>}),
            l_extendedprice => @l_extendedprice({type=!db.decimal<15,2>}),
            l_discount => @l_discount({type=!db.decimal<15,2>}),
            l_tax => @l_tax({type=!db.decimal<15,2>}),
            l_returnflag => @l_returnflag({type=!db.string}),
            l_linestatus => @l_linestatus({type=!db.string}),
            l_shipdate => @l_shipdate({type=!db.date<day>}),
            l_commitdate => @l_commitdate({type=!db.date<day>}),
            l_receiptdate => @l_receiptdate({type=!db.date<day>}),
            l_shipinstruct => @l_shipinstruct({type=!db.string}),
            l_shipmode => @l_shipmode({type=!db.string}),
            l_comment => @l_comment({type=!db.string})
        }
        %2 = relalg.basetable @part { table_identifier="part", rows=20000 , pkey=["p_partkey"]} columns: {p_partkey => @p_partkey({type=!db.int<64>}),
            p_name => @p_name({type=!db.string}),
            p_mfgr => @p_mfgr({type=!db.string}),
            p_brand => @p_brand({type=!db.string}),
            p_type => @p_type({type=!db.string}),
            p_size => @p_size({type=!db.int<32>}),
            p_container => @p_container({type=!db.string}),
            p_retailprice => @p_retailprice({type=!db.decimal<15,2>}),
            p_comment => @p_comment({type=!db.string})
        }
        %3 = relalg.crossproduct %1, %2
        %5 = relalg.selection %3(%4: !relalg.tuple) {
            %6 = relalg.getattr %4 @part::@p_partkey : !db.int<64>
            %7 = relalg.getattr %4 @lineitem::@l_partkey : !db.int<64>
            %8 = db.compare eq %6 : !db.int<64>,%7 : !db.int<64>
            %9 = relalg.getattr %4 @part::@p_brand : !db.string
            %10 = db.constant ("Brand#12") :!db.string
            %11 = db.compare eq %9 : !db.string,%10 : !db.string
            %12 = relalg.getattr %4 @part::@p_container : !db.string
            %13 = db.constant ("SM CASE") :!db.string
            %14 = db.compare eq %12 : !db.string,%13 : !db.string
            %15 = db.constant ("SM BOX") :!db.string
            %16 = db.compare eq %12 : !db.string,%15 : !db.string
            %17 = db.constant ("SM PACK") :!db.string
            %18 = db.compare eq %12 : !db.string,%17 : !db.string
            %19 = db.constant ("SM PKG") :!db.string
            %20 = db.compare eq %12 : !db.string,%19 : !db.string
            %21 = db.or %14 : !db.bool,%16 : !db.bool,%18 : !db.bool,%20 : !db.bool
            %22 = relalg.getattr %4 @lineitem::@l_quantity : !db.decimal<15,2>
            %23 = db.constant (1) :!db.decimal<15,2>
            %24 = db.compare gte %22 : !db.decimal<15,2>,%23 : !db.decimal<15,2>
            %25 = relalg.getattr %4 @lineitem::@l_quantity : !db.decimal<15,2>
            %26 = db.constant (1) :!db.int<64>
            %27 = db.constant (10) :!db.int<64>
            %28 = db.add %26 : !db.int<64>,%27 : !db.int<64>
            %29 = db.cast %28 : !db.int<64> -> !db.decimal<15,2>
            %30 = db.compare lte %25 : !db.decimal<15,2>,%29 : !db.decimal<15,2>
            %31 = relalg.getattr %4 @part::@p_size : !db.int<32>
            %32 = db.constant (1) :!db.int<64>
            %33 = db.constant (5) :!db.int<64>
            %34 = db.cast %31 : !db.int<32> -> !db.int<64>
            %35 = db.compare gte %34 : !db.int<64>,%32 : !db.int<64>
            %36 = db.cast %31 : !db.int<32> -> !db.int<64>
            %37 = db.compare lte %36 : !db.int<64>,%33 : !db.int<64>
            %38 = db.and %35 : !db.bool,%37 : !db.bool
            %39 = relalg.getattr %4 @lineitem::@l_shipmode : !db.string
            %40 = db.constant ("AIR") :!db.string
            %41 = db.compare eq %39 : !db.string,%40 : !db.string
            %42 = db.constant ("AIR REG") :!db.string
            %43 = db.compare eq %39 : !db.string,%42 : !db.string
            %44 = db.or %41 : !db.bool,%43 : !db.bool
            %45 = relalg.getattr %4 @lineitem::@l_shipinstruct : !db.string
            %46 = db.constant ("DELIVER IN PERSON") :!db.string
            %47 = db.compare eq %45 : !db.string,%46 : !db.string
            %48 = db.and %8 : !db.bool,%11 : !db.bool,%21 : !db.bool,%24 : !db.bool,%30 : !db.bool,%38 : !db.bool,%44 : !db.bool,%47 : !db.bool
            %49 = relalg.getattr %4 @part::@p_partkey : !db.int<64>
            %50 = relalg.getattr %4 @lineitem::@l_partkey : !db.int<64>
            %51 = db.compare eq %49 : !db.int<64>,%50 : !db.int<64>
            %52 = relalg.getattr %4 @part::@p_brand : !db.string
            %53 = db.constant ("Brand#23") :!db.string
            %54 = db.compare eq %52 : !db.string,%53 : !db.string
            %55 = relalg.getattr %4 @part::@p_container : !db.string
            %56 = db.constant ("MED BAG") :!db.string
            %57 = db.compare eq %55 : !db.string,%56 : !db.string
            %58 = db.constant ("MED BOX") :!db.string
            %59 = db.compare eq %55 : !db.string,%58 : !db.string
            %60 = db.constant ("MED PKG") :!db.string
            %61 = db.compare eq %55 : !db.string,%60 : !db.string
            %62 = db.constant ("MED PACK") :!db.string
            %63 = db.compare eq %55 : !db.string,%62 : !db.string
            %64 = db.or %57 : !db.bool,%59 : !db.bool,%61 : !db.bool,%63 : !db.bool
            %65 = relalg.getattr %4 @lineitem::@l_quantity : !db.decimal<15,2>
            %66 = db.constant (10) :!db.decimal<15,2>
            %67 = db.compare gte %65 : !db.decimal<15,2>,%66 : !db.decimal<15,2>
            %68 = relalg.getattr %4 @lineitem::@l_quantity : !db.decimal<15,2>
            %69 = db.constant (10) :!db.int<64>
            %70 = db.constant (10) :!db.int<64>
            %71 = db.add %69 : !db.int<64>,%70 : !db.int<64>
            %72 = db.cast %71 : !db.int<64> -> !db.decimal<15,2>
            %73 = db.compare lte %68 : !db.decimal<15,2>,%72 : !db.decimal<15,2>
            %74 = relalg.getattr %4 @part::@p_size : !db.int<32>
            %75 = db.constant (1) :!db.int<64>
            %76 = db.constant (10) :!db.int<64>
            %77 = db.cast %74 : !db.int<32> -> !db.int<64>
            %78 = db.compare gte %77 : !db.int<64>,%75 : !db.int<64>
            %79 = db.cast %74 : !db.int<32> -> !db.int<64>
            %80 = db.compare lte %79 : !db.int<64>,%76 : !db.int<64>
            %81 = db.and %78 : !db.bool,%80 : !db.bool
            %82 = relalg.getattr %4 @lineitem::@l_shipmode : !db.string
            %83 = db.constant ("AIR") :!db.string
            %84 = db.compare eq %82 : !db.string,%83 : !db.string
            %85 = db.constant ("AIR REG") :!db.string
            %86 = db.compare eq %82 : !db.string,%85 : !db.string
            %87 = db.or %84 : !db.bool,%86 : !db.bool
            %88 = relalg.getattr %4 @lineitem::@l_shipinstruct : !db.string
            %89 = db.constant ("DELIVER IN PERSON") :!db.string
            %90 = db.compare eq %88 : !db.string,%89 : !db.string
            %91 = db.and %51 : !db.bool,%54 : !db.bool,%64 : !db.bool,%67 : !db.bool,%73 : !db.bool,%81 : !db.bool,%87 : !db.bool,%90 : !db.bool
            %92 = relalg.getattr %4 @part::@p_partkey : !db.int<64>
            %93 = relalg.getattr %4 @lineitem::@l_partkey : !db.int<64>
            %94 = db.compare eq %92 : !db.int<64>,%93 : !db.int<64>
            %95 = relalg.getattr %4 @part::@p_brand : !db.string
            %96 = db.constant ("Brand#34") :!db.string
            %97 = db.compare eq %95 : !db.string,%96 : !db.string
            %98 = relalg.getattr %4 @part::@p_container : !db.string
            %99 = db.constant ("LG CASE") :!db.string
            %100 = db.compare eq %98 : !db.string,%99 : !db.string
            %101 = db.constant ("LG BOX") :!db.string
            %102 = db.compare eq %98 : !db.string,%101 : !db.string
            %103 = db.constant ("LG PACK") :!db.string
            %104 = db.compare eq %98 : !db.string,%103 : !db.string
            %105 = db.constant ("LG PKG") :!db.string
            %106 = db.compare eq %98 : !db.string,%105 : !db.string
            %107 = db.or %100 : !db.bool,%102 : !db.bool,%104 : !db.bool,%106 : !db.bool
            %108 = relalg.getattr %4 @lineitem::@l_quantity : !db.decimal<15,2>
            %109 = db.constant (20) :!db.decimal<15,2>
            %110 = db.compare gte %108 : !db.decimal<15,2>,%109 : !db.decimal<15,2>
            %111 = relalg.getattr %4 @lineitem::@l_quantity : !db.decimal<15,2>
            %112 = db.constant (20) :!db.int<64>
            %113 = db.constant (10) :!db.int<64>
            %114 = db.add %112 : !db.int<64>,%113 : !db.int<64>
            %115 = db.cast %114 : !db.int<64> -> !db.decimal<15,2>
            %116 = db.compare lte %111 : !db.decimal<15,2>,%115 : !db.decimal<15,2>
            %117 = relalg.getattr %4 @part::@p_size : !db.int<32>
            %118 = db.constant (1) :!db.int<64>
            %119 = db.constant (15) :!db.int<64>
            %120 = db.cast %117 : !db.int<32> -> !db.int<64>
            %121 = db.compare gte %120 : !db.int<64>,%118 : !db.int<64>
            %122 = db.cast %117 : !db.int<32> -> !db.int<64>
            %123 = db.compare lte %122 : !db.int<64>,%119 : !db.int<64>
            %124 = db.and %121 : !db.bool,%123 : !db.bool
            %125 = relalg.getattr %4 @lineitem::@l_shipmode : !db.string
            %126 = db.constant ("AIR") :!db.string
            %127 = db.compare eq %125 : !db.string,%126 : !db.string
            %128 = db.constant ("AIR REG") :!db.string
            %129 = db.compare eq %125 : !db.string,%128 : !db.string
            %130 = db.or %127 : !db.bool,%129 : !db.bool
            %131 = relalg.getattr %4 @lineitem::@l_shipinstruct : !db.string
            %132 = db.constant ("DELIVER IN PERSON") :!db.string
            %133 = db.compare eq %131 : !db.string,%132 : !db.string
            %134 = db.and %94 : !db.bool,%97 : !db.bool,%107 : !db.bool,%110 : !db.bool,%116 : !db.bool,%124 : !db.bool,%130 : !db.bool,%133 : !db.bool
            %135 = db.or %48 : !db.bool,%91 : !db.bool,%134 : !db.bool
            relalg.return %135 : !db.bool
        }
        %137 = relalg.map @map1 %5 (%136: !relalg.tuple) {
            %138 = relalg.getattr %136 @lineitem::@l_extendedprice : !db.decimal<15,2>
            %139 = db.constant (1) :!db.decimal<15,2>
            %140 = relalg.getattr %136 @lineitem::@l_discount : !db.decimal<15,2>
            %141 = db.sub %139 : !db.decimal<15,2>,%140 : !db.decimal<15,2>
            %142 = db.mul %138 : !db.decimal<15,2>,%141 : !db.decimal<15,2>
            relalg.addattr @aggfmname1({type=!db.decimal<15,2>}) %142
            relalg.return
        }
        %144 = relalg.aggregation @aggr1 %137 [] (%143 : !relalg.relation) {
            %145 = relalg.aggrfn sum @map1::@aggfmname1 %143 : !db.decimal<15,2,nullable>
            relalg.addattr @aggfmname2({type=!db.decimal<15,2,nullable>}) %145
            relalg.return
        }
        %146 = relalg.materialize %144 [@aggr1::@aggfmname2] => ["revenue"] : !db.table
        return %146 : !db.table
    }
}

