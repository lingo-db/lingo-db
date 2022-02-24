module @querymodule{
    func  @main ()  -> !db.table{
        %1 = relalg.basetable @lineitem { table_identifier="lineitem", rows=600572 , pkey=["l_orderkey","l_linenumber"]} columns: {l_orderkey => @l_orderkey({type=!db.int<32>}),
            l_partkey => @l_partkey({type=!db.int<32>}),
            l_suppkey => @l_suppkey({type=!db.int<32>}),
            l_linenumber => @l_linenumber({type=!db.int<32>}),
            l_quantity => @l_quantity({type=!db.decimal<15,2>}),
            l_extendedprice => @l_extendedprice({type=!db.decimal<15,2>}),
            l_discount => @l_discount({type=!db.decimal<15,2>}),
            l_tax => @l_tax({type=!db.decimal<15,2>}),
            l_returnflag => @l_returnflag({type=!db.char<1>}),
            l_linestatus => @l_linestatus({type=!db.char<1>}),
            l_shipdate => @l_shipdate({type=!db.date<day>}),
            l_commitdate => @l_commitdate({type=!db.date<day>}),
            l_receiptdate => @l_receiptdate({type=!db.date<day>}),
            l_shipinstruct => @l_shipinstruct({type=!db.string}),
            l_shipmode => @l_shipmode({type=!db.string}),
            l_comment => @l_comment({type=!db.string})
        }
        %2 = relalg.basetable @part { table_identifier="part", rows=20000 , pkey=["p_partkey"]} columns: {p_partkey => @p_partkey({type=!db.int<32>}),
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
            %6 = relalg.getattr %4 @part::@p_partkey : !db.int<32>
            %7 = relalg.getattr %4 @lineitem::@l_partkey : !db.int<32>
            %8 = db.compare eq %6 : !db.int<32>,%7 : !db.int<32>
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
            %21 = db.or %14 : i1,%16 : i1,%18 : i1,%20 : i1
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
            %38 = db.and %35 : i1,%37 : i1
            %39 = relalg.getattr %4 @lineitem::@l_shipmode : !db.string
            %40 = db.constant ("AIR") :!db.string
            %41 = db.compare eq %39 : !db.string,%40 : !db.string
            %42 = db.constant ("AIR REG") :!db.string
            %43 = db.compare eq %39 : !db.string,%42 : !db.string
            %44 = db.or %41 : i1,%43 : i1
            %45 = relalg.getattr %4 @lineitem::@l_shipinstruct : !db.string
            %46 = db.constant ("DELIVER IN PERSON") :!db.string
            %47 = db.compare eq %45 : !db.string,%46 : !db.string
            %48 = db.and %8 : i1,%11 : i1,%21 : i1,%24 : i1,%30 : i1,%38 : i1,%44 : i1,%47 : i1
            %49 = relalg.getattr %4 @part::@p_partkey : !db.int<32>
            %50 = relalg.getattr %4 @lineitem::@l_partkey : !db.int<32>
            %51 = db.compare eq %49 : !db.int<32>,%50 : !db.int<32>
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
            %64 = db.or %57 : i1,%59 : i1,%61 : i1,%63 : i1
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
            %81 = db.and %78 : i1,%80 : i1
            %82 = relalg.getattr %4 @lineitem::@l_shipmode : !db.string
            %83 = db.constant ("AIR") :!db.string
            %84 = db.compare eq %82 : !db.string,%83 : !db.string
            %85 = db.constant ("AIR REG") :!db.string
            %86 = db.compare eq %82 : !db.string,%85 : !db.string
            %87 = db.or %84 : i1,%86 : i1
            %88 = relalg.getattr %4 @lineitem::@l_shipinstruct : !db.string
            %89 = db.constant ("DELIVER IN PERSON") :!db.string
            %90 = db.compare eq %88 : !db.string,%89 : !db.string
            %91 = db.and %51 : i1,%54 : i1,%64 : i1,%67 : i1,%73 : i1,%81 : i1,%87 : i1,%90 : i1
            %92 = relalg.getattr %4 @part::@p_partkey : !db.int<32>
            %93 = relalg.getattr %4 @lineitem::@l_partkey : !db.int<32>
            %94 = db.compare eq %92 : !db.int<32>,%93 : !db.int<32>
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
            %107 = db.or %100 : i1,%102 : i1,%104 : i1,%106 : i1
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
            %124 = db.and %121 : i1,%123 : i1
            %125 = relalg.getattr %4 @lineitem::@l_shipmode : !db.string
            %126 = db.constant ("AIR") :!db.string
            %127 = db.compare eq %125 : !db.string,%126 : !db.string
            %128 = db.constant ("AIR REG") :!db.string
            %129 = db.compare eq %125 : !db.string,%128 : !db.string
            %130 = db.or %127 : i1,%129 : i1
            %131 = relalg.getattr %4 @lineitem::@l_shipinstruct : !db.string
            %132 = db.constant ("DELIVER IN PERSON") :!db.string
            %133 = db.compare eq %131 : !db.string,%132 : !db.string
            %134 = db.and %94 : i1,%97 : i1,%107 : i1,%110 : i1,%116 : i1,%124 : i1,%130 : i1,%133 : i1
            %135 = db.or %48 : i1,%91 : i1,%134 : i1
            relalg.return %135 : i1
        }
        %137 = relalg.map @map %5 (%136: !relalg.tuple) {
            %138 = relalg.getattr %136 @lineitem::@l_extendedprice : !db.decimal<15,2>
            %139 = db.constant (1) :!db.decimal<15,2>
            %140 = relalg.getattr %136 @lineitem::@l_discount : !db.decimal<15,2>
            %141 = db.sub %139 : !db.decimal<15,2>,%140 : !db.decimal<15,2>
            %142 = db.mul %138 : !db.decimal<15,2>,%141 : !db.decimal<15,2>
            %143 = relalg.addattr %136, @aggfmname1({type=!db.decimal<15,2>}) %142
            relalg.return %143 : !relalg.tuple
        }
        %146 = relalg.aggregation @aggr %137 [] (%144 : !relalg.tuplestream, %145 : !relalg.tuple) {
            %147 = relalg.aggrfn sum @map::@aggfmname1 %144 : !db.nullable<!db.decimal<15,2>>
            %148 = relalg.addattr %145, @aggfmname2({type=!db.nullable<!db.decimal<15,2>>}) %147
            relalg.return %148 : !relalg.tuple
        }
        %149 = relalg.materialize %146 [@aggr::@aggfmname2] => ["revenue"] : !db.table
        return %149 : !db.table
    }
}

