// RUN: db-run %s | FileCheck %s
module {
    func @main ()  {
        %conststr1 = db.constant ( "str1" ) : !db.string
        %conststr2 = db.constant ( "str2" ) : !db.string
        %conststr3 = db.constant ( "nostr" ) : !db.string
        %pattern = db.constant ( "str%" ) : !db.string
        %substr = db.substr %conststr1[1 : 2] : !db.string
        db.dump %substr : !db.string
		//CHECK: bool(false)
        %0 = db.compare eq %conststr1 : !db.string, %conststr2 : !db.string
        db.dump %0 : i1
        //CHECK: bool(true)
        %1 = db.compare eq %conststr1 : !db.string, %conststr1 : !db.string
        db.dump %1 : i1
		//CHECK: bool(true)
        %2 = db.compare lt %conststr1 : !db.string, %conststr2 : !db.string
        db.dump %2 : i1
        //CHECK: bool(false)
        %3 = db.compare lt %conststr1 : !db.string, %conststr1 : !db.string
        db.dump %3 : i1
		//CHECK: bool(true)
        %4 = db.compare lte %conststr1 : !db.string, %conststr2 : !db.string
        db.dump %4 : i1
        //CHECK: bool(true)
        %5 = db.compare lte %conststr1 : !db.string, %conststr1 : !db.string
        db.dump %5 : i1
        %6 = db.compare gt %conststr2 : !db.string, %conststr1 : !db.string
        db.dump %6 : i1
        //CHECK: bool(false)
        %7 = db.compare gt %conststr1 : !db.string, %conststr1 : !db.string
        db.dump %7 : i1
		//CHECK: bool(true)
        %8 = db.compare gte %conststr2 : !db.string, %conststr1: !db.string
        db.dump %8 : i1
        //CHECK: bool(true)
        %9 = db.compare gte %conststr1 : !db.string, %conststr1 : !db.string
        db.dump %9 : i1
		//CHECK: bool(true)
        %10 = db.compare neq %conststr1 : !db.string, %conststr2 : !db.string
        db.dump %10 : i1
        //CHECK: bool(false)
        %11 = db.compare neq %conststr1 : !db.string, %conststr1 : !db.string
        db.dump %11 : i1
		//CHECK: bool(true)
        %12 = db.compare like %conststr1 : !db.string, %pattern : !db.string
        db.dump %12 : i1
		//CHECK: bool(false)
        %13 = db.compare like %conststr3 : !db.string, %pattern : !db.string
        db.dump %13 : i1


        %intstr= db.constant ("42") : !db.string
        %floatstr= db.constant ("1.00001") : !db.string
        %decimalstr= db.constant ("1.000001") : !db.string
        %int = db.constant (42) : i32
        %float32 = db.constant (1.01) : f32
        %float64 = db.constant (1.0001) : f64
        %decimal = db.constant ("1.0000001") : !db.decimal<10,7>

        //CHECK: int(42)
        %14 = db.cast %intstr : !db.string -> i32
        db.dump %14 : i32
        //CHECK: float(1.00001)
        %15 = db.cast %floatstr : !db.string -> f32
        db.dump %15 : f32
        //CHECK: float(1.00001)
        %16 = db.cast %floatstr : !db.string -> f64
        db.dump %16 : f64
        //CHECK: decimal(1.0000010)
        %17 = db.cast %decimalstr : !db.string -> !db.decimal<10,7>
        db.dump %17 :  !db.decimal<10,7>
        //CHECK: string("42")
        %18 = db.cast %int : i32 -> !db.string
        db.dump %18 :  !db.string
        //CHECK: string("1.01")
        %19 = db.cast %float32 : f32 -> !db.string
        db.dump %19 :  !db.string
        //CHECK: string("1.0001")
        %20 = db.cast %float64 : f64 -> !db.string
        db.dump %20 :  !db.string
        //CHECK: string("1.0000001")
        %21 = db.cast %decimal : !db.decimal<10,7> -> !db.string
        db.dump %21 :  !db.string
        return
    }
}