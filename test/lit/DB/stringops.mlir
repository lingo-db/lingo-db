// RUN: env LINGODB_EXECUTION_MODE=DEFAULT run-mlir %s | FileCheck %s
// RUN: if [ "$(uname)" = "Linux" ]; then env LINGODB_EXECUTION_MODE=BASELINE run-mlir %s | FileCheck %s; fi

module {
    func.func @main ()  {
        %conststr1 = db.constant ( "str1" ) : !db.string
        %conststr2 = db.constant ( "str2" ) : !db.string
        %conststr3 = db.constant ( "nostr" ) : !db.string
        %pattern = db.constant ( "str%" ) : !db.string
        %pattern2 = db.constant ( "%o%t%" ) : !db.string
        %pattern3 = db.constant ( "%str" ) : !db.string
        %pattern4 = db.constant ( "____" ) : !db.string

        %from = db.constant( 1 ) : i32
        %to = db.constant( 2 ) : i32
        %substr = db.runtime_call "Substring" (%conststr1,%from,%to) : (!db.string,i32,i32) -> (!db.string)
        db.runtime_call "DumpValue" (%substr) : (!db.string) -> ()
		//CHECK: bool(false)
        %0 = db.compare eq %conststr1 : !db.string, %conststr2 : !db.string
        db.runtime_call "DumpValue" (%0) : (i1) -> ()
        //CHECK: bool(true)
        %1 = db.compare eq %conststr1 : !db.string, %conststr1 : !db.string
        db.runtime_call "DumpValue" (%1) : (i1) -> ()
		//CHECK: bool(true)
        %2 = db.compare lt %conststr1 : !db.string, %conststr2 : !db.string
        db.runtime_call "DumpValue" (%2) : (i1) -> ()
        //CHECK: bool(false)
        %3 = db.compare lt %conststr1 : !db.string, %conststr1 : !db.string
        db.runtime_call "DumpValue" (%3) : (i1) -> ()
		//CHECK: bool(true)
        %4 = db.compare lte %conststr1 : !db.string, %conststr2 : !db.string
        db.runtime_call "DumpValue" (%4) : (i1) -> ()
        //CHECK: bool(true)
        %5 = db.compare lte %conststr1 : !db.string, %conststr1 : !db.string
        db.runtime_call "DumpValue" (%5) : (i1) -> ()
        %6 = db.compare gt %conststr2 : !db.string, %conststr1 : !db.string
        db.runtime_call "DumpValue" (%6) : (i1) -> ()
        //CHECK: bool(false)
        %7 = db.compare gt %conststr1 : !db.string, %conststr1 : !db.string
        db.runtime_call "DumpValue" (%7) : (i1) -> ()
		//CHECK: bool(true)
        %8 = db.compare gte %conststr2 : !db.string, %conststr1: !db.string
        db.runtime_call "DumpValue" (%8) : (i1) -> ()
        //CHECK: bool(true)
        %9 = db.compare gte %conststr1 : !db.string, %conststr1 : !db.string
        db.runtime_call "DumpValue" (%9) : (i1) -> ()
		//CHECK: bool(true)
        %10 = db.compare neq %conststr1 : !db.string, %conststr2 : !db.string
        db.runtime_call "DumpValue" (%10) : (i1) -> ()
        //CHECK: bool(false)
        %11 = db.compare neq %conststr1 : !db.string, %conststr1 : !db.string
        db.runtime_call "DumpValue" (%11) : (i1) -> ()
		//CHECK: bool(true)
        %12 = db.runtime_call "Like" (%conststr1,%pattern) : (!db.string,!db.string) -> i1
        db.runtime_call "DumpValue" (%12) : (i1) -> ()
		//CHECK: bool(false)
        %13 = db.runtime_call "Like" (%conststr3,%pattern) : (!db.string,!db.string) -> i1
        db.runtime_call "DumpValue" (%13) : (i1) -> ()
        //CHECK: bool(false)
        %like2 = db.runtime_call "Like" (%conststr1,%pattern2) : (!db.string,!db.string) -> i1
        db.runtime_call "DumpValue" (%like2) : (i1) -> ()
        //CHECK: bool(true)
        %like3 = db.runtime_call "Like" (%conststr3,%pattern2) : (!db.string,!db.string) -> i1
        db.runtime_call "DumpValue" (%like3) : (i1) -> ()
         //CHECK: bool(false)
        %like4 = db.runtime_call "Like" (%conststr1,%pattern3) : (!db.string,!db.string) -> i1
        db.runtime_call "DumpValue" (%like4) : (i1) -> ()
        //CHECK: bool(true)
        %like5 = db.runtime_call "Like" (%conststr3,%pattern3) : (!db.string,!db.string) -> i1
        db.runtime_call "DumpValue" (%like5) : (i1) -> ()
        //CHECK: bool(true)
        %like6 = db.runtime_call "Like" (%conststr1,%pattern4) : (!db.string,!db.string) -> i1
        db.runtime_call "DumpValue" (%like6) : (i1) -> ()
        //CHECK: bool(false)
        %like7 = db.runtime_call "Like" (%conststr3,%pattern4) : (!db.string,!db.string) -> i1
        db.runtime_call "DumpValue" (%like7) : (i1) -> ()
        %intstr= db.constant ("42") : !db.string
        %floatstr= db.constant ("1.00001") : !db.string
        %decimalstr= db.constant ("1.000001") : !db.string
        %int = db.constant (42) : i32
        %float32 = db.constant (1.01) : f32
        %float64 = db.constant (1.0001) : f64
        %decimal = db.constant ("1.0000001") : !db.decimal<10,7>

        //CHECK: int(42)
        %14 = db.cast %intstr : !db.string -> i32
        db.runtime_call "DumpValue" (%14) : (i32) -> ()
        //CHECK: float(1.00001)
        %15 = db.cast %floatstr : !db.string -> f32
        db.runtime_call "DumpValue" (%15) : (f32) -> ()
        //CHECK: float(1.00001)
        %16 = db.cast %floatstr : !db.string -> f64
        db.runtime_call "DumpValue" (%16) : (f64) -> ()
        //CHECK: decimal(1.0000010)
        %17 = db.cast %decimalstr : !db.string -> !db.decimal<10,7>
        db.runtime_call "DumpValue" (%17) : (!db.decimal<10,7>) -> ()
        //CHECK: string("42")
        %18 = db.cast %int : i32 -> !db.string
        db.runtime_call "DumpValue" (%18) : (!db.string) -> ()
        //CHECK: string("1.01")
        %19 = db.cast %float32 : f32 -> !db.string
        db.runtime_call "DumpValue" (%19) : (!db.string) -> ()
        //CHECK: string("1.0001")
        %20 = db.cast %float64 : f64 -> !db.string
        db.runtime_call "DumpValue" (%20) : (!db.string) -> ()
        //CHECK: string("1.0000001")
        %21 = db.cast %decimal : !db.decimal<10,7> -> !db.string
        db.runtime_call "DumpValue" (%21) : (!db.string) -> ()


        // Like:Simple
        %like8 = db.constant("Jos_!") : !db.string

        %c1 = db.constant("Jose!") : !db.string
        %r1 = db.runtime_call "Like"(%c1, %like8) : (!db.string, !db.string) -> (i1)
        db.runtime_call "DumpValue"(%r1) : (i1) -> ()
        //CHECK: bool(true)

        %c2 = db.constant("José!") : !db.string
        %r2 = db.runtime_call "Like"(%c2, %like8) : (!db.string, !db.string) -> (i1)
        db.runtime_call "DumpValue"(%r2) : (i1) -> ()
        //CHECK: bool(true)

        // Like:Combinations

        // Multi-byte Wildcard
        %like9 = db.constant("J_sé!") : !db.string

        %c3 = db.constant("José!") : !db.string
        %r3 = db.runtime_call "Like"(%c3, %like9) : (!db.string, !db.string) -> (i1)
        db.runtime_call "DumpValue"(%r3) : (i1) -> ()
        //CHECK: bool(true)

        %c4 = db.constant("Jéssé!") : !db.string
        %r4 = db.runtime_call "Like"(%c4, %like9) : (!db.string, !db.string) -> (i1)
        db.runtime_call "DumpValue"(%r4) : (i1) -> ()
        //CHECK: bool(false)

        // Matching across boundaries
        %like10 = db.constant("J%!") : !db.string

        %c5 = db.constant("José!") : !db.string
        %r5 = db.runtime_call "Like"(%c5, %like10) : (!db.string, !db.string) -> (i1)
        db.runtime_call "DumpValue"(%r5) : (i1) -> ()
        //CHECK: bool(true)

        %c6 = db.constant("Jośę!") : !db.string
        %r6 = db.runtime_call "Like"(%c6, %like10) : (!db.string, !db.string) -> (i1)
        db.runtime_call "DumpValue"(%r6) : (i1) -> ()
        //CHECK: bool(true)

        %c7 = db.constant("J!") : !db.string
        %r7 = db.runtime_call "Like"(%c7, %like10) : (!db.string, !db.string) -> (i1)
        db.runtime_call "DumpValue"(%r7) : (i1) -> ()
        //CHECK: bool(true)

        // Combining Characters
        %like11 = db.constant("Jo_%!") : !db.string

        %c8 = db.constant("Joé!") : !db.string
        %r8 = db.runtime_call "Like"(%c8, %like11) : (!db.string, !db.string) -> (i1)
        db.runtime_call "DumpValue"(%r8) : (i1) -> ()
        //CHECK: bool(true)

        %c9 = db.constant("Joé!") : !db.string
        %r9 = db.runtime_call "Like"(%c9, %like11) : (!db.string, !db.string) -> (i1)
        db.runtime_call "DumpValue"(%r9) : (i1) -> ()
        //CHECK: bool(true)

        %c10 = db.constant("Joe!") : !db.string
        %r10 = db.runtime_call "Like"(%c10, %like11) : (!db.string, !db.string) -> (i1)
        db.runtime_call "DumpValue"(%r10) : (i1) -> ()
        //CHECK: bool(true)

        // UTF-8 Boundary with %
        %like12 = db.constant("%é!") : !db.string

        %c11 = db.constant("José!") : !db.string
        %r11 = db.runtime_call "Like"(%c11, %like12) : (!db.string, !db.string) -> (i1)
        db.runtime_call "DumpValue"(%r11) : (i1) -> ()
        //CHECK: bool(true)

        %c12 = db.constant("Jøse!") : !db.string
        %r12 = db.runtime_call "Like"(%c12, %like12) : (!db.string, !db.string) -> (i1)
        db.runtime_call "DumpValue"(%r12) : (i1) -> ()
        //CHECK: bool(false)

        // Multiple _
        %like13 = db.constant("____!") : !db.string

        %c13 = db.constant("José!") : !db.string
        %r13 = db.runtime_call "Like"(%c13, %like13) : (!db.string, !db.string) -> (i1)
        db.runtime_call "DumpValue"(%r13) : (i1) -> ()
        //CHECK: bool(true)

        %c14 = db.constant("Jose!") : !db.string
        %r14 = db.runtime_call "Like"(%c14, %like13) : (!db.string, !db.string) -> (i1)
        db.runtime_call "DumpValue"(%r14) : (i1) -> ()
        //CHECK: bool(true)

        %c15 = db.constant("Jósé!") : !db.string
        %r15 = db.runtime_call "Like"(%c15, %like13) : (!db.string, !db.string) -> (i1)
        db.runtime_call "DumpValue"(%r15) : (i1) -> ()
        //CHECK: bool(true)

        // Start and End Anchors
        %like14 = db.constant("%é") : !db.string

        %c16 = db.constant("Café") : !db.string
        %r16 = db.runtime_call "Like"(%c16, %like14) : (!db.string, !db.string) -> (i1)
        db.runtime_call "DumpValue"(%r16) : (i1) -> ()
        //CHECK: bool(true)

        %c17 = db.constant("Cafe") : !db.string
        %r17 = db.runtime_call "Like"(%c17, %like14) : (!db.string, !db.string) -> (i1)
        db.runtime_call "DumpValue"(%r17) : (i1) -> ()
        //CHECK: bool(false)

        // Non-Match Edge Cases
        %like15 = db.constant("Jo%!"): !db.string

        %c18 = db.constant("Jóse") : !db.string
        %r18 = db.runtime_call "Like"(%c18, %like15) : (!db.string, !db.string) -> (i1)
        db.runtime_call "DumpValue"(%r18) : (i1) -> ()
        //CHECK: bool(false)

        %c19 = db.constant("Jośé!!") : !db.string
        %r19 = db.runtime_call "Like"(%c19, %like15) : (!db.string, !db.string) -> (i1)
        db.runtime_call "DumpValue"(%r19) : (i1) -> ()
        //CHECK: bool(true)

        %like16 = db.constant("%Jos"): !db.string
        %c20 = db.constant("Jose!!") : !db.string
        %r20 = db.runtime_call "Like"(%c20, %like16) : (!db.string, !db.string) -> (i1)
        db.runtime_call "DumpValue"(%r20) : (i1) -> ()
        //CHECK: bool(false)

        %like17 = db.constant("a_b%abc%%_aáé"): !db.string
        %c21 = db.constant("aabxxabcxaáéxaáé") : !db.string
        %r21 = db.runtime_call "Like"(%c21, %like17) : (!db.string, !db.string) -> (i1)
        db.runtime_call "DumpValue"(%r21) : (i1) -> ()
        //CHECK: bool(true)

        %like18 = db.constant("a_b%%%_aáé"): !db.string
        %c22 = db.constant("aabxxxxaáéáé") : !db.string
        %r22 = db.runtime_call "Like"(%c22, %like18) : (!db.string, !db.string) -> (i1)
        db.runtime_call "DumpValue"(%r22) : (i1) -> ()
        //CHECK: bool(false)

        %like19 = db.constant("a_b%abc%bc"): !db.string
        %c23 = db.constant("aababc") : !db.string
        %r23 = db.runtime_call "Like"(%c23, %like19) : (!db.string, !db.string) -> (i1)
        db.runtime_call "DumpValue"(%r23) : (i1) -> ()
        //CHECK: bool(false)

        %like20 = db.constant("a_b%abc%_bc"): !db.string
        %c24 = db.constant("aababcabcbbc") : !db.string
        %r24 = db.runtime_call "Like"(%c24, %like20) : (!db.string, !db.string) -> (i1)
        db.runtime_call "DumpValue"(%r24) : (i1) -> ()
        //CHECK: bool(true)

        %like21 = db.constant("a_b%__%abc%"): !db.string
        %c25 = db.constant("abbabcaa") : !db.string
        %r25 = db.runtime_call "Like"(%c25, %like21) : (!db.string, !db.string) -> (i1)
        db.runtime_call "DumpValue"(%r25) : (i1) -> ()
        //CHECK: bool(false)
        return
    }
}
