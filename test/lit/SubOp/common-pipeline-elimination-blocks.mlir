//RUN: mlir-db-opt %s -subop-common-pipeline-elimination | FileCheck %s

// subop-common-pipeline-elimination merges identical "scan table -> insert
// into a fresh multimap" pipelines (hash-join build sides). It may only do so
// within one block: a pipeline inside a loop of a map lambda (a nested query
// evaluated in place, e.g. nested SQL in a hipy UDF loop) runs once per
// iteration and must keep its own multimap (it used to be merged with the
// top-level one, so the top-level lookup used a value defined in the loop:
// "operand #1 does not dominate this use"). Identical pipelines in the same
// block are still merged (control).

//CHECK: %[[TOP:.*]] = subop.create !subop.multimap
//CHECK-NOT: subop.create !subop.multimap
//CHECK: subop.lookup %{{.*}}%[[TOP]]
//CHECK: subop.lookup %{{.*}}%[[TOP]]
//CHECK: scf.for
//CHECK: %[[INNER:.*]] = subop.create !subop.multimap
//CHECK: subop.insert %{{.*}}%[[INNER]]
//CHECK: subop.lookup %{{.*}}%[[INNER]]
module {
  func.func @main() {
    %keys, %keys_streams = subop.generate [@k::@key({type = i64})] {
      %c1 = arith.constant 1 : i64
      subop.generate_emit %c1 : i64
      tuples.return
    }
    %a_t = subop.get_external "0000FEFF0000040000000000000070617274000001000200000000000000FEFF00000400000000000000706B2430000001000200000000000000706B0100FFFFFEFF00000400000000000000737A2430000001000200000000000000737A0100FFFF010002000200000000000000FEFF00000200000000000000706B00000100000000000000000001000200060200030001000000000000000300040000000000000000000400050000000000000000000500060000000000000000000600FFFFFEFF00000200000000000000706B00000100000000000000000001000200060200030001000000000000000300040000000000000000000400050000000000000000000500060000000000000000000600FFFF0200030000000000000000000300040000000000000000000400FFFF0000" : !subop.table<[pk$0 : i64, sz$0 : !db.nullable<i64>], true>
    %a_s = subop.scan %a_t : !subop.table<[pk$0 : i64, sz$0 : !db.nullable<i64>], true> {pk$0 => @pa::@pk({type = i64}), sz$0 => @pa::@sz({type = !db.nullable<i64>})}
    %mma = subop.create !subop.multimap<[member$1 : i64], [member$2 : !db.nullable<i64>]>
    subop.insert %a_s%mma : !subop.multimap<[member$1 : i64], [member$2 : !db.nullable<i64>]> {@pa::@sz => member$2, @pa::@pk => member$1} eq: ([%aa], [%ba]) {
      %eq = db.compare eq %aa : i64, %ba : i64
      tuples.return %eq : i1
    }
    %a_l = subop.lookup %keys%mma [@k::@key] : !subop.multimap<[member$1 : i64], [member$2 : !db.nullable<i64>]> @pa_l::@list({type = !subop.list<!subop.multi_map_entry_ref<<[member$1 : i64], [member$2 : !db.nullable<i64>]>>>}) eq: ([%ca], [%da]) {
      %eq = db.compare eq %ca : i64, %da : i64
      tuples.return %eq : i1
    }
    %b_t = subop.get_external "0000FEFF0000040000000000000070617274000001000200000000000000FEFF00000400000000000000706B2430000001000200000000000000706B0100FFFFFEFF00000400000000000000737A2430000001000200000000000000737A0100FFFF010002000200000000000000FEFF00000200000000000000706B00000100000000000000000001000200060200030001000000000000000300040000000000000000000400050000000000000000000500060000000000000000000600FFFFFEFF00000200000000000000706B00000100000000000000000001000200060200030001000000000000000300040000000000000000000400050000000000000000000500060000000000000000000600FFFF0200030000000000000000000300040000000000000000000400FFFF0000" : !subop.table<[pk$0 : i64, sz$0 : !db.nullable<i64>], true>
    %b_s = subop.scan %b_t : !subop.table<[pk$0 : i64, sz$0 : !db.nullable<i64>], true> {pk$0 => @pb::@pk({type = i64}), sz$0 => @pb::@sz({type = !db.nullable<i64>})}
    %mmb = subop.create !subop.multimap<[member$3 : i64], [member$4 : !db.nullable<i64>]>
    subop.insert %b_s%mmb : !subop.multimap<[member$3 : i64], [member$4 : !db.nullable<i64>]> {@pb::@sz => member$4, @pb::@pk => member$3} eq: ([%ab], [%bb]) {
      %eq = db.compare eq %ab : i64, %bb : i64
      tuples.return %eq : i1
    }
    %b_l = subop.lookup %keys%mmb [@k::@key] : !subop.multimap<[member$3 : i64], [member$4 : !db.nullable<i64>]> @pb_l::@list({type = !subop.list<!subop.multi_map_entry_ref<<[member$3 : i64], [member$4 : !db.nullable<i64>]>>>}) eq: ([%cb], [%db]) {
      %eq = db.compare eq %cb : i64, %db : i64
      tuples.return %eq : i1
    }
    %mapped = subop.map %keys computes : [@m::@r({type = i64})] input : [@k::@key] (%n: i64) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      scf.for %i = %c0 to %n step %c1 : i64 {
        %ikeys, %ikeys_streams = subop.generate [@k::@key({type = i64})] {
          subop.generate_emit %i : i64
          tuples.return
        }
        %c_t = subop.get_external "0000FEFF0000040000000000000070617274000001000200000000000000FEFF00000400000000000000706B2431000001000200000000000000706B0100FFFFFEFF00000400000000000000737A2431000001000200000000000000737A0100FFFF010002000200000000000000FEFF00000200000000000000706B00000100000000000000000001000200060200030001000000000000000300040000000000000000000400050000000000000000000500060000000000000000000600FFFFFEFF00000200000000000000706B00000100000000000000000001000200060200030001000000000000000300040000000000000000000400050000000000000000000500060000000000000000000600FFFF0200030000000000000000000300040000000000000000000400FFFF0000" : !subop.table<[pk$1 : i64, sz$1 : !db.nullable<i64>], true>
        %c_s = subop.scan %c_t : !subop.table<[pk$1 : i64, sz$1 : !db.nullable<i64>], true> {pk$1 => @pc::@pk({type = i64}), sz$1 => @pc::@sz({type = !db.nullable<i64>})}
        %mmc = subop.create !subop.multimap<[member$7 : i64], [member$8 : !db.nullable<i64>]>
        subop.insert %c_s%mmc : !subop.multimap<[member$7 : i64], [member$8 : !db.nullable<i64>]> {@pc::@sz => member$8, @pc::@pk => member$7} eq: ([%ac], [%bc]) {
          %eq = db.compare eq %ac : i64, %bc : i64
          tuples.return %eq : i1
        }
        %c_l = subop.lookup %ikeys%mmc [@k::@key] : !subop.multimap<[member$7 : i64], [member$8 : !db.nullable<i64>]> @pc_l::@list({type = !subop.list<!subop.multi_map_entry_ref<<[member$7 : i64], [member$8 : !db.nullable<i64>]>>>}) eq: ([%cc], [%dc]) {
          %eq = db.compare eq %cc : i64, %dc : i64
          tuples.return %eq : i1
        }
      }
      tuples.return %n : i64
    }
    return
  }
}
