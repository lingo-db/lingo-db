//RUN: run-mlir %s | FileCheck %s
//CHECK: |                            id  |                          rank  |                             l  |
//CHECK: ----------------------------------------------------------------------------------------------------
//CHECK: |                             0  |                          0.03  |                             2  |
//CHECK: |                             1  |           0.31308551992225453  |                             1  |
//CHECK: |                             2  |            0.2961226919339164  |                             1  |
//CHECK: |                             4  |            0.3180417881438289  |                             1  |
//CHECK: |                             3  |          0.042749999999999996  |                             1  |
module{
    func.func @main(){
        %numVertices = subop.create_simple_state !subop.simple_state<[numVertices: index]> initial: {
             %c0 = arith.constant 0  : index
            tuples.return %c0 : index
        }
        %edges = subop.create !subop.buffer<[edgeFrom : i32, edgeTo : i32]>

        %edgeData = subop.generate [@c::@from({type=i32}),@c::@to({type=i32})] {
            %c0 = db.constant(0) : i32
            %c1 = db.constant(1) : i32
            %c2 = db.constant(2) : i32
            %c3 = db.constant(3) : i32
            %c4 = db.constant(4) : i32
            subop.generate_emit %c0, %c1 : i32,i32
            subop.generate_emit %c1, %c2 : i32,i32
            subop.generate_emit %c2, %c4 : i32,i32
            subop.generate_emit %c3, %c4 : i32,i32
            subop.generate_emit %c4, %c1 : i32,i32
            subop.generate_emit %c0, %c3 : i32,i32
            tuples.return
        }
        %vertexMapping = subop.create !subop.hashmap<[ vertexId : i32],[denseId : i32]>
        %reverseVertexMapping = subop.create !subop.hashmap<[ revDenseId : i32],[revVertexId : i32]>
        %lookedMappingUpFrom = subop.lookup_or_insert %edgeData %vertexMapping[@c::@from] : !subop.hashmap<[ vertexId : i32],[denseId : i32]> @fromMapping::@ref({type=!subop.lookup_entry_ref<!subop.hashmap<[ vertexId : i32],[denseId : i32]>>})
            eq: ([%l], [%r]){
                %eq = arith.cmpi eq, %l, %r :i32
                tuples.return %eq : i1
            }
            initial: {
                %m1 = arith.constant -1 : i32
                tuples.return %m1 : i32
            }
        %lookedMappingUpTo = subop.lookup_or_insert %lookedMappingUpFrom %vertexMapping[@c::@to] : !subop.hashmap<[ vertexId : i32],[denseId : i32]> @toMapping::@ref({type=!subop.lookup_entry_ref<!subop.hashmap<[ vertexId : i32],[denseId : i32]>>})
            eq: ([%l], [%r]){
                %eq = arith.cmpi eq, %l, %r :i32
                tuples.return %eq : i1
            }
            initial: {
                %m1 = arith.constant -1 : i32
                tuples.return %m1 : i32
            }
        %gatheredFromDenseId = subop.gather %lookedMappingUpTo @fromMapping::@ref {denseId => @from::@denseId({type=i32})}
        %gatheredToDenseId = subop.gather %gatheredFromDenseId @toMapping::@ref {denseId => @to::@denseId({type=i32})}
        %lookedUpNV = subop.lookup %gatheredToDenseId %numVertices[] : !subop.simple_state<[numVertices:index]> @numVertices::@ref({type=!subop.entry_ref<!subop.simple_state<[numVertices:index]>>})
        %gatheredNV= subop.gather %lookedUpNV @numVertices::@ref {numVertices => @numVertices::@val({type=index})}
        %newDenseIds = subop.map %gatheredNV computes: [@m::@newFromId({type=i32}),@m::@newToId({type=i32}), @m::@newNumElements({type=index})] (%tpl: !tuples.tuple){
             %numV = tuples.getcol %tpl @numVertices::@val : index
             %from = tuples.getcol %tpl @c::@from : i32
             %to = tuples.getcol %tpl @c::@to : i32
             %fromDense = tuples.getcol %tpl @from::@denseId : i32
             %toDense = tuples.getcol %tpl @to::@denseId : i32
             %c0 = arith.constant 0 : i32
             %c1i = arith.constant 1 : index
             %fromInvalid = arith.cmpi slt, %fromDense,%c0 : i32
             %toInvalid = arith.cmpi slt, %toDense,%c0 : i32
             %newFromId, %numVertices1 = scf.if %fromInvalid -> (i32, index) {
                %newFromDense = arith.index_cast %numV : index to i32
                %newNumVertices = arith.addi %numV, %c1i : index
                scf.yield %newFromDense, %newNumVertices : i32, index
             } else {
                scf.yield %fromDense, %numV : i32, index
             }
              %newToId, %numVertices2 = scf.if %toInvalid -> (i32, index) {
                 %newToDense = arith.index_cast %numVertices1 : index to i32
                 %newNumVertices = arith.addi %numVertices1, %c1i : index
                 scf.yield %newToDense, %newNumVertices : i32, index
              } else {
                 scf.yield %toDense, %numVertices1 : i32, index
              }
             tuples.return %newFromId, %newToId, %numVertices2 : i32, i32, index
        }
        subop.materialize %newDenseIds {@m::@newFromId=>edgeFrom, @m::@newToId => edgeTo}, %edges : !subop.buffer<[edgeFrom : i32, edgeTo : i32]>
        subop.scatter %newDenseIds @fromMapping::@ref {@m::@newFromId => denseId}
        subop.scatter %newDenseIds @toMapping::@ref {@m::@newToId => denseId}
        subop.scatter %newDenseIds @numVertices::@ref { @m::@newNumElements => numVertices}

        %rStream1 = subop.scan %vertexMapping : !subop.hashmap<[ vertexId : i32],[denseId : i32]> {vertexId => @vM::@vertexId({type=i32}), denseId  => @vM::@denseId({type=i32})}
        
        %rStream2 = subop.lookup_or_insert %rStream1 %reverseVertexMapping[@vM::@denseId] : !subop.hashmap<[ revDenseId : i32],[revVertexId : i32]> @rM::@ref({type=!subop.lookup_entry_ref<!subop.hashmap<[ revDenseId : i32],[revVertexId : i32]>>})
        eq: ([%l], [%r]){
            %eq = arith.cmpi eq, %l, %r :i32
            tuples.return %eq : i1
        }
        initial: {
            %c0 = arith.constant 0 : i32
            tuples.return %c0 : i32
        }
        subop.scatter %rStream2 @rM::@ref {@vM::@vertexId => revVertexId}
        
        %initialWeights = subop.create_array %numVertices : !subop.simple_state<[numVertices:index]> -> !subop.array<[initialRank : f64, initialL: i32]>
        %initialWeightsView = subop.create_continuous_view %initialWeights : !subop.array<[initialRank : f64, initialL: i32]> -> !subop.continuous_view<!subop.array<[initialRank : f64, initialL: i32]>>
        %iStream1 = subop.scan %edges :!subop.buffer<[edgeFrom : i32, edgeTo : i32]> {edgeFrom => @edge::@from1({type=i32}),edgeTo => @edge::@to1({type=i32})}
        %iStream2 = subop.get_begin_ref %iStream1 %initialWeightsView :!subop.continuous_view<!subop.array<[initialRank : f64, initialL: i32]>> @view::@begin({type=!subop.continous_entry_ref<!subop.continuous_view<!subop.array<[initialRank : f64, initialL: i32]>>>})
        %iStream3 = subop.offset_ref_by %iStream2 @view::@begin @edge::@from1 @view::@ref({type=!subop.continous_entry_ref<!subop.continuous_view<!subop.array<[initialRank : f64, initialL: i32]>>>})
        %iStream4 = subop.lookup %iStream3 %numVertices[] : !subop.simple_state<[numVertices:index]> @numVertices::@ref2({type=!subop.entry_ref<!subop.simple_state<[numVertices:index]>>})
        %iStream5 = subop.gather %iStream4 @numVertices::@ref2 {numVertices => @numVertices::@val2({type=index})}
       subop.reduce %iStream5 @view::@ref [@numVertices::@val2] ["initialRank","initialL"] ([%totalVertices],[%currRank, %currL]){
            %c1 = arith.constant 1 : i32
            %newL = arith.addi %currL, %c1 : i32
            %c1f = arith.constant 1.0 : f64
            %totalVerticesI64 = arith.index_cast %totalVertices : index to i64
            %totalVerticesf = arith.uitofp %totalVerticesI64 : i64 to f64
            %newRank = arith.divf %c1f, %totalVerticesf : f64

            tuples.return %newRank, %newL : f64, i32
        }
        %ctr = subop.create_simple_state !subop.simple_state<[ctr:i32]> initial: {
             %c0 = db.constant(0) : i32
            tuples.return %c0 : i32
        }
        %finalWeights = subop.loop %initialWeights :  !subop.array<[initialRank : f64, initialL: i32]> (%weights) ->  !subop.array<[rank : f64, l: i32]> {
                %nextWeights = subop.create_array %numVertices : !subop.simple_state<[numVertices:index]> -> !subop.array<[nextRank: f64,nextL : i32]>
                %weightsView = subop.create_continuous_view %weights : !subop.array<[rank : f64, l: i32]> -> !subop.continuous_view<!subop.array<[rank : f64, l: i32]>>
                %nextWeightsView = subop.create_continuous_view %nextWeights : !subop.array<[nextRank: f64,nextL : i32]> -> !subop.continuous_view<!subop.array<[nextRank: f64,nextL : i32]>>
                %iLStream1 = subop.scan_refs %nextWeightsView : !subop.continuous_view<!subop.array<[nextRank: f64,nextL : i32]>> @nextWeightsView::@ref({type=!subop.continous_entry_ref<!subop.continuous_view<!subop.array<[nextRank: f64,nextL : i32]>>>})
                %iLStream2 = subop.lookup %iLStream1 %numVertices[] : !subop.simple_state<[numVertices:index]> @numVertices::@ref3({type=!subop.entry_ref<!subop.simple_state<[numVertices:index]>>})
                %iLStream3 = subop.gather %iLStream2 @numVertices::@ref3 {numVertices => @numVertices::@val3({type=index})}
                %iLStream4 = subop.map %iLStream3 computes: [@m::@initialRank({type=f64})] (%tpl: !tuples.tuple){
                    %totalVertices = tuples.getcol %tpl @numVertices::@val3 : index
                    %totalVerticesI64 = arith.index_cast %totalVertices : index to i64
                    %totalVerticesf = arith.uitofp %totalVerticesI64 : i64 to f64
                    %c15 = arith.constant 0.15 : f64
                    %initialRank = arith.divf %c15,%totalVerticesf : f64
                    tuples.return %initialRank : f64
                }
                %iLStream5 = subop.get_begin_ref %iLStream4 %nextWeightsView :!subop.continuous_view<!subop.array<[nextRank : f64, nextL: i32]>> @nextWeightsView::@begin({type=!subop.continous_entry_ref<!subop.continuous_view<!subop.array<[nextRank : f64, nextL: i32]>>>})
                %iLStream6 =  subop.entries_between %iLStream5 @nextWeightsView::@begin @nextWeightsView::@ref @nextWeightsView::@id({type=index})
                %iLStream7 = subop.get_begin_ref %iLStream6 %weightsView :!subop.continuous_view<!subop.array<[rank : f64, l: i32]>> @weightsView::@begin({type=!subop.continous_entry_ref<!subop.continuous_view<!subop.array<[rank : f64, l: i32]>>>})
                %iLStream8 = subop.offset_ref_by %iLStream7 @weightsView::@begin @nextWeightsView::@id @weights::@ref({type=!subop.continous_entry_ref<!subop.continuous_view<!subop.array<[rank : f64, l: i32]>>>})
                %iLStream9 = subop.gather %iLStream8 @weights::@ref {l => @weights::@l({type=i32})}
                subop.scatter %iLStream9 @nextWeightsView::@ref { @m::@initialRank => nextRank, @weights::@l => nextL }
                %hashmap = subop.create !subop.hashmap<[centroidId : i32],[sumX : f64, sumY : f64, count : i32]>
                %stream1 = subop.scan %edges :!subop.buffer<[edgeFrom : i32, edgeTo : i32]> {edgeFrom => @edge::@from({type=i32}),edgeTo => @edge::@to({type=i32})} {attr="1"}
                %stream2 = subop.get_begin_ref %stream1 %weightsView :!subop.continuous_view<!subop.array<[rank : f64, l: i32]>> @weightsView::@begin({type=!subop.continous_entry_ref<!subop.continuous_view<!subop.array<[rank : f64, l: i32]>>>})
                %stream3 = subop.offset_ref_by %stream2 @weightsView::@begin @edge::@from @from::@ref({type=!subop.continous_entry_ref<!subop.continuous_view<!subop.array<[rank : f64, l: i32]>>>})
                %stream4 = subop.get_begin_ref %stream3 %nextWeightsView :!subop.continuous_view<!subop.array<[nextRank: f64,nextL : i32]>> @nextWeightsView::@begin({type=!subop.continous_entry_ref<!subop.continuous_view<!subop.array<[nextRank: f64,nextL : i32]>>>})
                %stream5 = subop.offset_ref_by %stream4 @nextWeightsView::@begin @edge::@to @to::@ref({type=!subop.continous_entry_ref<!subop.continuous_view<!subop.array<[nextRank: f64,nextL : i32]>>>})
                %gatheredFrom = subop.gather %stream5 @from::@ref {rank => @from::@rank({type=f64}), l => @from::@l({type=i32})}
                subop.reduce %gatheredFrom @to::@ref [@from::@rank,@from::@l] ["nextRank"] ([%currRank,%currL],[%rank]){
                    %c085 = arith.constant 0.85 : f64
                    %c1 = arith.constant 1 : i32
                    %safeL = arith.maxui %c1, %currL :i32
                    %currLF= arith.uitofp %safeL : i32 to f64
                    %toAdd = arith.divf %currRank, %currLF : f64
                    %damped= arith.mulf %toAdd, %c085 : f64
                    %newRank = arith.addf %rank, %damped : f64
                    tuples.return %newRank : f64
                }

            %20 = subop.scan_refs %ctr : !subop.simple_state<[ctr:i32]> @s::@ref({type=!subop.entry_ref<!subop.simple_state<[ctr:i32]>>})
            %21 = subop.gather %20 @s::@ref {ctr=> @s::@ctr({type=i32})}
            %s23 = subop.map %21 computes: [@m::@p1({type=i32}),@m::@continue({type=i1})] (%tpl: !tuples.tuple){
                 %ctrVal = tuples.getcol %tpl @s::@ctr : i32
                 %c1 = db.constant(1) : i32
                 %p1 = arith.addi %c1, %ctrVal : i32
                 %c5 = arith.constant 1000 : i32
                 %p1Lt5 = arith.cmpi slt, %p1, %c5 : i32
                 tuples.return %p1, %p1Lt5 : i32,i1
            }
            subop.scatter %s23 @s::@ref {@m::@p1 => ctr}
            subop.loop_continue (%s23[@m::@continue]) %nextWeights :!subop.array<[nextRank: f64,nextL : i32]>
        }
        //todo: properly implement damping factor...
        //todo: backwards mapping
         %finalWeightsView = subop.create_continuous_view %finalWeights : !subop.array<[rank : f64, l: i32]> -> !subop.continuous_view<!subop.array<[rank : f64, l: i32]>>
         %fstream1 = subop.scan_refs %finalWeightsView : !subop.continuous_view<!subop.array<[rank : f64, l: i32]>> @finalWeights::@ref({type=!subop.continous_entry_ref<!subop.continuous_view<!subop.array<[rank : f64, l: i32]>>>})
         %fstream2 = subop.gather %fstream1 @finalWeights::@ref { rank => @weights::@rank({type=f64}), l =>@weights::@l({type=i1})}
         %fstream3 = subop.get_begin_ref %fstream2 %finalWeightsView :!subop.continuous_view<!subop.array<[rank : f64, l: i32]>> @finalWeights::@begin({type=!subop.continous_entry_ref<!subop.continuous_view<!subop.array<[rank : f64, l: i32]>>>})
         %fstream4 =  subop.entries_between %fstream3 @finalWeights::@begin @finalWeights::@ref @finalWeights::@id({type=i32})

         %fstream5 = subop.lookup_or_insert %fstream4 %reverseVertexMapping[@finalWeights::@id] : !subop.hashmap<[ revDenseId : i32],[revVertexId : i32]> @rM::@ref({type=!subop.lookup_entry_ref<!subop.hashmap<[ revDenseId : i32],[revVertexId : i32]>>})
        eq: ([%l], [%r]){
            %eq = arith.cmpi eq, %l, %r :i32
            tuples.return %eq : i1
        }
        initial: {
            %c0 = arith.constant 0 : i32
            tuples.return %c0 : i32
        }
        %fstream6 = subop.gather %fstream5 @rM::@ref {revVertexId => @weights::@id({type=i32})}
         %result_table = subop.create !subop.result_table<[id0:i32, rank0 : f64, l0 :i32]>
         subop.materialize %fstream6 { @weights::@id => id0, @weights::@rank => rank0, @weights::@l => l0}, %result_table : !subop.result_table<[id0:i32,rank0 : f64, l0 :i32]>
        %local_table = subop.create_from ["id","rank","l"] %result_table : !subop.result_table<[id0:i32,rank0 : f64, l0 :i32]> -> !subop.local_table<[id0:i32,rank0 : f64, l0 :i32],["id","rank","l"]>
        subop.set_result 0 %local_table  : !subop.local_table<[id0:i32,rank0 : f64, l0 :i32],["id","rank","l"]>
        return
    }
}