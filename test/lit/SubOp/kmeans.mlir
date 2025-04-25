//RUN: run-mlir %s | FileCheck %s
//CHECK: |                            id  |                             x  |                             y  |
//CHECK: ----------------------------------------------------------------------------------------------------
//CHECK: |                             0  |                          1.75  |                           1.5  |
//CHECK: |                             1  |                     2.3333333  |                     4.6666665  |
//CHECK: |                             2  |                     6.6666665  |                             4  |

module{
    func.func @main(){
        %subop_result = subop.execution_group (){
            %initialCentroids = subop.create !subop.buffer<[initialClusterX : f32, initialClusterY : f32, initialClusterId : i32]>
            %numPoints = subop.create_simple_state !subop.simple_state<[numPoints: i64]> initial: {
                %c0 = db.constant(0) : i64
              tuples.return %c0 : i64
            }
            %points = subop.create !subop.buffer<[pointX : f32, pointY : f32]>
            %pointsStream, %streams:10 = subop.generate [@p::@x({type=f32}),@p::@y({type=f32})] {
              %x1 = db.constant(1) : f32
              %y1 = db.constant(1) : f32
              subop.generate_emit %x1, %y1 : f32,f32
              %x2 = db.constant(1) : f32
              %y2 = db.constant(2) : f32
              subop.generate_emit %x2, %y2 : f32,f32
              %x3 = db.constant(2) : f32
              %y3 = db.constant(1) : f32
              subop.generate_emit %x3, %y3 : f32,f32
              %x4 = db.constant(2) : f32
              %y4 = db.constant(4) : f32
              subop.generate_emit %x4, %y4 : f32,f32
              %x5 = db.constant(2) : f32
              %y5 = db.constant(5) : f32
              subop.generate_emit %x5, %y5 : f32,f32
              %x6 = db.constant(3) : f32
              %y6 = db.constant(2) : f32
              subop.generate_emit %x6, %y6 : f32,f32
              %x7 = db.constant(3) : f32
              %y7 = db.constant(5) : f32
              subop.generate_emit %x7, %y7 : f32,f32
              %x8 = db.constant(6) : f32
              %y8 = db.constant(3) : f32
              subop.generate_emit %x8, %y8 : f32,f32
              %x9 = db.constant(6) : f32
              %y9 = db.constant(5) : f32
              subop.generate_emit %x9, %y9 : f32,f32
              %x10 = db.constant(8) : f32
              %y10 = db.constant(4) : f32
              subop.generate_emit %x10, %y10 : f32,f32
              tuples.return
            }
            subop.materialize %pointsStream {@p::@x=>pointX, @p::@y => pointY}, %points: !subop.buffer<[pointX : f32, pointY : f32]>
            %numPointsRef = subop.lookup %pointsStream %numPoints[] : !subop.simple_state<[numPoints: i64]> @numPoints::@ref({type=!subop.lookup_entry_ref<!subop.simple_state<[numPoints: i64]>>})
            subop.reduce %numPointsRef @numPoints::@ref [] ["numPoints"] ([],[%currNumPoints]){
              %c1 = arith.constant 1 : i64
              %nextNumPoints = arith.addi %currNumPoints, %c1 : i64
              tuples.return %nextNumPoints : i64
            }
            %continuousPoints = subop.create_continuous_view %points : !subop.buffer<[pointX : f32, pointY : f32]> -> !subop.continuous_view<!subop.buffer<[pointX : f32, pointY : f32]>>
            %numPointsStream = subop.scan %numPoints : !subop.simple_state<[numPoints: i64]>  {numPoints => @numPoints::@value({type=i64})}

            %nested = subop.nested_map %numPointsStream [@numPoints::@value](%t, %n){
              %randomSampleStream, %streams2 = subop.generate [@generated::@id({type=index}),@generated::@idx({type=index})] {
                %k = arith.constant 3 : index
                %c0 = arith.constant 0 : index
                %c064 = arith.constant 0 : i64
                %c1 = arith.constant 1 : index
                %false = arith.constant 0 : i1

                scf.for %i = %c0 to %k step %c1 {
                  subop.generate_emit %i, %i : index, index
                }
                tuples.return
              }
              tuples.return %randomSampleStream : !tuples.tuplestream
            }
            %beginRef = subop.get_begin_ref %nested %continuousPoints : !subop.continuous_view<!subop.buffer<[pointX : f32, pointY : f32]>> @view::@begin({type=!subop.continous_entry_ref<!subop.continuous_view<!subop.buffer<[pointX : f32, pointY : f32]>>>})
            %offsetRef = subop.offset_ref_by %beginRef @view::@begin @generated::@idx @view::@ref({type=!subop.continous_entry_ref<!subop.continuous_view<!subop.buffer<[pointX : f32, pointY : f32]>>>})
            %gathered = subop.gather %offsetRef @view::@ref { pointX => @sample::@x({type=f32}),pointY => @sample::@y({type=f32}) }
            subop.materialize %gathered {@sample::@x=>initialClusterX, @sample::@y => initialClusterY, @generated::@id => initialClusterId}, %initialCentroids: !subop.buffer<[initialClusterX : f32, initialClusterY : f32, initialClusterId : i32]>


            %finalCentroids = subop.loop %initialCentroids : !subop.buffer<[initialClusterX : f32, initialClusterY : f32, initialClusterId : i32]> (%centroids) -> !subop.buffer<[clusterX : f32, clusterY : f32, clusterId : i32]> {
                    %nextCentroids = subop.create !subop.buffer<[nextClusterX : f32, nextClusterY : f32, nextClusterId : i32]>
                    %hashmap = subop.create !subop.hashmap<[centroidId : i32],[sumX : f32, sumY : f32, count : i32]>
                     %stream1 = subop.scan %points : !subop.buffer<[pointX : f32, pointY : f32]> {pointX => @point::@x({type=f32}),pointY => @point::@y({type=f32})}
                     %stream2 = subop.nested_map %stream1 [@point::@x,@point::@y](%t, %x, %y){
                          %local_best = subop.create_simple_state !subop.simple_state<[min_dist: f32, arg_min : i32]> initial: {
                             %initial_dist = db.constant(1000000000) : f32
                             %initial_id = db.constant(1000000000) : i32
                            tuples.return %initial_dist, %initial_id : f32,i32
                          }

                          %cstream = subop.scan %centroids : !subop.buffer<[clusterX : f32, clusterY : f32, clusterId : i32]> {clusterX => @cluster::@x({type=f32}),clusterY => @cluster::@y({type=f32}),clusterId => @cluster::@id({type=i32})}
                          %cstream2 = subop.map %cstream computes : [@m::@dist({type=f32})] input : [@cluster::@x, @cluster::@y] (%clusterX : f32, %clusterY : f32){
                             %diffX = arith.subf %clusterX, %x : f32
                             %diffY = arith.subf %clusterY, %y : f32
                             %diffX2 = arith.mulf %diffX, %diffX :f32
                             %diffY2 = arith.mulf %diffY, %diffY : f32
                             %dist = arith.addf %diffX2, %diffY2 : f32

                             tuples.return %dist : f32
                          }
                          %cstream3 = subop.lookup %cstream2 %local_best[] : !subop.simple_state<[min_dist: f32, arg_min : i32]> @local_best::@ref({type=!subop.lookup_entry_ref<!subop.simple_state<[min_dist: f32, arg_min : i32]>>})
                          subop.reduce %cstream3 @local_best::@ref [@m::@dist, @cluster::@id] ["min_dist","arg_min"] ([%curr_dist, %curr_id],[%min_dist,%min_id]){
                            %lt = arith.cmpf olt, %curr_dist, %min_dist : f32
                            %new_min_dist, %new_arg_min = scf.if %lt -> (f32,i32) {
                                scf.yield %curr_dist, %curr_id : f32, i32
                            } else {
                                scf.yield %min_dist, %min_id : f32 , i32
                            }
                            tuples.return %new_min_dist, %new_arg_min : f32, i32
                          }
                          %bstream = subop.scan %local_best : !subop.simple_state<[min_dist: f32, arg_min : i32]> {min_dist => @best::@dist({type=f32}),arg_min => @best::@id({type=i32})}
                        tuples.return %bstream : !tuples.tuplestream
                     }
                     %sstream3 =subop.lookup_or_insert %stream2 %hashmap[@best::@id] : !subop.hashmap<[centroidId : i32],[sumX : f32, sumY : f32, count : i32]> @aggr::@ref({type=!subop.lookup_entry_ref<!subop.hashmap<[centroidId : i32],[sumX : f32, sumY : f32, count : i32]>>})
                                            eq: ([%l], [%r]){
                                                %eq = arith.cmpi eq, %l, %r :i32
                                                tuples.return %eq : i1
                                            }
                                            initial: {
                                                %zero = arith.constant 0.0 : f32
                                                %zeroi = arith.constant 0 : i32
                                                tuples.return %zero,%zero,%zeroi : f32,f32, i32
                                            }
                      subop.reduce %sstream3 @aggr::@ref [@point::@x,@point::@y] ["sumX","sumY","count"] ([%curr_x, %curr_y],[%sum_x,%sum_y,%count]){
                        %c1 = arith.constant 1 : i32
                        %new_count = arith.addi %count, %c1 : i32
                        %new_sum_x = arith.addf %sum_x, %curr_x : f32
                        %new_sum_y = arith.addf %sum_y, %curr_y : f32
                        tuples.return %new_sum_x,%new_sum_y, %new_count : f32,f32,i32
                      }
                     %fstream = subop.scan %hashmap : !subop.hashmap<[centroidId : i32],[sumX : f32, sumY : f32, count : i32]> {centroidId => @centroid::@id({type=i32}), sumX =>@hm::@sum_x({type=i32}) , sumY =>@hm::@sum_y({type=i32}),count => @hm::@count({type=i32})}
                     %fstream1 = subop.map %fstream computes : [@centroid::@x({type=f32}),@centroid::@y({type=f32})] input: [@hm::@sum_x,@hm::@sum_y,@hm::@count] (%sum_x : f32, %sum_y : f32, %count : i32){
                        %countf = arith.sitofp %count : i32 to f32
                        %x = arith.divf %sum_x, %countf : f32
                        %y = arith.divf %sum_y, %countf : f32
                        tuples.return %x, %y : f32, f32
                     }

                     subop.materialize %fstream1 {@centroid::@id => nextClusterId, @centroid::@x => nextClusterX, @centroid::@y => nextClusterY}, %nextCentroids : !subop.buffer<[nextClusterX : f32, nextClusterY : f32, nextClusterId : i32]>
                    %changed = subop.create_simple_state !subop.simple_state<[changed :i1]> initial: {
                      %false = arith.constant 0 : i1
                      tuples.return %false : i1
                    }
                     %cstream = subop.scan %centroids : !subop.buffer<[clusterX : f32, clusterY : f32, clusterId : i32]> {clusterX => @cluster::@x({type=f32}),clusterY => @cluster::@y({type=f32}),clusterId => @cluster::@id({type=i32})}
                     %cstream2 =subop.lookup_or_insert %cstream %hashmap[@cluster::@id] : !subop.hashmap<[centroidId : i32],[sumX : f32, sumY : f32, count : i32]> @hm::@ref({type=!subop.lookup_entry_ref<!subop.hashmap<[centroidId : i32],[sumX : f32, sumY : f32, count : i32]>>})
                                            eq: ([%l], [%r]){
                                                %eq = arith.cmpi eq, %l, %r :i32
                                                tuples.return %eq : i1
                                            } initial: {
                                                                                 %zero = arith.constant 0.0 : f32
                                                                                 %zeroi = arith.constant 0 : i32
                                                                                 tuples.return %zero,%zero,%zeroi : f32,f32, i32
                                                                             }
                     %cstream3 = subop.lookup %cstream2 %changed[] :  !subop.simple_state<[changed :i1]> @changed::@ref({type=!subop.lookup_entry_ref< !subop.simple_state<[changed :i1]>>})
                     %cstream4 = subop.gather %cstream3 @hm::@ref {sumX => @hm::@sum_x({type=f32}), sumY => @hm::@sum_y({type=f32}), count => @hm::@count({type=i32})}
                      %cstream5 = subop.map %cstream4 computes : [@m::@iseq({type=i1})] input: [@cluster::@x, @cluster::@y, @hm::@sum_x, @hm::@sum_y, @hm::@count] (%old_x : f32, %old_y : f32, %sum_x : f32, %sum_y : f32, %count : i32){
                         %countf = arith.sitofp %count : i32 to f32
                         %x = arith.divf %sum_x, %countf : f32
                         %y = arith.divf %sum_y, %countf : f32
                         %xeq = arith.cmpf oeq, %x,%old_x : f32
                         %yeq = arith.cmpf oeq, %y,%old_y : f32
                         %botheq = arith.andi %xeq, %yeq : i1
                         tuples.return %botheq :i1
                      }
                      subop.reduce %cstream5 @changed::@ref [@m::@iseq] ["changed"] ([%iseq],[%has_changed]){
                        %c1 = arith.constant 1 : i1
                        %notEq = arith.xori %iseq,%c1 :i1
                        %new_has_changed = arith.ori %notEq, %has_changed : i1
                        tuples.return %new_has_changed : i1
                      }
                     subop.loop_continue (%changed : !subop.simple_state<[changed :i1]> ["changed"]) %nextCentroids : !subop.buffer<[nextClusterX : f32, nextClusterY : f32, nextClusterId : i32]>
            }
             %fstream1 = subop.scan %finalCentroids :  !subop.buffer<[clusterX : f32, clusterY : f32, clusterId : i32]>  { clusterX => @centroid::@x({type=f32}),clusterY => @centroid::@y({type=f32}), clusterId => @centroid::@id({type=i32})}
             %result_table = subop.create !subop.result_table<[id0 : i32, x0 : f32, y0 : f32]>
             subop.materialize %fstream1 {@centroid::@id => id0, @centroid::@x => x0, @centroid::@y => y0}, %result_table : !subop.result_table<[id0 : i32, x0 : f32, y0 : f32]>
            %local_table = subop.create_from ["id","x","y"] %result_table : !subop.result_table<[id0 : i32, x0 : f32, y0 : f32]> -> !subop.local_table<[id0 : i32, x0 : f32, y0 : f32],["id","x","y"]>
            subop.execution_group_return %local_table : !subop.local_table<[id0 : i32, x0 : f32, y0 : f32],["id","x","y"]>
        } -> !subop.local_table<[id0 : i32, x0 : f32, y0 : f32],["id","x","y"]>
        subop.set_result 0 %subop_result : !subop.local_table<[id0 : i32, x0 : f32, y0 : f32],["id","x","y"]>
        return
    }
}
