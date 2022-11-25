//RUN: run-mlir %s | FileCheck %s
//CHECK: |                            id  |                             x  |                             y  |
//CHECK: ----------------------------------------------------------------------------------------------------
//CHECK: |                             1  |                          1.75  |                           1.5  |
//CHECK: |                             0  |                     2.3333333  |                     4.6666665  |
//CHECK: |                             2  |                     6.6666665  |                             4  |
module{
    func.func @main(){
        %initialCentroids = subop.create !subop.buffer<[initialClusterX : f32, initialClusterY : f32, initialClusterId : i32]>
        %initialCentroidStream = subop.generate [@c::@x({type=f32}),@c::@y({type=f32}),@c::@i({type=i32})] {
            %x1 = db.constant(1) : f32
            %y1 = db.constant(6) : f32
            %id1 = db.constant(0) : i32
            subop.generate_emit  %x1, %y1, %id1 : f32,f32,i32
            %x2 = db.constant(3) : f32
            %y2 = db.constant(1) : f32
            %id2 = db.constant(1) : i32
            subop.generate_emit %x2, %y2, %id2 : f32,f32,i32
            %x3 = db.constant(7) : f32
            %y3 = db.constant(2) : f32
            %id3 = db.constant(2) : i32
            subop.generate_emit %x3, %y3, %id3 : f32,f32,i32
            tuples.return
        }
        subop.materialize %initialCentroidStream {@c::@x=>initialClusterX, @c::@y => initialClusterY, @c::@i => initialClusterId}, %initialCentroids: !subop.buffer<[initialClusterX : f32, initialClusterY : f32, initialClusterId : i32]>

        %points = subop.create !subop.buffer<[pointX : f32, pointY : f32]>
        %pointsStream = subop.generate [@p::@x({type=f32}),@p::@y({type=f32})] {
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
                      %cstream2 = subop.map %cstream computes : [@m::@dist({type=f32})] (%tpl: !tuples.tuple){
                         %clusterX = tuples.getcol %tpl @cluster::@x : f32
                         %clusterY = tuples.getcol %tpl @cluster::@y : f32
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
                 %fstream1 = subop.map %fstream computes : [@centroid::@x({type=f32}),@centroid::@y({type=f32})] (%tpl: !tuples.tuple){
                    %sum_x = tuples.getcol %tpl @hm::@sum_x : f32
                    %sum_y = tuples.getcol %tpl @hm::@sum_y : f32
                    %count = tuples.getcol %tpl @hm::@count : i32
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
                  %cstream5 = subop.map %cstream4 computes : [@m::@iseq({type=i1})] (%tpl: !tuples.tuple){
                     %old_x = tuples.getcol %tpl @cluster::@x : f32
                     %old_y = tuples.getcol %tpl @cluster::@y : f32
                     %sum_x = tuples.getcol %tpl @hm::@sum_x : f32
                     %sum_y = tuples.getcol %tpl @hm::@sum_y : f32
                     %count = tuples.getcol %tpl @hm::@count : i32
                     %countf = arith.sitofp %count : i32 to f32
                     %x = arith.divf %sum_x, %countf : f32
                     %y = arith.divf %sum_y, %countf : f32
                     %xeq = arith.cmpf oeq, %x,%old_x : f32
                     %yeq = arith.cmpf oeq, %y,%old_y : f32
                     %botheq = arith.andi %xeq, %yeq : i1
                     tuples.return %botheq :i1
                  }
                  subop.reduce %cstream5 @changed::@ref [@m::@iseq] ["changed"] ([%iseq],[%has_changed]){
                    %new_has_changed = arith.ori %iseq, %has_changed : i1
                    tuples.return %new_has_changed : i1
                  }
                  %changed_stream = subop.scan %changed  :  !subop.simple_state<[changed :i1]> {changed => @s::@changed({type=i1})}
                 subop.loop_continue (%changed_stream [@s::@changed]) %nextCentroids : !subop.buffer<[nextClusterX : f32, nextClusterY : f32, nextClusterId : i32]>
        }
         %fstream1 = subop.scan %finalCentroids :  !subop.buffer<[clusterX : f32, clusterY : f32, clusterId : i32]>  { clusterX => @centroid::@x({type=f32}),clusterY => @centroid::@y({type=f32}), clusterId => @centroid::@id({type=i32})}
         %result_table = subop.create_result_table ["id","x","y"] -> !subop.result_table<[id0 : i32, x0 : f32, y0 : f32]>
         subop.materialize %fstream1 {@centroid::@id => id0, @centroid::@x => x0, @centroid::@y => y0}, %result_table : !subop.result_table<[id0 : i32, x0 : f32, y0 : f32]>
         subop.set_result 0 %result_table : !subop.result_table<[id0 : i32, x0 : f32, y0 : f32]>




        return
    }
}