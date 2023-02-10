import subop
from subop import f32,i32

initialCentroids=subop.create_buffer({"initialClusterX":f32,"initialClusterY":f32,"initialClusterId":i32})
initialCentroidStream = subop.generate_constant({"a":f32,"b":f32,"c":i32},[(1,6,0),(3,1,1),(7,2,2)])
subop.materialize(initialCentroidStream,initialCentroids,{"initialClusterX":"a","initialClusterY":"b","initialClusterId":"c"})

points = subop.create_buffer({"pointX":f32,"pointY":f32})
pointStream = subop.generate_constant({"x":f32,"y":f32},[(1,1),(1,2),(2,1),(2,4),(2,5),(3,2),(3,5),(6,3)])
subop.materialize(pointStream,points,{"pointX":"x","pointY":"y"})
#results, loop_region = subop.loop()
with subop.loop(["test"]) as args:
    nextCentroids=subop.create_buffer({"nextClusterX":f32,"nextClusterY":f32,"nextClusterId":i32})
    print(args)
    subop.ret(args)
newStream, region=subop.map(pointStream,["newCol"])
with region as tuple:
    a=tuple["a"]
    subop.ret(True)
print(newStream)