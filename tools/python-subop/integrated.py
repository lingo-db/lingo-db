import subop
from subop import f32,i32


p1=subop.pipeline(inputs={"a":i32,"b":i32},outputs={"c":i32})
with p1 as stream:
    subop.pipeline_return(stream.map(["newCol"],lambda tuple: tuple["c"].then((tuple["a"]-5)**2+(tuple["b"]+42)**2).otherwise(1)))
