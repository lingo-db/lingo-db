f32 = "f32"
i32 = "i32"


class Buffer:
    def __init__(self):
        pass


class TupleStream:
    def __init__(self):
        pass

    def materialize(stream, state, mapping):
        print("materialize", stream, state, mapping)

    def map(stream, computedCols, lambdaFn=None):
        if lambdaFn is not None:
            print("map", lambdaFn(Tuple()))
        else:
            print("map")
            return "newStreamVal", GenericTupleRegionOp()


class Expression:
    def __add__(self, other):
        if not isinstance(other,Expression):
            other=Constant(other)
        return ArithBinOp("+", self, other)
    def __sub__(self, other):
        if not isinstance(other,Expression):
            other=Constant(other)
        return ArithBinOp("-", self, other)
    def __mul__(self, other):
        if not isinstance(other,Expression):
            other=Constant(other)
        return ArithBinOp("*", self, other)
    def __pow__(self, power, modulo=None):
        if power==2:
            return ArithBinOp("*",self,self)
        raise NotImplementedError()
    def __i

class Constant(Expression):
    def __init__(self, const):
        self.const = const

    def __str__(self):
        return "(Const " + str(self.const) + ")"


class ArithBinOp(Expression):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

    def __str__(self):
        return "(ArithBinOp  " + str(self.op) + " " + str(self.left) + "," + str(self.right) + ")"


class Comparison(Expression):
    def __init__(self, type, left, right):
        self.type = type
        self.left = left
        self.right = right


class TupleAccess(Expression):
    def __init__(self, tuple, col):
        self.tuple = tuple
        self.col = col

    def __str__(self):
        return "(Column \"" + self.col + "\")"


class Tuple:
    def __init__(self):
        pass

    def __getitem__(self, name):
        print("getting item", name)
        return TupleAccess(self, name)


class SubOpReturnException(Exception):
    def __init__(self, args):
        self.return_vals = args


class SubOpPipelineReturnException(Exception):
    def __init__(self, stream):
        self.resultingStream = stream


class Loop:
    def __init__(self):
        self.args = []
        pass

    def __enter__(self):
        print("entering")
        return ["arg1", "arg2", "arg3"]

    def __exit__(self, type, value, traceback):
        if type is SubOpReturnException:
            print("loop return", value.return_vals)
            return True
        elif type is None:
            raise RuntimeError("return required")
        return False


class GenericTupleRegionOp:
    def __init__(self):
        pass

    def __enter__(self):
        return Tuple()

    def __exit__(self, type, value, traceback):
        if type is SubOpReturnException:
            print("return", value.return_vals)
            return True
        elif type is None:
            raise RuntimeError("return required")
        return False


class Pipeline:
    def __init__(self):
        self.initialStream = TupleStream()
        pass

    def __enter__(self):
        return self.initialStream

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is SubOpPipelineReturnException:
            self.resultingStream = exc_val.resultingStream
            print("pipeline return:", self.resultingStream)
            return True
        elif exc_type is None:
            raise RuntimeError("pipeline return required")
        return False


def pipeline(inputs, outputs):
    return Pipeline()


def loop(args):
    print("creating loop", args)
    return Loop()


def ret(args):
    raise SubOpReturnException(args)


def pipeline_return(stream):
    raise SubOpPipelineReturnException(stream)


def create_buffer(members):
    print("create_buffer", members)
    return Buffer()


def generate_constant(columns, values):
    print("generate_constant", columns, values)
    return TupleStream()
